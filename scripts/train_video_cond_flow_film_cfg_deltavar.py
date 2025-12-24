# scripts/train_video_cond_flow_film_cfg_deltavar.py
"""
FiLM + CFG-training baseline extension:

  - Pair-CFG training: for the same (x_t, t, v_target), compute BOTH conditional and unconditional RF losses.
  - Δv stability regularization: Var(||Δv||_2) where Δv = v_c - v_u, computed on the SAME (x_t, t).

Design constraints (must hold):
  1) scripts/train_video_cond_flow_film_cfg.py is untouched and keeps identical behavior.
  2) Only additive changes: new script, new ckpt name.
  3) RectifiedFlow.get_loss is extended in a backward compatible way (defaults unchanged).
  4) Reuse existing components: BaseConfig / Dataset / eval_one_epoch / set_seed / UNetVideoCondFiLM.
  5) AMP / AdamW / best-val checkpoint logic stays aligned with baseline.
  6) Produced ckpt is compatible with existing sampling scripts: ckpt["model"] is a state_dict.
"""
import os, sys

# Force using local rectified-flow implementation in this repo (avoid picking up other editable installs)
RF_LOCAL = "/home/zfl/code/point2/rectified-flow"
if os.path.isdir(RF_LOCAL) and RF_LOCAL not in sys.path:
    sys.path.insert(0, RF_LOCAL)

from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from rectified_flow.rectified_flow import RectifiedFlow
import rectified_flow.rectified_flow as rf_mod
print("[debug] RectifiedFlow file:", rf_mod.__file__)

# ✅严格复用 baseline 的 Dataset / eval / set_seed / Config
from core.config import TrainConfig as BaseConfig
from core.datasets.video_cond_radar import VideoCondRadarDataset
from core.train_utils import eval_one_epoch
from core.utils.seed import set_seed

from models.unet_video_cond_film import UNetVideoCondFiLM


# class Config(BaseConfig):
#     # ✅新 ckpt 名字，绝不覆盖旧实验
#     ckpt_name = "video_cond_flow_film_cfg_deltavar_best.pth"

#     # Pair-CFG training
#     pair_cfg_train = True

#     # Δv 稳定正则系数
#     lambda_delta_var = 1e-3

#     # 仍保留 CFG-training 的随机 dropout，但默认更小（因为我们显式训练 uncond）
#     cond_drop_prob = 0.1

#     # 每 N step 打印一下分项 loss（便于确认正则在起作用）
#     log_interval = 50

#     # 建议默认 batch_size 更保守一点，方便先跑通（不影响原脚本）
#     batch_size = 64
import os

class Config(BaseConfig):
    # ✅新 ckpt 名字，默认不覆盖旧实验（但允许外部覆盖）
    ckpt_name = os.getenv("CKPT_NAME", "video_cond_flow_film_cfg_deltavar_best.pth")

    # ✅Pair-CFG training（一般不扫，固定 True）
    pair_cfg_train = True

    # ✅Δv 稳定正则（要扫）
    lambda_delta_var = float(os.getenv("LAMBDA_DELTA_VAR", "1e-3"))

    # ✅CFG-training dropout（要扫）
    cond_drop_prob = float(os.getenv("COND_DROP_PROB", "0.1"))

    # ✅打印频率（不扫也行，但保留）
    log_interval = int(os.getenv("LOG_INTERVAL", "50"))

    # ✅batch/epoch/seed（可选扫）
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    epochs = int(os.getenv("EPOCHS", str(getattr(BaseConfig, "epochs", 50))))
    seed = int(os.getenv("SEED", str(getattr(BaseConfig, "seed", 0))))


def _delta_var_loss(v_c: torch.Tensor, v_u: torch.Tensor) -> torch.Tensor:
    """loss_delta = Var(d_i), d_i = ||Δv_i||_2, unbiased=False"""
    delta = (v_c - v_u).float()
    # per-sample L2 norm over non-batch dims
    dims = tuple(range(1, delta.ndim))
    d = torch.sqrt(torch.sum(delta * delta, dim=dims) + 1e-12)
    return torch.var(d, unbiased=False)


def train_one_epoch_pair_cfg_deltavar(
    rf: RectifiedFlow,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int,
) -> Dict[str, float]:
    """
    Pair training on SAME (x_t, t, v_target):

      loss = loss_c + loss_u + lambda * Var(||v_c - v_u||_2)

    - cond branch samples (t, x0, xt) internally and returns v_pred + (xt,t,v_target)
    - uncond branch reuses external_xt/external_t and reuse_target=True
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    sum_total = 0.0
    sum_c = 0.0
    sum_u = 0.0
    sum_dv = 0.0
    n = 0

    int_total = 0.0
    int_c = 0.0
    int_u = 0.0
    int_dv = 0.0
    int_n = 0

    print(f"[train_video_cond_flow_film_cfg_deltavar] Epoch {epoch} start, total batches = {len(loader)}")

    for step, batch in enumerate(loader):
        radar = batch["radar"].to(device)  # (B,3,H,W)
        video = batch["video"].to(device)  # (B,T,3,H,W)

        # Optional: keep small random condition dropout (CFG-training) to retain robustness
        p = float(getattr(cfg, "cond_drop_prob", 0.0))
        if p > 0:
            drop_mask = (torch.rand(video.size(0), device=device) < p)
            if drop_mask.any():
                video = video.clone()
                video[drop_mask] = 0.0

        y0 = torch.zeros_like(video)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # 1) cond branch: sample xt/t inside and return details
            loss_c, det_c = rf.get_loss(
                x_0=None,
                x_1=radar,
                y=video,
                return_pred=True,
                return_xt=True,
            )
            x_t = det_c["x_t"]
            t = det_c["t"]
            v_c = det_c["v_pred"]

            # 2) uncond branch: reuse same xt/t AND same v_target
            loss_u, det_u = rf.get_loss(
                x_0=None,
                x_1=radar,
                y=y0,
                external_xt=x_t,
                external_t=t,
                reuse_target=True,
                return_pred=True,
                return_xt=False,
            )
            v_u = det_u["v_pred"]

            loss_dv = _delta_var_loss(v_c, v_u)
            loss = loss_c + loss_u + float(cfg.lambda_delta_var) * loss_dv

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = int(radar.size(0))
        n += bs
        sum_total += float(loss.item()) * bs
        sum_c += float(loss_c.item()) * bs
        sum_u += float(loss_u.item()) * bs
        sum_dv += float(loss_dv.item()) * bs

        int_n += bs
        int_total += float(loss.item()) * bs
        int_c += float(loss_c.item()) * bs
        int_u += float(loss_u.item()) * bs
        int_dv += float(loss_dv.item()) * bs

        if (step + 1) % int(getattr(cfg, "log_interval", 50)) == 0:
            print(
                f"[train_video_cond_flow_film_cfg_deltavar] e{epoch:03d} "
                f"[{step+1:05d}/{len(loader):05d}] "
                f"loss={int_total/int_n:.6f} "
                f"(c={int_c/int_n:.6f}, u={int_u/int_n:.6f}, dv_var={int_dv/int_n:.6f}) "
                f"lambda_dv={float(cfg.lambda_delta_var):g} drop_p={float(cfg.cond_drop_prob):g}"
            )
            int_total = int_c = int_u = int_dv = 0.0
            int_n = 0

    return {
        "loss": sum_total / max(n, 1),
        "loss_c": sum_c / max(n, 1),
        "loss_u": sum_u / max(n, 1),
        "loss_dv_var": sum_dv / max(n, 1),
    }


def main():
    cfg = Config()
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    print(f"[train_video_cond_flow_film_cfg_deltavar] device = {device}")
    print(
        f"[train_video_cond_flow_film_cfg_deltavar] cfg: batch_size={cfg.batch_size}, "
        f"clip_len={cfg.clip_len}, img_size={cfg.img_size}, amp={cfg.use_amp}, "
        f"pair_cfg_train={cfg.pair_cfg_train}, lambda_delta_var={cfg.lambda_delta_var}, "
        f"cond_drop_prob={cfg.cond_drop_prob}"
    )
    print(f"[train_video_cond_flow_film_cfg_deltavar] ckpt_name={cfg.ckpt_name}")

    # ===== Dataset & Loader（严格复用 baseline）=====
    train_set = VideoCondRadarDataset(
        cfg.data_root, "train", img_size=cfg.img_size, clip_len=cfg.clip_len, enable_cache=True
    )
    val_set = VideoCondRadarDataset(
        cfg.data_root, "val", img_size=cfg.img_size, clip_len=cfg.clip_len, enable_cache=True
    )

    train_set.preload_all_videos()
    val_set.preload_all_videos()

    print(
        f"[train_video_cond_flow_film_cfg_deltavar] train samples = {len(train_set)}, "
        f"val samples = {len(val_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # ===== Model & RectifiedFlow =====
    model = UNetVideoCondFiLM(
        img_channels=3,
        base_ch=64,
        time_emb_dim=cfg.time_emb_dim,
    ).to(device)

    rf = RectifiedFlow(
        data_shape=(3, cfg.img_size, cfg.img_size),
        velocity_field=model,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg.ckpt_name

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch_pair_cfg_deltavar(
            rf, model, train_loader, optimizer, device, cfg, epoch
        )
        val_loss = eval_one_epoch(rf, model, val_loader, device, cfg)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_stats['loss']:.6f} "
            f"(c={train_stats['loss_c']:.6f}, u={train_stats['loss_u']:.6f}, dv_var={train_stats['loss_dv_var']:.6f}) "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                ckpt_path,
            )
            print(
                f"[train_video_cond_flow_film_cfg_deltavar] New best val_loss = {best_val:.6f}, saved to {ckpt_path}"
            )

    print(f"[train_video_cond_flow_film_cfg_deltavar] Done. Best val_loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
