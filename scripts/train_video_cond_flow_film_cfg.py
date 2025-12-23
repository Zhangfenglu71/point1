# scripts/train_video_cond_flow_film_cfg.py
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rectified_flow.rectified_flow import RectifiedFlow

# ✅严格复用 baseline 的 Dataset / eval / set_seed / Config
from core.config import TrainConfig as BaseConfig
from core.datasets.video_cond_radar import VideoCondRadarDataset
from core.train_utils import eval_one_epoch
from core.utils.seed import set_seed
from core.constants import ACTION_TO_ID


from models.unet_video_cond_film import UNetVideoCondFiLM


class Config(BaseConfig):
    # ✅只改 ckpt 名字 + 新增 CFG-training 参数
    ckpt_name = "video_cond_flow_film_cfg_best.pth"

    # ✅ CFG-training：condition dropout 概率
    cond_drop_prob = 0.25  # 建议 0.2~0.3；先用 0.25


def train_one_epoch_cfg(
    rf: RectifiedFlow,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int,
):
    """
    严格对齐 baseline 的 train_one_epoch：
      - batch 是 dict：batch["radar"], batch["video"]
      - AMP + GradScaler
      - loss 调用方式保持一致：rf.get_loss(x_0=None, x_1=radar, y=video)
    唯一新增：训练时对 video 做 condition dropout
    """
    model.train()
    total_loss = 0.0
    n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    print(f"[train_video_cond_flow_film_cfg] Epoch {epoch} start, total batches = {len(loader)}")

    for step, batch in enumerate(loader):
        radar = batch["radar"].to(device)  # (B,3,H,W)
        video = batch["video"].to(device)  # (B,T,3,H,W)

        # ✅ CFG-training：condition dropout（只在训练阶段）
        p = float(getattr(cfg, "cond_drop_prob", 0.0))
        if p > 0:
            drop_mask = (torch.rand(video.size(0), device=device) < p)
            if drop_mask.any():
                video = video.clone()
                video[drop_mask] = 0.0

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            loss = rf.get_loss(x_0=None, x_1=radar, y=video)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = radar.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    print(f"[train_video_cond_flow_film_cfg] device = {device}")
    print(
        f"[train_video_cond_flow_film_cfg] cfg: batch_size={cfg.batch_size}, "
        f"clip_len={cfg.clip_len}, img_size={cfg.img_size}, amp={cfg.use_amp}, "
        f"cond_drop_prob={cfg.cond_drop_prob}"
    )

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
        f"[train_video_cond_flow_film_cfg] train samples = {len(train_set)}, "
        f"val samples = {len(val_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,  # baseline 默认 0
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
    # print("[debug] fetching one batch...")
    # batch = next(iter(train_loader))
    # print("[debug] got one batch:",
    #       batch["radar"].shape, batch["video"].shape, batch.get("label", None))


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
        train_loss = train_one_epoch_cfg(
            rf, model, train_loader, optimizer, device, cfg, epoch
        )
        val_loss = eval_one_epoch(rf, model, val_loader, device, cfg)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}"
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
                f"[train_video_cond_flow_film_cfg] New best val_loss = {best_val:.6f}, "
                f"saved to {ckpt_path}"
            )

    print(f"[train_video_cond_flow_film_cfg] Done. Best val_loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
