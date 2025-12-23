# scripts/train_video_cond_flow_film_cfg_deltavar.py
"""
FiLM + Pair-CFG training + Delta-v variance regularization (Var(||v_c - v_u||_2)).

Constraints:
- DO NOT change scripts/train_video_cond_flow_film_cfg.py behavior.
- Additive changes only.
- Keep ckpt format compatible with sampling scripts: ckpt["model"] is a state_dict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from core.config import TrainConfig as BaseConfig
from core.datasets.video_cond_radar import VideoCondRadarDataset
from core.train_utils import eval_one_epoch
from core.utils.seed import set_seed

from models.unet_video_cond_film import UNetVideoCondFiLM
from rectified_flow.rectified_flow import RectifiedFlow


# -------------------------
# Config (extends BaseConfig)
# -------------------------
@dataclass
class Config(BaseConfig):
    # IMPORTANT: use a new ckpt name to avoid overwriting any existing experiments
    ckpt_name: str = "video_cond_flow_film_cfg_deltavar_best.pth"

    # Pair-CFG training switch
    pair_cfg_train: bool = True

    # Delta-v variance regularization weight
    lambda_delta_var: float = 1e-3

    # Since we already explicitly train uncond branch, suggest small/no random drop.
    # Keep it configurable (but default smaller than baseline cfg-training).
    cond_drop_prob: float = 0.1

    # Print interval (steps)
    log_interval: int = 50


# -------------------------
# Helpers
# -------------------------
def _zeros_like_video(video: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(video)


def _delta_var_loss(v_c: torch.Tensor, v_u: torch.Tensor) -> torch.Tensor:
    """
    v_c, v_u: (B, C, H, W) (or any shape with batch first)
    d_i = ||Δv_i||_2, computed per sample over all non-batch dims
    loss_delta = Var(d_i) with unbiased=False
    """
    delta = v_c - v_u
    # per-sample L2 norm
    d = torch.sqrt(torch.sum(delta.float() ** 2, dim=tuple(range(1, delta.ndim))) + 1e-12)
    # batch variance (unbiased=False)
    return torch.var(d, unbiased=False)


# -------------------------
# Train one epoch (Pair-CFG + deltavar)
# -------------------------
def train_one_epoch_pair_cfg_deltavar(
    *,
    cfg: Config,
    rf: RectifiedFlow,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    # running meters
    sum_total = 0.0
    sum_c = 0.0
    sum_u = 0.0
    sum_dv = 0.0
    n = 0

    # interval meters
    int_total = 0.0
    int_c = 0.0
    int_u = 0.0
    int_dv = 0.0
    int_n = 0

    for step, batch in enumerate(loader):
        # --- adapt to your dataset batch keys ---
        # Common patterns: batch["radar"], batch["video"]
        radar = batch["radar"].to(device, non_blocking=True)
        video = batch["video"].to(device, non_blocking=True)

        # optional CFG random drop (kept for compatibility, but not the only uncond now)
        if cfg.cond_drop_prob > 0:
            drop_mask = (torch.rand(video.shape[0], device=device) < cfg.cond_drop_prob).view(-1, 1, 1, 1, 1)
            # video is typically (B, T, C, H, W) or (B, C, T, H, W) depending on your implementation
            # zeros_like keeps shape
            video = torch.where(drop_mask, torch.zeros_like(video), video)

        y0 = _zeros_like_video(video)

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp and scaler is not None:
            with torch.cuda.amp.autocast(True):
                # cond: sample internal (x_t, t) and get v_pred
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

                # uncond: reuse same (x_t, t) and same target (reuse_target=True)
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
                loss = loss_c + loss_u + cfg.lambda_delta_var * loss_dv

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # non-AMP path
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
            loss = loss_c + loss_u + cfg.lambda_delta_var * loss_dv

            loss.backward()
            optimizer.step()

        # update meters
        bsz = radar.shape[0]
        n += bsz
        sum_total += float(loss.detach()) * bsz
        sum_c += float(loss_c.detach()) * bsz
        sum_u += float(loss_u.detach()) * bsz
        sum_dv += float(loss_dv.detach()) * bsz

        int_n += bsz
        int_total += float(loss.detach()) * bsz
        int_c += float(loss_c.detach()) * bsz
        int_u += float(loss_u.detach()) * bsz
        int_dv += float(loss_dv.detach()) * bsz

        if (step + 1) % cfg.log_interval == 0:
            print(
                f"[train][e{epoch:03d}][{step+1:05d}/{len(loader):05d}] "
                f"loss={int_total/int_n:.6f} "
                f"(c={int_c/int_n:.6f}, u={int_u/int_n:.6f}, dv_var={int_dv/int_n:.6f}) "
                f"lambda_dv={cfg.lambda_delta_var:g} drop_p={cfg.cond_drop_prob:g}"
            )
            int_total = int_c = int_u = int_dv = 0.0
            int_n = 0

    return {
        "loss": sum_total / max(1, n),
        "loss_c": sum_c / max(1, n),
        "loss_u": sum_u / max(1, n),
        "loss_dv_var": sum_dv / max(1, n),
    }


# -------------------------
# Main
# -------------------------
def main():
    cfg = Config()

    # reproducibility
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    # dataset / loader
    train_set = VideoCondRadarDataset(
        split="train",
        radar_root=cfg.radar_root_train,
        video_root=cfg.video_root_train,
        clip_len=cfg.clip_len,
        img_size=cfg.img_size,
        preload_videos=getattr(cfg, "preload_videos", True),
    )
    val_set = VideoCondRadarDataset(
        split="val",
        radar_root=cfg.radar_root_val,
        video_root=cfg.video_root_val,
        clip_len=cfg.clip_len,
        img_size=cfg.img_size,
        preload_videos=getattr(cfg, "preload_videos", True),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = UNetVideoCondFiLM(base_ch=cfg.base_ch).to(device)

    # rectified flow wrapper (data_shape should match radar output shape excluding batch)
    # If radar is (B,1,H,W), data_shape=(1,H,W). If (B,C,H,W) likewise.
    # We'll infer from one batch.
    with torch.no_grad():
        ex = next(iter(train_loader))
        ex_radar = ex["radar"]
        data_shape = tuple(ex_radar.shape[1:])

    rf = RectifiedFlow(
        data_shape=data_shape,
        velocity_field=model,
        interp="straight",
        source_distribution="normal",
        is_independent_coupling=True,
        train_time_distribution="uniform",
        train_time_weight="uniform",
        criterion="mse",
        device=device,
        dtype=torch.float32,
    )

    # optimizer / scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp)) if device.type == "cuda" else None

    print(f"[train_video_cond_flow_film_cfg_deltavar] device={device}")
    print(
        f"[train_video_cond_flow_film_cfg_deltavar] pair_cfg_train={cfg.pair_cfg_train} "
        f"lambda_delta_var={cfg.lambda_delta_var} cond_drop_prob={cfg.cond_drop_prob} "
        f"batch_size={cfg.batch_size} amp={cfg.amp}"
    )
    print(f"[train_video_cond_flow_film_cfg_deltavar] ckpt_name={cfg.ckpt_name}")

    # training loop with best-val checkpoint
    best_val = float("inf")
    ckpt_dir = cfg.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, cfg.ckpt_name)

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch_pair_cfg_deltavar(
            cfg=cfg,
            rf=rf,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        # val: reuse your existing eval logic
        val_loss = eval_one_epoch(
            rf=rf,
            model=model,
            loader=val_loader,
            device=device,
            amp=bool(cfg.amp),
        )

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
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
                    "val_loss": float(best_val),
                },
                ckpt_path,
            )
            print(f"[train_video_cond_flow_film_cfg_deltavar] New best val_loss={best_val:.6f} saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
