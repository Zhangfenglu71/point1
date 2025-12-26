from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rectified_flow.rectified_flow import RectifiedFlow

from core.config import TrainConfig
from core.datasets.video_cond_radar import VideoCondRadarDataset
from core.train_utils import eval_one_epoch
from core.utils.seed import set_seed
from models.unet_video_cond_film import UNetVideoCondFiLM


class Config(TrainConfig):
    ckpt_name: str = "video_cond_flow_film_cfg_best.pth"
    cond_drop_prob: float = 0.25


@torch.no_grad()
def _build_dataloaders(cfg: Config):
    train_set = VideoCondRadarDataset(
        cfg.data_root, "train", img_size=cfg.img_size, clip_len=cfg.clip_len, preload_all_videos=True
    )
    val_set = VideoCondRadarDataset(
        cfg.data_root, "val", img_size=cfg.img_size, clip_len=cfg.clip_len, preload_all_videos=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    rf: RectifiedFlow,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int,
):
    model.train()
    total_loss = 0.0
    n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    drop_prob = float(getattr(cfg, "cond_drop_prob", 0.0))

    for batch in loader:
        radar = batch["radar"].to(device)
        video = batch["video"].to(device)

        if drop_prob > 0:
            mask = torch.rand(video.size(0), device=device) < drop_prob
            if mask.any():
                video = video.clone()
                video[mask] = 0.0

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

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = _build_dataloaders(cfg)

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
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / cfg.ckpt_name

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(rf, model, train_loader, optimizer, device, cfg, epoch)
        val_loss = eval_one_epoch(rf, model, val_loader, device, cfg)

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
            f"[Epoch {epoch:03d}/{cfg.epochs}] train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, best={best_val:.6f}"
        )

    print(f"Training done. Best val={best_val:.6f}, ckpt -> {ckpt_path}")


if __name__ == "__main__":
    main()

