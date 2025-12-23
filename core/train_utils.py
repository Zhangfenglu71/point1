# core/train_utils.py
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from rectified_flow.rectified_flow import RectifiedFlow
except Exception:
    RectifiedFlow = object  # type: ignore

from .config import TrainConfig


@torch.no_grad()
def eval_one_epoch(
    rf: RectifiedFlow,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    """
    Eval one epoch without building autograd graph (prevents CUDA OOM).
    Keeps AMP consistent with training when cfg.use_amp=True.
    """
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        video = batch["video"].to(device, non_blocking=True)

        # AMP for eval too (saves memory, consistent with training)
        with torch.cuda.amp.autocast(enabled=getattr(cfg, "use_amp", False)):
            loss = rf.get_loss(x_0=None, x_1=radar, y=video)

        bs = int(radar.size(0))
        total_loss += float(loss.item()) * bs
        n += bs

    # Optional: reduce fragmentation after eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return total_loss / max(n, 1)
