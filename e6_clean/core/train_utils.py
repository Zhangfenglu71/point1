from typing import Dict

import torch
from torch.utils.data import DataLoader

from rectified_flow.rectified_flow import RectifiedFlow


def eval_one_epoch(
    rf: RectifiedFlow,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    scaler_enabled = bool(getattr(cfg, "use_amp", False))

    with torch.no_grad():
        for batch in loader:
            radar = batch["radar"].to(device)
            video = batch["video"].to(device)

            with torch.cuda.amp.autocast(enabled=scaler_enabled):
                loss = rf.get_loss(x_0=None, x_1=radar, y=video)

            bs = radar.size(0)
            total += loss.item() * bs
            n += bs

    return total / max(n, 1)

