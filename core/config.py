# core/config.py
from __future__ import annotations

# NOTE: kept as a simple attribute-style config for maximum backward compatibility.

class TrainConfig:
    device = "cuda"

    img_size = 120        # 雷达谱图 + 视频裁剪后的分辨率
    clip_len = 32         # 每个视频 clip 的帧数

    # 先用保守配置，保证能跑通
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 50
    num_workers = 0       # 多进程先关掉，稳定为主
    use_amp = True

    data_root = "data"

    ckpt_dir = "checkpoints"
    ckpt_name = "video_cond_flow_best.pth"

    seed = 2025
    time_emb_dim = 256

# Backward-compatible alias
Config = TrainConfig
