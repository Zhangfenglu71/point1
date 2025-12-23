# models/unet_video_cond.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== 时间嵌入 =====
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) 或 (B,1)，值在 [0,1] 区间
    return: (B, dim)
    """
    if t.dim() == 2:
        t = t.squeeze(-1)
    assert t.dim() == 1, f"t shape should be (B,) or (B,1), got {t.shape}"

    half_dim = dim // 2
    emb_scale = math.log(10000.0) / (half_dim - 1)
    # (half_dim,)
    freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
    # (B, half_dim)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half_dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))  # 补一维
    return emb  # (B, dim)


class TimeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


# ===== 3D CNN 视频编码器 =====
class VideoEncoder3D(nn.Module):
    """
    输入: (B, T, 3, H, W) 或 (B, 3, T, H, W)
    输出: (B, out_dim)
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),   # 时序不降采样，只在空间上 pooling

            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, 3, H, W) or (B, 3, T, H, W)
        """
        if video.dim() != 5:
            raise ValueError(f"video must be 5D, got {video.shape}")

        if video.shape[2] == 3:
            # (B, T, 3, H, W) -> (B, 3, T, H, W)
            video = video.permute(0, 2, 1, 3, 4).contiguous()
        elif video.shape[1] == 3:
            # already (B, 3, T, H, W)
            pass
        else:
            raise ValueError(
                f"video shape should be (B,T,3,H,W) or (B,3,T,H,W), got {video.shape}"
            )

        x = self.net(video)  # (B, C, T', H', W')
        x = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1))  # (B, C,1,1,1)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.proj(x)           # (B, out_dim)
        return x


# ===== U-Net 基础模块 =====
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        groups = min(8, out_ch)
        assert out_ch % groups == 0, "out_ch must be divisible by num_groups"

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W)
        # emb: (B, emb_dim)
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # 注入条件（时间 + 视频）
        e = self.emb_proj(emb)  # (B, out_ch)
        h = h + e[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h


class UNetVideoCond(nn.Module):
    """
    视频条件 Rectified Flow 的速度场：
      输入:
        x: 雷达谱图 (B, 3, H, W)
        t: 时间标量 (B,) or (B,1)，范围通常 [0,1]
        y: 视频 clip，形状 (B, T, 3, H_v, W_v)
      输出:
        v(x,t,y): (B, 3, H, W)
    """

    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 64,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.base_ch = base_ch
        self.time_emb_dim = time_emb_dim

        # 时间 + 视频 embedding
        self.time_mlp = TimeMLP(time_emb_dim)
        self.video_encoder = VideoEncoder3D(time_emb_dim)

        # U-Net 结构：3 层下采样 + bottleneck + 3 层上采样
        ch = base_ch
        self.enc1 = ConvBlock(img_channels, ch, time_emb_dim)      # 120x120
        self.down1 = nn.MaxPool2d(kernel_size=2)                   # -> 60x60

        self.enc2 = ConvBlock(ch, ch * 2, time_emb_dim)            # 60x60
        self.down2 = nn.MaxPool2d(kernel_size=2)                   # -> 30x30

        self.enc3 = ConvBlock(ch * 2, ch * 4, time_emb_dim)        # 30x30
        self.down3 = nn.MaxPool2d(kernel_size=2)                   # -> 15x15

        self.bottleneck = ConvBlock(ch * 4, ch * 4, time_emb_dim)  # 15x15

        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, kernel_size=2, stride=2)  # 15->30
        self.dec3 = ConvBlock(ch * 4 + ch * 4, ch * 2, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, kernel_size=2, stride=2)  # 30->60
        self.dec2 = ConvBlock(ch * 2 + ch * 2, ch, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)          # 60->120
        self.dec1 = ConvBlock(ch + ch, ch, time_emb_dim)

        self.out_conv = nn.Conv2d(ch, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None) -> torch.Tensor:
        """
        y: 这里直接当作视频 clip 使用，形状 (B, T, 3, H, W)
        """
        if y is None:
            raise ValueError("UNetVideoCond 需要视频条件 y (视频 clip)，不能为 None")

        video = y

        # ===== 时间 + 视频 embedding =====
        # t -> (B,)
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)   # (B, D)
        t_emb = self.time_mlp(t_emb)                              # (B, D)

        v_emb = self.video_encoder(video)                         # (B, D)

        emb = t_emb + v_emb                                       # (B, D)

        # ===== U-Net 前向 =====
        e1 = self.enc1(x, emb)
        p1 = self.down1(e1)

        e2 = self.enc2(p1, emb)
        p2 = self.down2(e2)

        e3 = self.enc3(p2, emb)
        p3 = self.down3(e3)

        b = self.bottleneck(p3, emb)

        # up 3
        u3 = self.up3(b)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1), emb)

        # up 2
        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), emb)

        # up 1
        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), emb)

        out = self.out_conv(d1)
        return out
