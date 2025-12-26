from typing import Tuple

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.mlp(emb)


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_term = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = self.cond_proj(cond_emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h + time_term
        h = h * (1 + gamma) + beta

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class VideoConditionEncoder(nn.Module):
    def __init__(self, base_ch: int, cond_dim: int):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, base_ch, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.SiLU(),
            nn.Conv3d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1, stride=(2, 2, 2)),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, cond_dim),
            nn.SiLU(),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, T, 3, H, W)
        x = video.permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv3d(x)
        return self.proj(x)


class UNetVideoCondFiLM(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 64,
        time_emb_dim: int = 256,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.cond_encoder = VideoConditionEncoder(base_ch=base_ch // 2, cond_dim=cond_dim)

        self.in_conv = nn.Conv2d(img_channels, base_ch, kernel_size=3, padding=1)

        chs = [base_ch, base_ch * 2, base_ch * 4]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, ch in enumerate(chs):
            in_ch = chs[i - 1] if i > 0 else base_ch
            block = FiLMResidualBlock(in_ch, ch, time_emb_dim, cond_dim)
            self.down_blocks.append(block)
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.mid_block1 = FiLMResidualBlock(chs[-1], chs[-1], time_emb_dim, cond_dim)
        self.mid_block2 = FiLMResidualBlock(chs[-1], chs[-1], time_emb_dim, cond_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for i, ch in enumerate(rev_chs):
            skip_ch = ch
            in_ch = ch + skip_ch
            block = FiLMResidualBlock(in_ch, ch, time_emb_dim, cond_dim)
            self.up_blocks.append(block)
            if i < len(rev_chs) - 1:
                self.upsamples.append(
                    nn.ConvTranspose2d(ch, rev_chs[i + 1], kernel_size=4, stride=2, padding=1)
                )

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W), t: (B,), y: (B,T,3,H,W)
        t_emb = self.time_embedding(t)
        cond_emb = self.cond_encoder(y)

        h = self.in_conv(x)

        skips = []
        cur = h
        for i, block in enumerate(self.down_blocks):
            cur = block(cur, t_emb, cond_emb)
            skips.append(cur)
            if i < len(self.downsamples):
                cur = self.downsamples[i](cur)

        cur = self.mid_block1(cur, t_emb, cond_emb)
        cur = self.mid_block2(cur, t_emb, cond_emb)

        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if cur.shape[2:] != skip.shape[2:]:
                cur = torch.nn.functional.interpolate(cur, size=skip.shape[2:], mode="nearest")
            cur = torch.cat([cur, skip], dim=1)
            cur = block(cur, t_emb, cond_emb)
            if i < len(self.upsamples):
                cur = self.upsamples[i](cur)

        out = self.out_norm(cur)
        out = self.out_act(out)
        out = self.out_conv(out)
        return out
