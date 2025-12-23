# models/unet_video_cond_film.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_video_cond import (
    sinusoidal_time_embedding,
    TimeMLP,
    VideoEncoder3D,
)

class ConvBlockFiLM(nn.Module):
    """
    ConvBlock + FiLM：h = h * (1 + gamma) + beta
    稳定策略：
      - emb_proj 的 bias=0；weight 小初始化（避免一上来调制过猛）
      - gamma 用 tanh + 缩放（避免 1+gamma 变得过大/为负太多）
      - emb 强制 float32，再 cast 回当前 dtype（AMP 更稳）
    """
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, gamma_scale: float = 0.1):
        super().__init__()
        groups = min(8, out_ch)
        assert out_ch % groups == 0, "out_ch must be divisible by num_groups"

        self.gamma_scale = float(gamma_scale)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)

        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)

        # ✅稳起步：bias=0，weight 很小
        nn.init.zeros_(self.emb_proj.bias)
        nn.init.normal_(self.emb_proj.weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        # ✅ AMP 更稳：emb 用 fp32 过线性层，再转回 h.dtype
        gb = self.emb_proj(emb.float()).to(h.dtype)   # (B, 2*out_ch)
        gamma, beta = gb.chunk(2, dim=1)

        # ✅ 限幅：避免 gamma 过大导致数值不稳
        gamma = self.gamma_scale * torch.tanh(gamma)

        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h



# models/unet_video_cond_film.py

class UNetVideoCondFiLM(nn.Module):
    def __init__(
        self,
        img_channels: int = 3,
        base_ch: int = 64,
        time_emb_dim: int = 256,
        num_actions: int = 4,   # ✅ 新增
    ):
        super().__init__()
        self.img_channels = img_channels
        self.base_ch = base_ch
        self.time_emb_dim = time_emb_dim

        # ✅ action embedding（含 null）
        self.num_actions = int(num_actions)
        self.null_action_id = self.num_actions
        self.action_embed = nn.Embedding(self.num_actions + 1, time_emb_dim)

        self.time_mlp = TimeMLP(time_emb_dim)
        self.video_encoder = VideoEncoder3D(time_emb_dim)

        ch = base_ch
        self.enc1 = ConvBlockFiLM(img_channels, ch, time_emb_dim)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlockFiLM(ch, ch * 2, time_emb_dim)
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlockFiLM(ch * 2, ch * 4, time_emb_dim)
        self.down3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlockFiLM(ch * 4, ch * 4, time_emb_dim)

        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlockFiLM(ch * 4 + ch * 4, ch * 2, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlockFiLM(ch * 2 + ch * 2, ch, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlockFiLM(ch + ch, ch, time_emb_dim)

        self.out_conv = nn.Conv2d(ch, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None, a=None) -> torch.Tensor:
        """
        y: video clip, (B,T,3,H,W)
        a: action id, (B,) in {0..3} or null_id(4)
        """
        if y is None:
            raise ValueError("UNetVideoCondFiLM 需要视频条件 y，不能为 None")

        if t.dim() == 2:
            t = t.squeeze(-1)

        B = x.size(0)

        # ✅ 默认：如果不给 a，就当作 null（用于兼容旧代码）
        if a is None:
            a = torch.full((B,), self.null_action_id, device=x.device, dtype=torch.long)
        else:
            if a.dim() == 2:
                a = a.squeeze(-1)
            a = a.to(device=x.device, dtype=torch.long)
            # 防御：越界的都当 null
            a = torch.where(
                (a >= 0) & (a < self.num_actions),
                a,
                torch.full_like(a, self.null_action_id),
            )

        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        v_emb = self.video_encoder(y)

        a_emb = self.action_embed(a)

        emb = t_emb + v_emb + a_emb

        e1 = self.enc1(x, emb); p1 = self.down1(e1)
        e2 = self.enc2(p1, emb); p2 = self.down2(e2)
        e3 = self.enc3(p2, emb); p3 = self.down3(e3)

        b = self.bottleneck(p3, emb)

        u3 = self.up3(b)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1), emb)

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), emb)

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), emb)

        return self.out_conv(d1)
    def _forward_with_emb(self, x, emb):
        e1 = self.enc1(x, emb); p1 = self.down1(e1)
        e2 = self.enc2(p1, emb); p2 = self.down2(e2)
        e3 = self.enc3(p2, emb); p3 = self.down3(e3)

        b = self.bottleneck(p3, emb)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1), emb)

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), emb)

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1), emb)

        return self.out_conv(d1)

