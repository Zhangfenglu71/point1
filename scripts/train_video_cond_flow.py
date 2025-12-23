# scripts/train_video_cond_flow.py
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

try:
    from rectified_flow.rectified_flow import RectifiedFlow
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        """Missing dependency: rectified_flow.
Install it (recommended):
  pip install -U git+https://github.com/lqiang67/rectified-flow.git
Or add it as a local package under your repo."""
    ) from e

from models.unet_video_cond import UNetVideoCond

# 防止 OpenCV 自己开很多线程
cv2.setNumThreads(0)

# 和前面保持一致的动作映射
ACTION_TO_ID: Dict[str, int] = {
    "box": 0,
    "jump": 1,
    "run": 2,
    "walk": 3,
}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}


class Config:
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


def set_seed(seed: int):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoCondRadarDataset(Dataset):
    """
    每个样本：
      - 一张雷达谱图 (3, H, W)
      - 对应 subject + action 的一段视频 clip (T, 3, H, W)
      - 动作 label (0~3)

    关键优化：
      1. 所有视频只解码一次：整段解码 -> 裁剪到人区域 -> resize -> 缓存在 self.video_cache
      2. __getitem__ 只在缓存里按时间截 clip，速度会比每次重新解视频快非常多
    """

    def __init__(
        self,
        root: str,
        split: str,
        img_size: int = 120,
        clip_len: int = 32,
        enable_cache: bool = True,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.clip_len = clip_len

        self.enable_cache = enable_cache
        self.video_cache: Dict[str, np.ndarray] = {}  # key: str(path), value: (T_all,H,W,3) float32 [0,1]

        self.radar_root = self.root / split / "radar"
        self.video_root = self.root / split / "video"

        if not self.radar_root.is_dir():
            raise FileNotFoundError(f"radar_root 不存在: {self.radar_root}")
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"video_root 不存在: {self.video_root}")

        # 1) 建立 (subject, action) -> [video_paths] 的索引
        self.video_index: Dict[Tuple[str, str], List[Path]] = {}
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}

        for subj_dir in sorted(self.video_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj_name = subj_dir.name  # S01, ...

            for action_dir in sorted(subj_dir.iterdir()):
                if not action_dir.is_dir():
                    continue
                action_name = action_dir.name.lower()
                if action_name not in ACTION_TO_ID:
                    continue

                paths = [
                    p
                    for p in action_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in video_exts
                ]
                if not paths:
                    print(f"[WARN] no videos in {action_dir}, skip its radars.")
                    continue

                self.video_index[(subj_name, action_name)] = sorted(paths)

        if not self.video_index:
            raise RuntimeError(
                f"[VideoCondRadarDataset] 在 {self.video_root} 下没有找到任何视频文件"
            )

        # 2) 收集所有雷达谱图路径（确保有对应的视频）
        img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.samples: List[Tuple[Path, str, str, int]] = []  # (radar_path, subj, action, label_id)

        for p in self.radar_root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in img_exts:
                continue

            rel = p.relative_to(self.radar_root).parts  # (Sxx, action, ...)
            if len(rel) < 2:
                continue
            subj_name, action_name = rel[0], rel[1].lower()

            if action_name not in ACTION_TO_ID:
                continue
            if (subj_name, action_name) not in self.video_index:
                # 理论上不应该发生
                continue

            label_id = ACTION_TO_ID[action_name]
            self.samples.append((p, subj_name, action_name, label_id))

        if not self.samples:
            raise RuntimeError(
                f"[VideoCondRadarDataset] 在 {self.radar_root} 下没有找到任何可用的雷达谱图"
            )

        print(
            f"[VideoCondRadarDataset] split={split}, "
            f"radar_root={self.radar_root}, video_root={self.video_root}, "
            f"num_samples={len(self.samples)}"
        )

        self.radar_transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),  # (3,H,W), [0,1]
            ]
        )

    # ========= 关键加速：预加载 & 缓存所有视频 =========
    def preload_all_videos(self):
        """
        把本 split 下所有视频都解码一遍，裁剪 + resize 后放进 self.video_cache。
        之后 __getitem__ 就不用再走 OpenCV 了，速度会快很多。
        """
        if not self.enable_cache:
            print(f"[VideoCondRadarDataset] cache disabled, skip preload for split={self.split}.")
            return

        print(f"[VideoCondRadarDataset] Preloading videos for split={self.split} ...")

        seen = set()
        total = 0

        for (subj, action), vlist in self.video_index.items():
            for vpath in vlist:
                key = str(vpath)
                if key in seen:
                    continue
                _ = self._get_video_frames(vpath)  # 内部会自动写入 cache
                seen.add(key)
                total += 1

        print(f"[VideoCondRadarDataset] Preloaded {total} videos into cache for split={self.split}.")

    # ========= Radar & Video 读取 =========
    def __len__(self):
        return len(self.samples)

    def _load_radar(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        radar = self.radar_transform(img)  # (3,H,W)
        return radar

    def _decode_full_video(self, video_path: Path) -> np.ndarray:
        """
        一次性把整条视频解码完，裁剪 + resize 到 (img_size,img_size)，
        返回 (T_all, H, W, 3), float32, [0,1]
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # === 利用你之前的“偏左中心裁剪”逻辑，把 1280x720 裁成方形 ===
            h, w = frame.shape[:2]
            short = min(h, w)
            top = (h - short) // 2

            if w > short:
                alpha = 0.3  # 向左偏一些
                left_float = (w - short) * alpha
                left = int(round(left_float))
                left = max(0, min(left, w - short))
            else:
                left = 0

            frame = frame[top: top + short, left: left + short]
            # ===========================================

            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames in video: {video_path}")

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T_all,H,W,3)
        return arr

    def _get_video_frames(self, video_path: Path) -> np.ndarray:
        """
        返回整个视频的 frames (T_all,H,W,3)，优先从缓存里取。
        """
        key = str(video_path)
        if self.enable_cache and key in self.video_cache:
            return self.video_cache[key]

        frames = self._decode_full_video(video_path)
        if self.enable_cache:
            self.video_cache[key] = frames
        return frames

    def _load_clip_from_one_video(self, video_path: Path) -> torch.Tensor:
        """
        从某个视频的全部 frames 里截一个长度为 clip_len 的片段：
          - 这里用“居中截取”的策略，和你之前 RealVideoRadarDataset 一致
        返回 (T,3,H,W) 的 tensor [0,1]
        """
        frames = self._get_video_frames(video_path)  # (T_all,H,W,3)
        T_all = frames.shape[0]
        T_len = self.clip_len

        if T_all <= T_len:
            start = 0
            end = T_all
        else:
            center = T_all // 2
            half = T_len // 2
            start = max(0, center - half)
            if start + T_len > T_all:
                start = T_all - T_len
            end = start + T_len

        clip = frames[start:end]  # (t,H,W,3)

        # 不足 clip_len 用最后一帧补齐
        if clip.shape[0] < T_len:
            pad_len = T_len - clip.shape[0]
            pad = np.repeat(clip[-1:, ...], pad_len, axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        # (T,H,W,3) -> (T,3,H,W)
        clip = np.transpose(clip, (0, 3, 1, 2))  # (T,3,H,W)
        return torch.from_numpy(clip)

    def __getitem__(self, idx: int):
        radar_path, subj_name, action_name, label_id = self.samples[idx]

        radar = self._load_radar(radar_path)  # (3,H,W)

        video_paths = self.video_index[(subj_name, action_name)]
        # 简单起见：随机选一条视频，再从中间截 clip
        video_path = random.choice(video_paths)

        clip = self._load_clip_from_one_video(video_path)  # (T,3,H,W)
        label = torch.tensor(label_id, dtype=torch.long)

        return {
            "radar": radar,
            "video": clip,
            "label": label,
            "subject": subj_name,
            "radar_path": str(radar_path),
            "video_path": str(video_path),
        }


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

    print(f"[train_video_cond_flow] Epoch {epoch} start, total batches = {len(loader)}")

    for step, batch in enumerate(loader):
        radar = batch["radar"].to(device)    # (B,3,H,W)
        video = batch["video"].to(device)    # (B,T,3,H,W)

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


@torch.no_grad()
def eval_one_epoch(
    rf: RectifiedFlow,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
):
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        radar = batch["radar"].to(device)
        video = batch["video"].to(device)

        loss = rf.get_loss(x_0=None, x_1=radar, y=video)

        bs = radar.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device(
        cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu"
    )
    print(f"[train_video_cond_flow] device = {device}")
    print(
    f"[train_video_cond_flow] cfg: batch_size={cfg.batch_size}, "
    f"clip_len={cfg.clip_len}, img_size={cfg.img_size}, amp={cfg.use_amp}"
)


    # ===== Dataset & Loader =====
    train_set = VideoCondRadarDataset(
        cfg.data_root, "train", img_size=cfg.img_size, clip_len=cfg.clip_len, enable_cache=True
    )
    val_set = VideoCondRadarDataset(
        cfg.data_root, "val", img_size=cfg.img_size, clip_len=cfg.clip_len, enable_cache=True
    )

    # ⚠ 一次性预解码 & 缓存所有视频（只做一次，后面训练会快很多）
    train_set.preload_all_videos()
    val_set.preload_all_videos()

    print(
        f"[train_video_cond_flow] train samples = {len(train_set)}, "
        f"val samples = {len(val_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,  # 先 0
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
    model = UNetVideoCond(
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
        train_loss = train_one_epoch(
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
                f"[train_video_cond_flow] New best val_loss = {best_val:.6f}, "
                f"saved to {ckpt_path}"
            )

    print(f"[train_video_cond_flow] Done. Best val_loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
