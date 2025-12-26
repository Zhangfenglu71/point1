import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image


class VideoCondRadarDataset(Dataset):
    """Load paired radar spectrograms and conditioning video clips."""

    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: int = 120,
        clip_len: int = 64,
        preload_all_videos: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = int(img_size)
        self.clip_len = int(clip_len)
        self.preload_all_videos = preload_all_videos

        self.radar_root = self.data_root / split / "radar"
        self.video_root = self.data_root / split / "video"
        if not self.radar_root.is_dir():
            raise FileNotFoundError(f"radar root not found: {self.radar_root}")
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"video root not found: {self.video_root}")

        self.samples: List[Tuple[Path, Path, str, str]] = []
        for subj_dir in sorted(self.radar_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj = subj_dir.name
            for act_dir in sorted(subj_dir.iterdir()):
                if not act_dir.is_dir():
                    continue
                action = act_dir.name
                radar_files = sorted(p for p in act_dir.iterdir() if p.suffix.lower() == ".png")
                for radar_path in radar_files:
                    base = radar_path.stem
                    video_path = self.video_root / subj / action / f"{base}.mp4"
                    if not video_path.exists():
                        continue
                    self.samples.append((radar_path, video_path, subj, action))

        if not self.samples:
            raise RuntimeError(f"No paired radar/video samples found under {self.data_root}")

        self.video_cache: Dict[Path, torch.Tensor] = {}
        self.radar_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        if video_path in self.video_cache:
            return self.video_cache[video_path]

        frames, _, _ = read_video(str(video_path), output_format="TCHW")
        frames = frames.float() / 255.0  # (T, C, H, W)

        if frames.size(1) != 3:
            if frames.size(1) == 1:
                frames = frames.repeat(1, 3, 1, 1)
            else:
                frames = frames[:, :3]

        frames = torch.nn.functional.interpolate(
            frames, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

        if self.preload_all_videos:
            self.video_cache[video_path] = frames

        return frames

    def _select_clip(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.size(0)
        L = self.clip_len
        if T <= L:
            if T == 0:
                pad = torch.zeros((L, 3, self.img_size, self.img_size), device=frames.device)
                return pad
            repeat = [frames[-1].unsqueeze(0)] * (L - T)
            frames = torch.cat([frames, *repeat], dim=0)
            T = frames.size(0)

        if self.split == "train":
            start = random.randint(0, T - L)
        else:
            start = max(0, (T - L) // 2)
        return frames[start : start + L]

    def preload_all_videos(self) -> None:
        for _, video_path, _, _ in self.samples:
            if video_path not in self.video_cache:
                self._load_video_frames(video_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        radar_path, video_path, subj, action = self.samples[idx]

        radar_img = Image.open(radar_path).convert("RGB")
        radar = self.radar_transform(radar_img)

        video_frames = self._load_video_frames(video_path)
        clip = self._select_clip(video_frames).contiguous()

        sample = {
            "radar": radar,
            "video": clip,
            "subject": subj,
            "action": action,
        }
        return sample

