from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import SDESampler

from core.config import TrainConfig
from models.unet_video_cond_film import UNetVideoCondFiLM

cv2.setNumThreads(0)

ACTIONS = ["box", "jump", "run", "walk"]
BASELINE_CFG_W = 3.0
BASELINE_OUT_DIR = "samples_baseline_e6_w3"


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VelocityCFG(nn.Module):
    def __init__(self, base: nn.Module, cfg_w: float):
        super().__init__()
        self.base = base
        self.cfg_w = float(cfg_w)

    def forward(self, x, t, y=None, **kwargs):
        assert y is not None
        if abs(self.cfg_w - 1.0) < 1e-8:
            return self.base(x, t, y=y, **kwargs)
        y0 = torch.zeros_like(y)
        v_u = self.base(x, t, y=y0, **kwargs)
        v_c = self.base(x, t, y=y, **kwargs)
        return v_u + self.cfg_w * (v_c - v_u)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Optional[Path] = None) -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _iso_from_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


class TestVideoDatasetAligned(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, clip_len: int, img_size: int, alpha: float):
        self.root = Path(root)
        self.split = split
        self.clip_len = int(clip_len)
        self.img_size = int(img_size)
        self.alpha = float(alpha)

        self.video_root = self.root / split / "video"
        if not self.video_root.is_dir():
            raise FileNotFoundError(self.video_root)
        exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
        self.samples: List[Tuple[Path, str, str]] = []
        for subj_dir in sorted(self.video_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            subj = subj_dir.name
            for act_dir in sorted(subj_dir.iterdir()):
                if not act_dir.is_dir():
                    continue
                action = act_dir.name.lower()
                for f in sorted(act_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in exts:
                        self.samples.append((f, subj, action))
        if not self.samples:
            raise RuntimeError("No test videos found")

    def __len__(self):
        return len(self.samples)

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        short = min(h, w)
        top = (h - short) // 2
        if w > short:
            left = int(round((w - short) * self.alpha))
            left = max(0, min(left, w - short))
        else:
            left = 0
        return frame[top : top + short, left : left + short]

    def _load_clip(self, path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        T = self.clip_len
        if frame_count <= 0:
            start, need = 0, T
        elif frame_count <= T:
            start, need = 0, frame_count
        else:
            center = frame_count // 2
            half = T // 2
            start = max(0, center - half)
            if start + T > frame_count:
                start = frame_count - T
            need = T

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: List[np.ndarray] = []
        last: Optional[np.ndarray] = None
        for _ in range(need):
            ok, frame = cap.read()
            if not ok:
                if last is None:
                    last = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(last.copy())
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self._crop(frame)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
            last = frame
        cap.release()

        while len(frames) < T:
            if last is None:
                last = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frames.append(last.copy())

        return np.stack(frames, axis=0)

    def first_by_subj_action(self) -> Dict[str, Dict[str, Path]]:
        m: Dict[str, Dict[str, Path]] = {}
        for p, subj, act in self.samples:
            m.setdefault(subj, {})
            m[subj].setdefault(act, p)
        return m


def _sample_images(rf: RectifiedFlow, y: torch.Tensor, num_steps: int, seed: int, B: int) -> torch.Tensor:
    sampler = SDESampler(rectified_flow=rf)
    sampler.sample_loop(num_samples=B, num_steps=num_steps, seed=seed, y=y)
    return sampler.trajectories[-1].clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--grid_n_per_action", type=int, default=4)
    parser.add_argument("--eval_n_per_action", type=int, default=28)
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_cond_flow_film_cfg_best.pth")
    args = parser.parse_args()

    train_cfg = TrainConfig()
    device = torch.device(
        train_cfg.device if torch.cuda.is_available() and "cuda" in str(train_cfg.device) else "cpu"
    )

    set_deterministic(args.seed)

    out_dir = Path(BASELINE_OUT_DIR)
    grid_dir = out_dir / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    ckpt_stat = ckpt_path.stat()

    train_cfg_from_ckpt = None
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        c = ckpt["cfg"]
        try:
            train_cfg_from_ckpt = dict(c)
        except Exception:
            try:
                train_cfg_from_ckpt = vars(c)
            except Exception:
                train_cfg_from_ckpt = str(c)

    meta = {
        "name": "baseline_e6_w3",
        "cfg_w": BASELINE_CFG_W,
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "seed_rule": "seed = base_seed + subj_idx*1000 + act_idx*10",
        "alpha": float(args.alpha),
        "grid_n_per_action": int(args.grid_n_per_action),
        "eval_n_per_action": int(args.eval_n_per_action),
        "ckpt": {
            "path": str(ckpt_path.resolve()),
            "sha256": _sha256(ckpt_path),
            "mtime_utc": _iso_from_mtime(ckpt_stat.st_mtime),
            "size_bytes": int(ckpt_stat.st_size),
        },
        "train_cfg_from_ckpt": train_cfg_from_ckpt,
        "train_cfg_runtime": train_cfg.__dict__,
        "git_commit": _git_commit(),
        "generated_summary": None,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    ds = TestVideoDatasetAligned(
        root=train_cfg.data_root,
        split="test",
        clip_len=train_cfg.clip_len,
        img_size=train_cfg.img_size,
        alpha=args.alpha,
    )
    subj2act = ds.first_by_subj_action()
    subjects = sorted(subj2act.keys())

    base = UNetVideoCondFiLM(img_channels=3, base_ch=64, time_emb_dim=train_cfg.time_emb_dim)
    base.load_state_dict(ckpt["model"], strict=True)
    base.to(device).eval()

    model = VelocityCFG(base, cfg_w=BASELINE_CFG_W).to(device).eval()

    rf = RectifiedFlow(
        data_shape=(3, train_cfg.img_size, train_cfg.img_size),
        velocity_field=model,
        device=device,
    )

    generated = {
        "subjects_total_found": int(len(subjects)),
        "subjects_done": 0,
        "subjects_skipped_missing_actions": 0,
        "images_saved_eval": 0,
        "images_saved_grid": 0,
    }

    for subj_idx, subj in enumerate(subjects):
        if subj not in subj2act:
            generated["subjects_skipped_missing_actions"] += 1
            continue
        if not all(a in subj2act[subj] for a in ACTIONS):
            generated["subjects_skipped_missing_actions"] += 1
            continue

        eval_imgs_by_action = {}
        for act_idx, action in enumerate(ACTIONS):
            vp = subj2act[subj][action]
            clip_np = ds._load_clip(vp)
            clip = torch.from_numpy(clip_np).float() / 255.0
            clip = clip.permute(0, 3, 1, 2).contiguous()
            video = clip.unsqueeze(0).to(device)

            B_eval = int(args.eval_n_per_action)
            y = video.expand(B_eval, -1, -1, -1, -1).contiguous()

            seed = args.seed + subj_idx * 1000 + act_idx * 10
            imgs = _sample_images(rf, y=y, num_steps=args.num_steps, seed=seed, B=B_eval)
            eval_imgs_by_action[action] = imgs

            for k, img in enumerate(imgs):
                save_image(img, out_dir / f"{action}_{subj}_{vp.stem}_w{BASELINE_CFG_W:g}_{k:03d}.png")
            generated["images_saved_eval"] += B_eval

        grid_all = []
        B_grid = int(args.grid_n_per_action)
        for action in ACTIONS:
            imgs = eval_imgs_by_action[action]
            grid_all.append(imgs[:B_grid])
        grid_imgs = torch.cat(grid_all, dim=0)
        save_image(grid_imgs, grid_dir / f"grid_{subj}_w{BASELINE_CFG_W:g}.png", nrow=B_grid)
        generated["images_saved_grid"] += len(ACTIONS) * B_grid

        generated["subjects_done"] += 1

    meta = json.loads(meta_path.read_text())
    meta["generated_summary"] = generated
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"Sampling done. Outputs -> {out_dir}")


if __name__ == "__main__":
    main()

