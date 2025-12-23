# scripts/sample_baseline_e6_w3.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import random
import hashlib
import subprocess
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.utils import save_image

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import SDESampler

from models.unet_video_cond_film import UNetVideoCondFiLM
from core.config import TrainConfig

cv2.setNumThreads(0)

# ===== 固定：baseline E6 的 4 类动作 =====
ACTIONS = ["box", "jump", "run", "walk"]

# ===== 固定：baseline E6 的 CFG w =====
BASELINE_CFG_W = 3.0

# ===== 固定：baseline 输出目录（建议永远不要改）=====
BASELINE_OUT_DIR = "samples_baseline_e6_w3"


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 采样侧更建议 deterministic=True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 sha256（大文件分块读，避免占内存）"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    """
    获取当前仓库 git commit hash。
    - 如果不是 git repo 或 git 不可用，返回 None
    """
    try:
        cwd = str(repo_root) if repo_root is not None else None
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def iso_time_from_mtime(mtime: float) -> str:
    """把 mtime 转成 ISO 时间字符串（UTC）"""
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.isoformat()


class TestVideoDatasetAligned(Dataset):
    """
    读取 data_root/<split>/video/<subject>/<action>/*.mp4|avi...
    每个 subject 每个 action 取“第一个视频”作为条件输入（与你原脚本一致）
    """

    def __init__(self, root: str, split: str, clip_len: int, img_size: int, alpha: float):
        self.root = Path(root)
        self.split = split
        self.clip_len = int(clip_len)
        self.img_size = int(img_size)
        self.alpha = float(alpha)

        self.video_root = self.root / split / "video"
        if not self.video_root.is_dir():
            raise FileNotFoundError(f"video_root 不存在: {self.video_root}")

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
            raise RuntimeError(f"[TestVideoDatasetAligned] 在 {self.video_root} 下没有任何视频")

    def __len__(self):
        return len(self.samples)

    def _crop_square_bias_left(self, frame: np.ndarray) -> np.ndarray:
        """
        与你当前预处理一致：裁成正方形 + 横向偏移（alpha 控制 left bias）
        """
        h, w = frame.shape[:2]
        short = min(h, w)
        top = (h - short) // 2
        if w > short:
            left = int(round((w - short) * self.alpha))
            left = max(0, min(left, w - short))
        else:
            left = 0
        return frame[top: top + short, left: left + short]

    def _load_clip(self, path: Path) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        T = self.clip_len

        # 取中间片段（与你原脚本一致）
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
            frame = self._crop_square_bias_left(frame)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
            last = frame

        cap.release()

        while len(frames) < T:
            if last is None:
                last = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frames.append(last.copy())

        return np.stack(frames, axis=0)

    def get_first_path_by_subject_action(self) -> Dict[str, Dict[str, Path]]:
        m: Dict[str, Dict[str, Path]] = {}
        for p, subj, action in self.samples:
            if subj not in m:
                m[subj] = {}
            if action not in m[subj]:
                m[subj][action] = p
        return m


class VelocityCFG(nn.Module):
    """
    v = v_uncond + w*(v_cond - v_uncond)
    - uncond 使用 y_zero
    """
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


def _sample_images(rf: RectifiedFlow, y: torch.Tensor, num_steps: int, seed: int, B: int) -> torch.Tensor:
    sampler = SDESampler(rectified_flow=rf)
    sampler.sample_loop(num_samples=B, num_steps=num_steps, seed=seed, y=y)
    return sampler.trajectories[-1].clamp(0.0, 1.0)


@torch.no_grad()
def main():
    """
    Baseline E6：VideoCond + FiLM + CFG-training + CFG-sampling(w=3)

    关键保证：
    - w 固定为 3.0（不可通过命令行修改）
    - eval 先采样（每动作 B_eval=28）
    - grid 直接取 eval 的前 B_grid 张 => 严格 grid ⊂ eval
    - 输出目录固定为 samples_baseline_e6_w3
    - 自动写 meta.json：ckpt sha256 / mtime / cfg_w / seed 规则 / ckpt["cfg"] / git commit / 生成统计
    """
    parser = argparse.ArgumentParser()

    # 只保留“不会改变 baseline 定义”的参数
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.3)

    # 论文里建议固定：grid=4，eval=28
    parser.add_argument("--grid_n_per_action", type=int, default=4)
    parser.add_argument("--eval_n_per_action", type=int, default=28)

    # ckpt 默认就是 E6 best；允许你手动指定（但 meta 会记录）
    parser.add_argument("--ckpt", type=str, default="checkpoints/video_cond_flow_film_cfg_best.pth")

    args = parser.parse_args()

    train_cfg = TrainConfig()
    device = torch.device(
        train_cfg.device if torch.cuda.is_available() and "cuda" in str(train_cfg.device) else "cpu"
    )

    print(
        f"[sample_baseline_e6_w3] device={device}, cfg_w={BASELINE_CFG_W}, "
        f"steps={args.num_steps}, seed={args.seed}, alpha={args.alpha}"
    )
    print(
        f"[sample_baseline_e6_w3] grid_n_per_action={args.grid_n_per_action}, "
        f"eval_n_per_action={args.eval_n_per_action}"
    )

    set_deterministic(args.seed)

    out_dir = Path(BASELINE_OUT_DIR)
    grid_dir = out_dir / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load ckpt（并收集 hash/mtime）=====
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"ckpt 不存在: {ckpt_path}"
    ckpt = torch.load(str(ckpt_path), map_location=device)

    ckpt_abs = str(ckpt_path.resolve())
    ckpt_stat = ckpt_path.stat()
    ckpt_mtime_iso = iso_time_from_mtime(ckpt_stat.st_mtime)
    ckpt_sha256 = file_sha256(ckpt_path)

    # ===== 尽量从 ckpt 里拿训练 cfg =====
    train_cfg_in_ckpt = None
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        train_cfg_in_ckpt = ckpt["cfg"]
    if train_cfg_in_ckpt is not None and not isinstance(train_cfg_in_ckpt, dict):
        try:
            train_cfg_in_ckpt = dict(train_cfg_in_ckpt)
        except Exception:
            try:
                train_cfg_in_ckpt = vars(train_cfg_in_ckpt)
            except Exception:
                train_cfg_in_ckpt = str(train_cfg_in_ckpt)

    # ===== 写 meta.json（先写一版，后面采样结束再更新统计）=====
    meta = {
        "name": "baseline_e6_w3",
        "out_dir": str(out_dir.resolve()),
        "cfg_w": BASELINE_CFG_W,
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "seed_rule": "seed = base_seed + subj_idx*1000 + act_idx*10",
        "alpha": float(args.alpha),
        "grid_n_per_action": int(args.grid_n_per_action),
        "eval_n_per_action": int(args.eval_n_per_action),
        "ckpt": {
            "path": ckpt_abs,
            "sha256": ckpt_sha256,
            "mtime_utc": ckpt_mtime_iso,
            "size_bytes": int(ckpt_stat.st_size),
        },
        "train_cfg_from_ckpt": train_cfg_in_ckpt,
        "train_cfg_runtime": {
            "data_root": getattr(train_cfg, "data_root", None),
            "clip_len": getattr(train_cfg, "clip_len", None),
            "img_size": getattr(train_cfg, "img_size", None),
            "time_emb_dim": getattr(train_cfg, "time_emb_dim", None),
            "device": getattr(train_cfg, "device", None),
        },
        "git_commit": get_git_commit(),
        "generated_summary": None,  # 采样后填
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[sample_baseline_e6_w3] meta saved -> {meta_path}")

    # ===== Dataset：只用 test split 的视频 =====
    ds = TestVideoDatasetAligned(
        root=train_cfg.data_root,
        split="test",
        clip_len=train_cfg.clip_len,
        img_size=train_cfg.img_size,
        alpha=args.alpha,
    )
    subj2actpath = ds.get_first_path_by_subject_action()
    subjects = sorted(subj2actpath.keys())

    # ===== Model =====
    base = UNetVideoCondFiLM(img_channels=3, base_ch=64, time_emb_dim=train_cfg.time_emb_dim)
    base.load_state_dict(ckpt["model"], strict=True)
    base.to(device).eval()

    # 固定 CFG w=3
    model = VelocityCFG(base, cfg_w=BASELINE_CFG_W).to(device).eval()

    rf = RectifiedFlow(
        data_shape=(3, train_cfg.img_size, train_cfg.img_size),
        velocity_field=model,
        device=device,
    )

    # ===== 生成统计（用于 meta 更新）=====
    generated = {
        "subjects_total_found": int(len(subjects)),
        "subjects_done": 0,
        "subjects_skipped_missing_actions": 0,
        "images_saved_eval": 0,
        "images_saved_grid": 0,
    }

    # ===== 采样 =====
    for subj_idx, subj in enumerate(subjects):
        if subj not in subj2actpath:
            generated["subjects_skipped_missing_actions"] += 1
            continue
        if not all(a in subj2actpath[subj] for a in ACTIONS):
            generated["subjects_skipped_missing_actions"] += 1
            continue

        # 先生成 eval：每动作 B_eval
        eval_imgs_by_action = {}
        for act_idx, action in enumerate(ACTIONS):
            vp = subj2actpath[subj][action]

            clip_np = ds._load_clip(vp)
            clip = torch.from_numpy(clip_np).float() / 255.0
            clip = clip.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
            video = clip.unsqueeze(0).to(device)          # (1,T,3,H,W)

            B_eval = int(args.eval_n_per_action)
            y = video.expand(B_eval, -1, -1, -1, -1).contiguous()

            seed = args.seed + subj_idx * 1000 + act_idx * 10  # 固定 seed 规则
            imgs = _sample_images(rf, y=y, num_steps=args.num_steps, seed=seed, B=B_eval)
            eval_imgs_by_action[action] = (imgs, vp)

            # 保存 eval 图片（eval_gen_with_cls 会按 action 前缀解析标签）
            for k, img in enumerate(imgs):
                save_image(img, out_dir / f"{action}_{subj}_{vp.stem}_w{BASELINE_CFG_W:g}_{k:03d}.png")
            generated["images_saved_eval"] += B_eval

        print(f"[sample_baseline_e6_w3] saved eval images for {subj} (total={4*args.eval_n_per_action})")

        # 再生成 grid：直接取 eval 的前 B_grid（严格子集）
        grid_all = []
        B_grid = int(args.grid_n_per_action)
        for action in ACTIONS:
            imgs, _vp = eval_imgs_by_action[action]
            grid_all.append(imgs[:B_grid])
        grid_imgs = torch.cat(grid_all, dim=0)  # 4 actions * B_grid

        save_image(grid_imgs, grid_dir / f"grid_{subj}_w{BASELINE_CFG_W:g}.png", nrow=B_grid)
        generated["images_saved_grid"] += (len(ACTIONS) * B_grid)

        print(f"[sample_baseline_e6_w3] saved grid_{subj}_w{BASELINE_CFG_W:g}.png")

        generated["subjects_done"] += 1

    # ===== 更新 meta：写入生成统计 =====
    meta = json.loads(meta_path.read_text())
    meta["generated_summary"] = generated
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[sample_baseline_e6_w3] meta updated -> {meta_path}")

    print("[sample_baseline_e6_w3] done.")
    print(f"[sample_baseline_e6_w3] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
