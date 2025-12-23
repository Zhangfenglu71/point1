# scripts/sample_baseline_e6_w3_delta.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
import random
import hashlib
import subprocess
from datetime import datetime, timezone
import csv

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.utils import save_image

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import SDESampler

from core.config import TrainConfig

# 两种模型都支持（根据 --model_type 选择）
from models.unet_video_cond import UNetVideoCond
from models.unet_video_cond_film import UNetVideoCondFiLM

cv2.setNumThreads(0)

# ===== 固定：baseline E6 的 4 类动作 =====
ACTIONS = ["box", "jump", "run", "walk"]

# ===== 固定：baseline E6 的 CFG w（不可改）=====
BASELINE_CFG_W = 3.0


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_git_commit(repo_root: Optional[Path] = None) -> Optional[str]:
    try:
        cwd = str(repo_root) if repo_root is not None else None
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def iso_time_from_mtime(mtime: float) -> str:
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.isoformat()


def _to_dict_like(x: Any) -> Optional[Dict[str, Any]]:
    """把 cfg / model_cfg 之类尽量转成 dict，失败返回 None"""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    try:
        return dict(x)
    except Exception:
        try:
            return vars(x)
        except Exception:
            return None


def _find_first_int(cfg: Dict[str, Any], keys: List[str], max_depth: int = 3) -> Optional[int]:
    """
    在 cfg (可能嵌套) 里找一些常见键名对应的 int 值。
    keys: ["base_ch", "base_channels", "ch", ...]
    """
    def _try_get(d: Dict[str, Any]) -> Optional[int]:
        for k in keys:
            if k in d:
                v = d.get(k)
                if isinstance(v, (int, np.integer)):
                    return int(v)
                # 有些可能是字符串
                if isinstance(v, str) and v.isdigit():
                    return int(v)
        return None

    # BFS/DFS 混合：逐层展开 dict
    q: List[Tuple[Dict[str, Any], int]] = [(cfg, 0)]
    seen = set()

    while q:
        d, dep = q.pop(0)
        if id(d) in seen:
            continue
        seen.add(id(d))

        v = _try_get(d)
        if v is not None:
            return v

        if dep >= max_depth:
            continue

        for _k, _v in d.items():
            if isinstance(_v, dict):
                q.append((_v, dep + 1))
            else:
                dd = _to_dict_like(_v)
                if isinstance(dd, dict):
                    q.append((dd, dep + 1))
    return None


def infer_base_ch_from_ckpt(ckpt: Any) -> Optional[int]:
    """
    base_ch 获取策略：
    - 优先从 ckpt['cfg'] / ckpt['model_cfg'] / ckpt['config'] 等读
    - 猜一些常见键名：base_ch / base_channels / ch / channels / model_base_ch / unet_ch
    """
    if not isinstance(ckpt, dict):
        return None

    candidates = []
    for k in ["cfg", "model_cfg", "config", "train_cfg", "model_config"]:
        if k in ckpt:
            d = _to_dict_like(ckpt.get(k))
            if isinstance(d, dict):
                candidates.append(d)

    if not candidates:
        return None

    key_aliases = [
        "base_ch",
        "base_channels",
        "ch",
        "channels",
        "model_base_ch",
        "unet_ch",
        "unet_base_ch",
        "base_channel",
    ]

    for c in candidates:
        v = _find_first_int(c, key_aliases, max_depth=3)
        if v is not None:
            return v
    return None


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


class DeltaStatsLogger:
    """
    逐 step 记录 Δv = v_cond - v_uncond 的统计（batch mean/std）。
    """
    def __init__(self):
        self.rows: List[Dict[str, float]] = []
        self._prev_delta: Optional[torch.Tensor] = None

    def reset(self):
        self.rows.clear()
        self._prev_delta = None

    @torch.no_grad()
    def log_step(self, step: int, t: float, v_u: torch.Tensor, v_c: torch.Tensor):
        delta = (v_c - v_u)

        B = delta.shape[0]
        d_flat = delta.reshape(B, -1)
        vu_flat = v_u.reshape(B, -1)
        vc_flat = v_c.reshape(B, -1)

        d_norm = torch.linalg.vector_norm(d_flat, ord=2, dim=1)
        delta_norm_mean = d_norm.mean().item()
        delta_norm_std = d_norm.std(unbiased=False).item()

        vu_norm_mean = torch.linalg.vector_norm(vu_flat, ord=2, dim=1).mean().item()
        vc_norm_mean = torch.linalg.vector_norm(vc_flat, ord=2, dim=1).mean().item()

        if self._prev_delta is None:
            cos_prev_mean = float("nan")
            cos_prev_std = float("nan")
        else:
            p_flat = self._prev_delta.reshape(B, -1)
            cos = torch.nn.functional.cosine_similarity(d_flat, p_flat, dim=1, eps=1e-8)
            cos_prev_mean = cos.mean().item()
            cos_prev_std = cos.std(unbiased=False).item()

        self.rows.append({
            "step": int(step),
            "t": float(t),
            "delta_norm_mean": float(delta_norm_mean),
            "delta_norm_std": float(delta_norm_std),
            "cos_prev_mean": float(cos_prev_mean),
            "cos_prev_std": float(cos_prev_std),
            "vc_norm_mean": float(vc_norm_mean),
            "vu_norm_mean": float(vu_norm_mean),
        })

        self._prev_delta = delta.detach()


class VelocityCFG(nn.Module):
    """
    v_cfg = v_uncond + w*(v_cond - v_uncond)

    支持两种模式：
    1) fixed:  w = cfg_w（原逻辑）
    2) adaptive (Δv-normalized): 让每一步的放大后的Δv“能量”保持在 target 上
       w_t = clip(cfg_w * target / (||Δv|| + eps), w_min, w_max)
       target 默认取每次采样的第0步 batch mean(||Δv||)
    """
    def __init__(
        self,
        base: nn.Module,
        cfg_w: float,
        delta_logger: Optional[DeltaStatsLogger] = None,
        adaptive: bool = False,
        w_min: float = 0.0,
        w_max: float = 6.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.base = base
        self.cfg_w = float(cfg_w)
        self.delta_logger = delta_logger

        self.adaptive = bool(adaptive)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.eps = float(eps)

        self._step_counter = 0
        self._target_norm = None  # 每次采样重置；用于 adaptive 模式

    def reset_stats_state(self):
        self._step_counter = 0
        self._target_norm = None
        if self.delta_logger is not None:
            self.delta_logger.reset()

    def forward(self, x, t, y=None, **kwargs):
        assert y is not None

        if torch.is_tensor(t):
            t_scalar = float(t.flatten()[0].item())
        else:
            t_scalar = float(t)

        # ====== 原逻辑：w==1 就只跑 cond（为了速度）======
        if (not self.adaptive) and abs(self.cfg_w - 1.0) < 1e-8:
            v_c = self.base(x, t, y=y, **kwargs)
            if self.delta_logger is not None:
                v_u = v_c
                self.delta_logger.log_step(self._step_counter, t_scalar, v_u, v_c)
                self._step_counter += 1
            return v_c

        # ====== 计算 uncond / cond ======
        y0 = torch.zeros_like(y)
        v_u = self.base(x, t, y=y0, **kwargs)
        v_c = self.base(x, t, y=y, **kwargs)

        if self.delta_logger is not None:
            self.delta_logger.log_step(self._step_counter, t_scalar, v_u, v_c)
            self._step_counter += 1

        delta = (v_c - v_u)

        # ====== fixed CFG ======
        if not self.adaptive:
            return v_u + self.cfg_w * delta

        # ====== adaptive CFG (Δv-normalized) ======
        B = delta.shape[0]
        d_flat = delta.reshape(B, -1)
        d_norm = torch.linalg.vector_norm(d_flat, ord=2, dim=1)  # (B,)

        # 用第0步的 batch mean(||Δv||) 做 target
        if self._target_norm is None:
            self._target_norm = d_norm.mean().detach()

        w_t = self.cfg_w * (self._target_norm / (d_norm + self.eps))  # (B,)
        w_t = torch.clamp(w_t, self.w_min, self.w_max)

        # broadcast 到 (B,C,H,W)
        w_t = w_t.view(B, 1, 1, 1)
        return v_u + w_t * delta



def _sample_images(rf: RectifiedFlow, y: torch.Tensor, num_steps: int, seed: int, B: int) -> torch.Tensor:
    sampler = SDESampler(rectified_flow=rf)
    sampler.sample_loop(num_samples=B, num_steps=num_steps, seed=seed, y=y)
    return sampler.trajectories[-1].clamp(0.0, 1.0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    # 不改变 baseline 定义的参数
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.3)

    parser.add_argument("--grid_n_per_action", type=int, default=4)
    parser.add_argument("--eval_n_per_action", type=int, default=28)

    parser.add_argument("--ckpt", type=str, default="checkpoints/video_cond_flow_film_cfg_best.pth")

    # ===== 新增：对比两种模型 =====
    parser.add_argument("--model_type", type=str, choices=["video", "film"], default="film",
                        help="Choose model backbone: video (UNetVideoCond) or film (UNetVideoCondFiLM).")
    parser.add_argument("--base_ch", type=int, default=None,
                        help="UNet base channels. If ckpt provides cfg/model_cfg, it will override this.")
    parser.add_argument("--out_dir", type=str, default="samples_baseline_e6_w3",
                        help="Output dir for images/grids/meta/csv (avoid overwrite by using different dirs).")

    # ===== delta stats =====
    parser.add_argument("--log_delta_stats", action="store_true",
                        help="Enable logging CFG delta stats (Δv=v_cond-v_uncond) per step.")
    parser.add_argument("--delta_stats_out", type=str, default="delta_stats.csv",
                        help="CSV filename or path (relative -> saved under out_dir).")

    args = parser.parse_args()

    train_cfg = TrainConfig()
    device = torch.device(
        train_cfg.device if torch.cuda.is_available() and "cuda" in str(train_cfg.device) else "cpu"
    )

    print(
        f"[sample_baseline_e6_w3_delta] device={device}, cfg_w={BASELINE_CFG_W}, "
        f"steps={args.num_steps}, seed={args.seed}, alpha={args.alpha}"
    )
    print(
        f"[sample_baseline_e6_w3_delta] model_type={args.model_type}, out_dir={args.out_dir}"
    )

    set_deterministic(args.seed)

    out_dir = Path(args.out_dir)
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

    # ===== base_ch 推断：ckpt 优先，否则命令行，否则默认 64 =====
    base_ch_ckpt = infer_base_ch_from_ckpt(ckpt)
    final_base_ch = base_ch_ckpt if base_ch_ckpt is not None else args.base_ch
    if final_base_ch is None:
        final_base_ch = 64
    print(f"[sample_baseline_e6_w3_delta] base_ch(final)={final_base_ch} (ckpt={base_ch_ckpt}, cli={args.base_ch})")

    # ===== 尽量从 ckpt 里拿训练 cfg（写 meta 用）=====
    train_cfg_in_ckpt = None
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        train_cfg_in_ckpt = _to_dict_like(ckpt["cfg"]) or ckpt["cfg"]
    if train_cfg_in_ckpt is not None and not isinstance(train_cfg_in_ckpt, dict):
        d = _to_dict_like(train_cfg_in_ckpt)
        train_cfg_in_ckpt = d if d is not None else str(train_cfg_in_ckpt)

    # ===== 写 meta_delta.json =====
    meta = {
        "name": "baseline_e6_w3_delta",
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
        "generated_summary": None,
        "delta_stats": {
            "enabled": bool(args.log_delta_stats),
            "csv_out": (str((out_dir / args.delta_stats_out).resolve())
                        if not Path(args.delta_stats_out).is_absolute()
                        else str(Path(args.delta_stats_out).resolve()))
        },
        # ===== 新增字段（你要求的）=====
        "model_type": str(args.model_type),
        "base_ch": int(final_base_ch),
    }
    meta_path = out_dir / "meta_delta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[sample_baseline_e6_w3_delta] meta saved -> {meta_path}")

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

    # ===== Model（根据 model_type 选择）=====
    if args.model_type == "video":
        base = UNetVideoCond(img_channels=3, base_ch=int(final_base_ch), time_emb_dim=train_cfg.time_emb_dim)
    else:
        base = UNetVideoCondFiLM(img_channels=3, base_ch=int(final_base_ch), time_emb_dim=train_cfg.time_emb_dim)

    assert isinstance(ckpt, dict) and "model" in ckpt, "ckpt 格式不对：需要 dict 且包含 key='model'"
    base.load_state_dict(ckpt["model"], strict=True)
    base.to(device).eval()

    # ===== 可选：delta stats logger =====
    delta_logger = DeltaStatsLogger() if args.log_delta_stats else None
    delta_rows_all: List[Dict[str, float]] = []

    model = VelocityCFG(
    base,
    cfg_w=BASELINE_CFG_W,
    delta_logger=delta_logger,
    adaptive=True,   # ✅ 开启 Δv-normalized adaptive CFG
    w_min=0.0,
    w_max=6.0,       # 你可以先设 6，避免过大
    ).to(device).eval()


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

        eval_imgs_by_action = {}
        for act_idx, action in enumerate(ACTIONS):
            vp = subj2actpath[subj][action]

            clip_np = ds._load_clip(vp)
            clip = torch.from_numpy(clip_np).float() / 255.0
            clip = clip.permute(0, 3, 1, 2).contiguous()  # (T,3,H,W)
            video = clip.unsqueeze(0).to(device)          # (1,T,3,H,W)

            B_eval = int(args.eval_n_per_action)
            y = video.expand(B_eval, -1, -1, -1, -1).contiguous()

            seed = args.seed + subj_idx * 1000 + act_idx * 10

            if args.log_delta_stats:
                model.reset_stats_state()

            imgs = _sample_images(rf, y=y, num_steps=args.num_steps, seed=seed, B=B_eval)
            eval_imgs_by_action[action] = (imgs, vp)

            for k, img in enumerate(imgs):
                save_image(img, out_dir / f"{action}_{subj}_{vp.stem}_w{BASELINE_CFG_W:g}_{k:03d}.png")
            generated["images_saved_eval"] += B_eval

            if args.log_delta_stats and delta_logger is not None:
                for r in delta_logger.rows:
                    r["subj"] = subj
                    r["action"] = action
                    r["seed"] = int(seed)
                delta_rows_all.extend(delta_logger.rows)

        print(f"[sample_baseline_e6_w3_delta] saved eval images for {subj} (total={4*args.eval_n_per_action})")

        # grid：取 eval 的前 B_grid（严格子集）
        grid_all = []
        B_grid = int(args.grid_n_per_action)
        for action in ACTIONS:
            imgs, _vp = eval_imgs_by_action[action]
            grid_all.append(imgs[:B_grid])
        grid_imgs = torch.cat(grid_all, dim=0)

        save_image(grid_imgs, grid_dir / f"grid_{subj}_w{BASELINE_CFG_W:g}.png", nrow=B_grid)
        generated["images_saved_grid"] += (len(ACTIONS) * B_grid)

        print(f"[sample_baseline_e6_w3_delta] saved grid_{subj}_w{BASELINE_CFG_W:g}.png")
        generated["subjects_done"] += 1

    # ===== 写 delta_stats.csv =====
    if args.log_delta_stats:
        out_csv = Path(args.delta_stats_out)
        if not out_csv.is_absolute():
            out_csv = out_dir / out_csv

        header = [
            "step", "t",
            "delta_norm_mean", "delta_norm_std",
            "cos_prev_mean", "cos_prev_std",
            "vc_norm_mean", "vu_norm_mean",
            "subj", "action", "seed",
        ]
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in delta_rows_all:
                w.writerow([
                    r.get("step", ""),
                    r.get("t", ""),
                    r.get("delta_norm_mean", ""),
                    r.get("delta_norm_std", ""),
                    r.get("cos_prev_mean", ""),
                    r.get("cos_prev_std", ""),
                    r.get("vc_norm_mean", ""),
                    r.get("vu_norm_mean", ""),
                    r.get("subj", ""),
                    r.get("action", ""),
                    r.get("seed", ""),
                ])
        print(f"[delta_stats] saved -> {out_csv}")

    # ===== 更新 meta_delta.json：写入生成统计 =====
    meta = json.loads(meta_path.read_text())
    meta["generated_summary"] = generated
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[sample_baseline_e6_w3_delta] meta updated -> {meta_path}")

    print("[sample_baseline_e6_w3_delta] done.")
    print(f"[sample_baseline_e6_w3_delta] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
