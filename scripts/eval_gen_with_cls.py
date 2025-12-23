# scripts/eval_gen_with_cls.py
from pathlib import Path
from typing import Dict, Tuple
import argparse

import torch
from torch.utils.data import DataLoader

from models.radar_cls_resnet import RadarResNet18
from core.constants import ACTION_TO_ID, ID_TO_ACTION
from core.datasets.gen_radar import GenRadarDataset


@torch.no_grad()
def eval_generated(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    per_total = {name: 0 for name in ACTION_TO_ID.keys()}
    per_correct = {name: 0 for name in ACTION_TO_ID.keys()}

    for batch in loader:
        # batch could be (x,y) or (x,y,path,...) or dict
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch["radar"], batch["label"]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=1)

        total += int(y.numel())
        correct += int((pred == y).sum().item())

        for aid in y.unique().tolist():
            aid = int(aid)
            name = ID_TO_ACTION[aid]
            m = (y == aid)
            per_total[name] += int(m.sum().item())
            per_correct[name] += int(((pred == y) & m).sum().item())

    overall_acc = correct / max(total, 1)
    per_class_acc = {
        name: (per_correct[name] / per_total[name] if per_total[name] > 0 else 0.0)
        for name in ACTION_TO_ID.keys()
    }
    return overall_acc, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="生成图像目录（包含 box/jump/run/walk 子目录）")
    parser.add_argument("--ckpt", type=str, required=True, help="分类器 ckpt 路径")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--img_size", type=int, default=120, help="输入分辨率（需与分类器训练一致）")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    gen_root = args.root
    ckpt_path = args.ckpt
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    print(f"[eval_gen_with_cls] device = {device}")
    print(f"[eval_gen_with_cls] root   = {gen_root}")
    print(f"[eval_gen_with_cls] ckpt   = {ckpt_path}")

    # ===== 加载生成数据 =====
    gen_set = GenRadarDataset(gen_root, args.img_size)
    gen_loader = DataLoader(
        gen_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ===== 加载分类器 =====
    model = RadarResNet18(num_classes=len(ACTION_TO_ID)).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 兼容：ckpt 可能是 {"model": ...} 或直接 state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    if isinstance(ckpt, dict):
        va = ckpt.get("val_acc", None)
        if va is not None:
            print(f"[eval_gen_with_cls] Loaded classifier, val_acc(best)={va:.4f}")

    # ===== 评估 =====
    overall_acc, per_class_acc = eval_generated(model, gen_loader, device)

    print("\n=== Generated Spectrogram Classification Accuracy ===")
    print(f"Overall ACC: {overall_acc*100:.2f}%")
    print("Per-class ACC:")
    for name in ACTION_TO_ID.keys():
        print(f"  {name:5s}: {per_class_acc[name]*100:.2f}%")


if __name__ == "__main__":
    main()
