# core/datasets/gen_radar.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import re

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from ..constants import ACTION_TO_ID, ID_TO_ACTION


class GenRadarDataset(Dataset):
    """
    兼容三种命名/组织方式来推断 label：
    1) action_xxx.png / action_xxx.jpg         (动作在文件名前缀)
    2) ywb_0_box_01_1.jpg                      (动作在文件名中间某个 token)
    3) data/test/radar/S10/box/xxx.jpg         (动作在父目录名中)
    优先级：文件名 > 父目录名
    """

    def __init__(self, root: str, img_size: int = 120):
        super().__init__()
        self.root = Path(root)
        self.img_size = int(img_size)

        if not self.root.is_dir():
            raise FileNotFoundError(f"gen_root 不存在: {self.root}")

        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.samples = []  # (path, label_id)

        def infer_action_from_path(p: Path) -> Optional[str]:
            """
            从文件名或目录名中推断动作名（返回 'box'/'jump'/'run'/'walk' 或 None）
            """
            # --------- 1) 从文件名 token 扫描（最强） ----------
            # 用下划线、短横线、空格、点等做切分
            stem = p.stem.lower()
            tokens = re.split(r"[_\-\s\.]+", stem)

            # 优先：第一个 token 就是动作（兼容原逻辑）
            if len(tokens) > 0 and tokens[0] in ACTION_TO_ID:
                return tokens[0]

            # 否则：扫描整个 tokens，找到任意一个动作名
            for t in tokens:
                if t in ACTION_TO_ID:
                    return t

            # --------- 2) 从父目录名兜底 ----------
            # 例如 .../S10/box/xxx.jpg
            for parent in [p.parent, *p.parents]:
                name = parent.name.lower()
                if name in ACTION_TO_ID:
                    return name

            return None

        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue

            # 跳过你 grid_* 这种合成图（如果需要的话）
            if p.name.lower().startswith("grid_"):
                continue

            action_name = infer_action_from_path(p)
            if action_name is None:
                continue

            label_id = ACTION_TO_ID[action_name]
            self.samples.append((p, label_id))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"[GenRadarDataset] 在 {self.root} 下没有找到可推断动作标签的图像。\n"
                f"支持示例：box_xxx.png 或 ywb_0_box_01_1.jpg 或目录名为 box/jump/run/walk"
            )

        # 统计一下各类数量
        counts: Dict[int, int] = {}
        for _, lid in self.samples:
            counts[lid] = counts.get(lid, 0) + 1

        print(f"[GenRadarDataset] root={self.root}, num_images={len(self.samples)}")
        for lid, cnt in sorted(counts.items()):
            print(f"  {ID_TO_ACTION[lid]}: {cnt}")

        self.transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(label_id, dtype=torch.long)
        return {"radar": x, "label": y, "path": str(path)}



@torch.no_grad()
def eval_generated(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_correct = 0
    total_samples = 0

    num_classes = len(ACTION_TO_ID)
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    for batch in loader:
        x = batch["radar"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        bs = x.size(0)
        total_samples += bs
        total_correct += (preds == y).sum().item()

        for i in range(bs):
            li = y[i].item()
            pi = preds[i].item()
            class_total[li] += 1
            if li == pi:
                class_correct[li] += 1

    overall_acc = total_correct / max(total_samples, 1)

    per_class_acc = {}
    for lid in range(num_classes):
        name = ID_TO_ACTION.get(lid, str(lid))
        acc = (class_correct[lid] / class_total[lid]) if class_total[lid] > 0 else 0.0
        per_class_acc[name] = acc

    return overall_acc, per_class_acc

