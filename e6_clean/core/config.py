from dataclasses import dataclass, field
from typing import Any, Dict


def _default_cfg() -> Dict[str, Any]:
    return {}


@dataclass
class TrainConfig:
    """Minimal training config for the clean E6 baseline."""

    data_root: str = "data"
    img_size: int = 120
    clip_len: int = 64
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    seed: int = 0
    use_amp: bool = True
    num_workers: int = 4
    ckpt_dir: str = "checkpoints"

    device: str = "cuda"
    time_emb_dim: int = 256

    extra: Dict[str, Any] = field(default_factory=_default_cfg)

    @classmethod
    def from_args(cls) -> "TrainConfig":
        import argparse

        parser = argparse.ArgumentParser(description="Train Rectified Flow E6 baseline")
        parser.add_argument("--data_root", type=str, default=cls.data_root)
        parser.add_argument("--img_size", type=int, default=cls.img_size)
        parser.add_argument("--clip_len", type=int, default=cls.clip_len)
        parser.add_argument("--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--weight_decay", type=float, default=cls.weight_decay)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--use_amp", type=int, default=int(cls.use_amp))
        parser.add_argument("--num_workers", type=int, default=cls.num_workers)
        parser.add_argument("--ckpt_dir", type=str, default=cls.ckpt_dir)
        parser.add_argument("--device", type=str, default=cls.device)
        parser.add_argument("--time_emb_dim", type=int, default=cls.time_emb_dim)

        args, unknown = parser.parse_known_args()
        cfg = cls(
            data_root=args.data_root,
            img_size=args.img_size,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seed=args.seed,
            use_amp=bool(args.use_amp),
            num_workers=args.num_workers,
            ckpt_dir=args.ckpt_dir,
            device=args.device,
            time_emb_dim=args.time_emb_dim,
        )

        if unknown:
            cfg.extra["unknown"] = unknown

        return cfg

