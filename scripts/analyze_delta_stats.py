# scripts/analyze_delta_stats.py
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def summarize(csv_path: str, name: str):
    df = pd.read_csv(csv_path)
    eps = 1e-8

    df2 = df.copy()
    df2["delta_cv"] = df2["delta_norm_std"] / (df2["delta_norm_mean"] + eps)

    out = {
        "name": name,
        "steps": int(df2["step"].nunique()),
        "avg_delta_norm_std": float(df2.groupby("step")["delta_norm_std"].mean().mean()),
        "avg_delta_cv": float(df2.groupby("step")["delta_cv"].mean().mean()),
        "avg_cos_prev_mean": float(
            df2.dropna(subset=["cos_prev_mean"]).groupby("step")["cos_prev_mean"].mean().mean()
        ),
        "avg_cos_prev_std": float(
            df2.dropna(subset=["cos_prev_std"]).groupby("step")["cos_prev_std"].mean().mean()
        ),
    }
    return out, df2


def default_name_from_path(p: str) -> str:
    stem = Path(p).stem
    # 常见映射（你也可以继续加）
    mapping = {
        "delta_E1": "VideoCond (E1)",
        "delta_E4": "FiLM+CFG (E4)",
        "delta_E7": "FiLM+CFG+ΔvVar (E7)",
    }
    return mapping.get(stem, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="One or more delta stats CSV files.")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Optional names for each CSV (same length as csvs).")
    parser.add_argument("--curve", action="store_true",
                        help="Print per-step curve for delta_norm_std (mean over batches).")
    parser.add_argument("--curve_idx", type=int, default=0,
                        help="Which input CSV index to print curve for (default 0).")
    args = parser.parse_args()

    if args.names is not None and len(args.names) not in (0, len(args.csvs)):
        raise ValueError("If --names is provided, it must have the same length as csvs.")

    sums = []
    dfs = []

    for i, csv_path in enumerate(args.csvs):
        name = args.names[i] if args.names else default_name_from_path(csv_path)
        s, d = summarize(csv_path, name)
        sums.append(s)
        dfs.append(d)

    print(pd.DataFrame(sums).to_string(index=False))

    if args.curve:
        idx = int(np.clip(args.curve_idx, 0, len(dfs) - 1))
        print("\n[Curve] delta_norm_std per step (mean over batches):")
        print(dfs[idx].groupby("step")["delta_norm_std"].mean().to_string())


if __name__ == "__main__":
    main()