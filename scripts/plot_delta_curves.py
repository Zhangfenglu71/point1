import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load(csv):
    df = pd.read_csv(csv)
    g = df.groupby("step").mean(numeric_only=True).reset_index()
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--name_a", default="E4")
    ap.add_argument("--name_b", default="E7")
    ap.add_argument("--out", default="delta_curves.png")
    args = ap.parse_args()

    a = load(args.csv_a)
    b = load(args.csv_b)

    # 1) delta_norm_std curve
    plt.figure()
    plt.plot(a["step"], a["delta_norm_std"], label=args.name_a)
    plt.plot(b["step"], b["delta_norm_std"], label=args.name_b)
    plt.xlabel("step")
    plt.ylabel("delta_norm_std")
    plt.legend()
    plt.tight_layout()
    out1 = Path(args.out).with_name(Path(args.out).stem + "_std.png")
    plt.savefig(out1, dpi=200)

    # 2) cos_prev_mean curve (skip NaN automatically)
    if "cos_prev_mean" in a.columns and "cos_prev_mean" in b.columns:
        plt.figure()
        plt.plot(a["step"], a["cos_prev_mean"], label=args.name_a)
        plt.plot(b["step"], b["cos_prev_mean"], label=args.name_b)
        plt.xlabel("step")
        plt.ylabel("cos_prev_mean")
        plt.legend()
        plt.tight_layout()
        out2 = Path(args.out).with_name(Path(args.out).stem + "_cos.png")
        plt.savefig(out2, dpi=200)

    print("[saved]", out1)
    if "cos_prev_mean" in a.columns and "cos_prev_mean" in b.columns:
        print("[saved]", out2)

if __name__ == "__main__":
    main()
