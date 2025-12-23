import pandas as pd
import numpy as np

def summarize(csv_path: str, name: str):
    df = pd.read_csv(csv_path)
    eps = 1e-8

    # 去掉第一步 cos_prev 是 NaN
    df2 = df.copy()
    # CV
    df2["delta_cv"] = df2["delta_norm_std"] / (df2["delta_norm_mean"] + eps)

    # 汇总（对 step 做平均）
    out = {
        "name": name,
        "steps": df2["step"].nunique(),
        "avg_delta_norm_std": float(df2.groupby("step")["delta_norm_std"].mean().mean()),
        "avg_delta_cv": float(df2.groupby("step")["delta_cv"].mean().mean()),
        "avg_cos_prev_mean": float(df2.dropna(subset=["cos_prev_mean"]).groupby("step")["cos_prev_mean"].mean().mean()),
        "avg_cos_prev_std": float(df2.dropna(subset=["cos_prev_std"]).groupby("step")["cos_prev_std"].mean().mean()),
    }
    return out, df2

def main():
    a_sum, a_df = summarize("samples_delta_videocond/delta_stats_video.csv", "VideoCond (E1)")
    b_sum, b_df = summarize("samples_delta_film_cfg/delta_stats_film.csv", "VideoCond+FiLM (E2/E6)")

    print(pd.DataFrame([a_sum, b_sum]).to_string(index=False))

    # 你也可以把下面两行打开：输出逐step均值曲线，方便你画图
    print("\n[Curve] delta_norm_std per step (mean over batches):")
    print(a_df.groupby("step")["delta_norm_std"].mean().to_string())

if __name__ == "__main__":
    main()
