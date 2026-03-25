import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Config ----------------
BASE_DIR  = "ALFM/logs/results/cifar100/dino_vit_g14"  # <- change per dataset
METHOD    = "disagreement"
VARIANTS  = ["V1","V2","V3","V4","V5"]            # order to display
OUT_DIR   = os.path.join(BASE_DIR, "aggregated")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# core steps to compare (in display order)
CORE_STEPS = ["t_subpool","t_modalities","t_sets","t_score","t_topk","t_diversity"]
LABELS     = {
    "t_subpool":    "Subpool",
    "t_modalities": "Modalities",
    "t_sets":       "Sets",
    "t_score":      "Score",
    "t_topk":       "Top-k",
    "t_diversity":  "Diversity"
}
# ----------------------------------------


def _load_step_means_or_recompute(variant: str) -> pd.DataFrame:
    """
    Try reading {variant}_timing_step_means.csv.
    If missing, compute from {variant}_timing_agg.csv by averaging per-iteration means.
    Returns a DataFrame with columns: step, mean_seconds, percent_of_total
    """
    step_path = os.path.join(OUT_DIR, f"{variant}_timing_step_means.csv")
    if os.path.exists(step_path):
        df = pd.read_csv(step_path)
        # ensure expected cols exist
        if {"step","mean_seconds","percent_of_total"}.issubset(df.columns):
            return df

    # recompute from timing_agg if needed
    agg_path = os.path.join(OUT_DIR, f"{variant}_timing_agg.csv")
    if not os.path.exists(agg_path):
        print(f"[WARN] No step means or timing_agg for {variant}. Skipping.")
        return pd.DataFrame()

    T = pd.read_csv(agg_path)
    # find *_mean columns for core steps
    means = {}
    for s in CORE_STEPS:
        col = f"{s}_mean"
        if col in T.columns:
            means[s] = float(T[col].mean())
    if not means:
        print(f"[WARN] No core step means in {agg_path}.")
        return pd.DataFrame()

    df = pd.DataFrame({"step": list(means.keys()), "mean_seconds": list(means.values())})
    total = df["mean_seconds"].sum()
    df["percent_of_total"] = np.where(total > 0, 100.0 * df["mean_seconds"]/total, np.nan)
    return df


def build_long_table():
    """
    Build one long table with columns:
    variant, step, mean_seconds, percent_of_total
    (only for CORE_STEPS, and only for variants we can load)
    """
    rows = []
    for v in VARIANTS:
        df = _load_step_means_or_recompute(v)
        if df.empty:
            continue
        # keep only core steps in desired order
        df = df[df["step"].isin(CORE_STEPS)].copy()
        df["variant"] = v
        # enforce display order
        df["step"] = pd.Categorical(df["step"], categories=CORE_STEPS, ordered=True)
        df = df.sort_values("step")
        rows.append(df[["variant","step","mean_seconds","percent_of_total"]])
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    return long


def plot_grouped_bars_seconds(long: pd.DataFrame):
    """
    Grouped bars: x=steps, bars=variants, height=mean_seconds
    """
    if long.empty:
        print("[WARN] No data for grouped bar plot.")
        return

    # pivot to steps x variants
    P = long.pivot_table(index="step", columns="variant", values="mean_seconds", aggfunc="mean")
    P = P.reindex(CORE_STEPS)  # enforce step order
    steps = [LABELS.get(s, s) for s in P.index]

    x = np.arange(len(P.index))
    n = len(P.columns)
    w = 0.10 if n >= 6 else 0.12  # bar width heuristic
    off0 = - (n-1) * w / 2

    plt.figure(figsize=(6.2, 3.6))
    for i, v in enumerate(P.columns):
        y = P[v].values
        plt.bar(x + off0 + i*w, y, width=w, label=v)

    plt.xticks(x, steps, rotation=0)
    plt.ylabel("Mean seconds per round (step)")
    plt.title("Per-step timings across variants")
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(ncol=min(len(P.columns), 6), fontsize=8)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "timing_step_means_grouped_seconds.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] saved {out}")


def plot_heatmap_percent(long: pd.DataFrame):
    """
    Heatmap: rows=variants, cols=steps, values=percent_of_total
    """
    if long.empty:
        print("[WARN] No data for heatmap.")
        return
    H = long.pivot_table(index="variant", columns="step", values="percent_of_total", aggfunc="mean")
    # reorder axes
    H = H.reindex(index=VARIANTS, columns=CORE_STEPS)
    if H.isna().all(None):
        print("[WARN] Heatmap empty after pivot.")
        return

    plt.figure(figsize=(6.2, 3.6))
    mat = H.values
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="% of query time")

    plt.yticks(np.arange(len(H.index)), H.index)
    plt.xticks(np.arange(len(H.columns)), [LABELS.get(s, s) for s in H.columns], rotation=0)
    plt.title("Time composition (% total) across variants")
    # gridlines
    plt.gca().set_xticks(np.arange(-.5, len(H.columns), 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, len(H.index), 1), minor=True)
    plt.grid(which="minor", color="w", linestyle="-", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "timing_step_means_heatmap_percent.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] saved {out}")


def main():
    long = build_long_table()
    if long.empty:
        print("[WARN] No timing step means found for any variant.")
        return
    # save the combined table (handy for LaTeX tables later)
    combined_path = os.path.join(OUT_DIR, "timing_step_means_all_variants.csv")
    long.to_csv(combined_path, index=False)
    print(f"[OK] wrote {combined_path}")

    plot_grouped_bars_seconds(long)
    plot_heatmap_percent(long)


if __name__ == "__main__":
    main()
