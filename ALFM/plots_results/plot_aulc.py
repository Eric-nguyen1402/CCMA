#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute & plot AULC for Active Learning runs.

Requirements:
  pip install pandas matplotlib numpy

What it does:
  - Loads per-method CSVs (method-*.csv) from BASE_DIR
  - Interpolates each seed/run to a common labeled-fraction grid
  - Computes AULC (normalized to [0,1]) and Final Accuracy (mean±std)
  - Saves:
      plots/curves_fraction_accuracy.png
      plots/bar_aulc.png
      aggregated/summary_aulc_finalacc.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
})
# --------- Config ---------
BASE_DIR = "ALFM/logs/results/cifar100/dino_vit_g14"  # change per dataset
METHODS = [
    "random",
    "uncertainty",
    "entropy",
    "margins",
    "bald",
    "powerbald",
    "coreset",
    "badge",
    "alfamix",
    # "typiclust",
    "probcover",
    "disagreement",
]
METRIC_KEY = "TEST_MulticlassAccuracy"  # can also use AUROC / F1
N_GRID = 50  # number of points for the common fraction grid
OUT_AGG_DIR = os.path.join(BASE_DIR, "aggregated")
OUT_PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_AGG_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
# --------------------------


def load_runs_for_method(method: str) -> list[pd.DataFrame]:
    """Load all CSV runs for a given method; return list of dataframes."""
    pattern = os.path.join(BASE_DIR, f"{method}-*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No CSVs found for method '{method}' with pattern {pattern}")
        return []
    dfs = []
    required = {"iteration", "num_samples", "TEST_MulticlassAccuracy",
                "TEST_MulticlassAUROC", "TEST_MulticlassF1Score"}
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            missing = required - set(df.columns)
            if missing:
                print(f"[WARN] {os.path.basename(p)} missing {missing}; skipping.")
                continue
            df["run_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return dfs


def interp_to_grid(df: pd.DataFrame, x_col: str, y_col: str, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation of a single run to a common x grid (monotonic x assumed)."""
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    # sort & uniquify x
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux, idx = np.unique(x, return_index=True)
    uy = y[idx]
    if ux.size < 2:
        return np.full_like(grid, np.nan, dtype=float)
    return np.interp(grid, ux, uy, left=np.nan, right=np.nan)


def compute_aulc_from_runs(dfs: list[pd.DataFrame],
                           metric: str = METRIC_KEY,
                           n_grid: int = N_GRID):
    """
    Returns a dict with:
      grid_f, mean_y, std_y, aulc_mu, aulc_sd, final_mu, final_sd, f_max
    """
    if not dfs:
        return None

    # Convert each run to labeled fraction & record max fraction reached
    per_run = []
    max_fracs = []
    max_nums = []

    for df in dfs:
        df = df.copy().sort_values("num_samples")
        run_max = float(df["num_samples"].max())
        df["frac"] = df["num_samples"] / run_max
        per_run.append(df)
        max_fracs.append(float(np.nanmax(df["frac"].to_numpy(dtype=float))))
        max_nums.append(run_max)

    common_max = float(np.min(max_nums))
    f_max_common = 1.0 * (common_max / common_max)  # == 1.0
    grid_f = np.linspace(0.0, 1.0, n_grid) 

    # Interpolate each run to the common grid
    Ys = []
    for run in per_run:
        run = run.sort_values("frac")
        y = interp_to_grid(run, "frac", metric, grid_f)
        Ys.append(y)
    Y = np.vstack(Ys)  # [seeds, len(grid)]

    # mean/std across seeds
    mean_y = np.nanmean(Y, axis=0)
    std_y = np.nanstd(Y, axis=0)

    # AULC per seed (normalized by f_max_common so it's in [0,1])
    aulc_vals = []
    for y in Ys:
        ok = np.isfinite(y)
        if ok.sum() >= 2:
            a = np.trapz(y[ok], grid_f[ok]) / f_max_common
        else:
            a = np.nan
        aulc_vals.append(a)
    aulc_vals = np.array(aulc_vals, dtype=float)
    aulc_mu = float(np.nanmean(aulc_vals))
    aulc_sd = float(np.nanstd(aulc_vals))

    # Final accuracy at end of grid
    finals = np.array([y[-1] if np.isfinite(y[-1]) else np.nan for y in Ys], dtype=float)
    final_mu = float(np.nanmean(finals))
    final_sd = float(np.nanstd(finals))

    return {
        "grid_f": grid_f, "mean_y": mean_y, "std_y": std_y,
        "aulc_mu": aulc_mu, "aulc_sd": aulc_sd,
        "final_mu": final_mu, "final_sd": final_sd,
        "f_max": f_max_common,
    }


def main():
    # Load runs for all methods
    all_runs = {m: load_runs_for_method(m) for m in METHODS}

    # Compute curves + AULC summaries
    curves = {}
    rows = []
    for method, dfs in all_runs.items():
        s = compute_aulc_from_runs(dfs, METRIC_KEY, N_GRID)
        if s is None:
            print(f"[WARN] Skipping {method}: insufficient data.")
            continue
        curves[method] = s
        rows.append({
            "method": method,
            "f_max_common": s["f_max"],
            "AULC_mean": s["aulc_mu"],
            "AULC_std":  s["aulc_sd"],
            "FinalAcc_mean": s["final_mu"],
            "FinalAcc_std":  s["final_sd"],
        })

    # Save summary table
    if rows:
        summary_df = pd.DataFrame(rows)#.sort_values("AULC_mean", ascending=False)
        method_order = [
            "random", "uncertainty", "entropy", "margins", "bald", 
            "powerbald", "coreset", "badge", "alfamix", "probcover", "disagreement"
        ]
        summary_df['method_cat'] = pd.Categorical(
            summary_df['method'], 
            categories=method_order, 
            ordered=True
        )
        summary_df = summary_df.sort_values('method_cat')
        out_csv = os.path.join(OUT_AGG_DIR, "summary_aulc_finalacc.csv")
        summary_df.to_csv(out_csv, index=False)
        print(f"[OK] Wrote AULC summary: {out_csv}")
    else:
        print("[WARN] No methods produced summaries; nothing to plot.")
        return

    # Plot curves (Accuracy vs labeled fraction with mean±std bands)
    plt.figure(figsize=(9, 6))
    for method, s in curves.items():
        x, y, sd = s["grid_f"], s["mean_y"], s["std_y"]
        plt.plot(x, y, label=method)
        if np.isfinite(sd).any():
            plt.fill_between(x, y - sd, y + sd, alpha=0.15)
    plt.xlabel("Labeled fraction")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs labeled fraction")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(title="Method", ncol=2, fontsize=9)
    plt.tight_layout()
    out_png = os.path.join(OUT_PLOTS_DIR, "curves_fraction_accuracy.png")
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved plot: {out_png}")

    # Plot AULC bar chart
    sdf = pd.read_csv(os.path.join(OUT_AGG_DIR, "summary_aulc_finalacc.csv"))
    # Keep only methods we actually have
    sdf = sdf[sdf["method"].isin(curves.keys())]

    method_order = [
        "random", "uncertainty", "entropy", "margins", "bald", 
        "powerbald", "coreset", "badge", "alfamix", "probcover", "disagreement"
    ]
    # Filter to only methods that exist in our data
    order = [m for m in method_order if m in sdf["method"].values]

    plt.figure(figsize=(10, 4.8))
    means = sdf.set_index("method").loc[order, "AULC_mean"].values
    stds  = sdf.set_index("method").loc[order, "AULC_std"].values
    xs = np.arange(len(order))
    plt.bar(xs, means, yerr=stds, capsize=3)
    plt.xticks(xs, order, rotation=30, ha="right")
    plt.ylabel("AULC (normalized)")
    plt.title("Area Under Learning Curve (mean ± std)")
    plt.tight_layout()
    out_bar_pgf = os.path.join(OUT_PLOTS_DIR, "bar_aulc_cifar100.pgf")
    plt.savefig(out_bar_pgf, dpi=220)
    out_bar = os.path.join(OUT_PLOTS_DIR, "bar_aulc.png")
    plt.savefig(out_bar, dpi=220)
    plt.close()
    print(f"[OK] Saved AULC bar: {out_bar}")

    # Optional: bar chart for Final Accuracy
    plt.figure(figsize=(10, 4.8))
    f_means = sdf.set_index("method").loc[order, "FinalAcc_mean"].values
    f_stds  = sdf.set_index("method").loc[order, "FinalAcc_std"].values
    plt.bar(xs, f_means, yerr=f_stds, capsize=3)
    plt.xticks(xs, order, rotation=30, ha="right")
    plt.ylabel("Final accuracy")
    plt.title("Final Accuracy at common budget (mean ± std)")
    plt.tight_layout()
    out_bar_final_pgf = os.path.join(OUT_PLOTS_DIR, "bar_finalacc_cifar100.pgf")
    plt.savefig(out_bar_final_pgf, dpi=220)
    out_bar_final = os.path.join(OUT_PLOTS_DIR, "bar_finalacc.png")
    plt.savefig(out_bar_final, dpi=220)
    plt.close()
    print(f"[OK] Saved Final-Acc bar: {out_bar_final}")


if __name__ == "__main__":
    main()
