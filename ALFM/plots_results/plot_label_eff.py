# save as: tools/label_efficiency_noconfig.py
# Runs with no CLI args. Configure BASE_DIR/METHODS/THRESHOLDS below.

import os, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
})

# ========= CONFIG (mirror your AULC plotting script) =========
BASE_DIR = "ALFM/logs/results/domainnetreal/dino_vit_g14"
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
THRESHOLDS = [80, 85, 90]  # target accuracies in %
OUT_DIR = os.path.join(BASE_DIR, "label_efficiency")
os.makedirs(OUT_DIR, exist_ok=True)
# =============================================================

def load_runs_for_method(method: str) -> list[pd.DataFrame]:
    """Load CSVs for a method, ensure required columns, sort by num_samples."""
    pattern = os.path.join(BASE_DIR, f"{method}-*.csv")
    paths = sorted(glob.glob(pattern))
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            req = {"iteration", "num_samples", "TEST_MulticlassAccuracy"}
            if not req.issubset(df.columns):
                print(f"[WARN] {os.path.basename(p)} missing {req - set(df.columns)}; skip")
                continue
            dfs.append(df.sort_values("num_samples", kind="mergesort"))
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    if not dfs:
        print(f"[WARN] No CSVs found for method '{method}' with pattern {pattern}")
    return dfs

def interp_labels_for_threshold(x: np.ndarray, y: np.ndarray, thr: float) -> float | None:
    """
    x: labeled counts (monotone increasing)
    y: accuracy series (0..1 or 0..100). thr in percent, e.g. 85.
    Returns interpolated labels where y crosses thr, else None if not reached.
    """
    y_ = y.astype(float).copy()
    if y_.max() <= 1.0 + 1e-6:  # treat as 0..1
        y_ *= 100.0
    for i in range(1, len(x)):
        if y_[i-1] < thr <= y_[i]:
            # linear interpolation between the two rounds
            denom = max(1e-9, (y_[i] - y_[i-1]))
            t = (thr - y_[i-1]) / denom
            return float(x[i-1] + t * (x[i] - x[i-1]))
    return None

def summarize_method(method: str):
    """Per-seed interpolation → aggregate mean±std for each threshold."""
    dfs = load_runs_for_method(method)
    if not dfs:
        return None, None

    per_run_rows = []
    for seed_i, df in enumerate(dfs):
        x = df["num_samples"].to_numpy()
        y = df["TEST_MulticlassAccuracy"].to_numpy()
        row = {"method": method, "run": seed_i}
        for thr in THRESHOLDS:
            n = interp_labels_for_threshold(x, y, thr)
            row[f"labels@{thr}"] = n  # might be None if never reached
        per_run_rows.append(row)

    per_run = pd.DataFrame(per_run_rows)

    # aggregate across seeds (ignore runs that never reached threshold)
    agg = {"method": method}
    for thr in THRESHOLDS:
        k = f"labels@{thr}"
        vals = per_run[k].dropna().to_numpy()
        agg[f"{k}_mean"] = float(vals.mean()) if len(vals) else math.nan
        agg[f"{k}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
        agg[f"{k}_n"]    = int(len(vals))
    agg_df = pd.DataFrame([agg])

    # save per-method per-run CSV
    per_run.to_csv(os.path.join(OUT_DIR, f"{method}_labels_per_run.csv"), index=False)
    return per_run, agg_df

def make_barplot(summary: pd.DataFrame, out_base: str, title: str):
    # sort by hardest threshold (last one) for readability
    method_order = [
        "random", "uncertainty", "entropy", "margins", "bald", 
        "powerbald", "coreset", "badge", "alfamix", "probcover", "disagreement"
    ]
    
    # Create a categorical column to enforce the order
    summary['method_cat'] = pd.Categorical(
        summary['method'], 
        categories=method_order, 
        ordered=True
    )
    summary = summary.sort_values('method_cat')

    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25 if len(THRESHOLDS) == 3 else 0.8 / max(1, len(THRESHOLDS))

    plt.figure(figsize=(10, 5.2))
    for j, thr in enumerate(THRESHOLDS):
        mu = summary[f"labels@{thr}_mean"].to_numpy()
        sd = summary[f"labels@{thr}_std"].to_numpy()
        plt.bar(x + (j - (len(THRESHOLDS)-1)/2)*width, mu, yerr=sd,
                width=width, capsize=3, label=f"{thr}%")

    plt.xticks(x, methods, rotation=30, ha="right")
    plt.ylabel("Labeled samples needed")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Target accuracy")
    plt.tight_layout()

    if out_base.endswith('.pgf'):
        base_name = out_base[:-4]  # Remove .pgf extension
    elif out_base.endswith('.png'):
        base_name = out_base[:-4]  # Remove .png extension
    else:
        base_name = out_base
    
    # Save PNG
    out_png = f"{base_name}.png"
    plt.savefig(out_png, dpi=200)
    print(f"[OK] saved plot: {out_png}")
    
    # Save PGF
    out_pgf = f"{base_name}.pgf"
    plt.savefig(out_pgf)
    print(f"[OK] saved plot: {out_pgf}")
    plt.close()
    

def main():
    per_run_all = []
    agg_all = []

    for m in METHODS:
        print(f"== Processing method: {m} ==")
        per_run, agg = summarize_method(m)
        if per_run is None or agg is None:
            continue
        per_run_all.append(per_run)
        agg_all.append(agg)

    if not agg_all:
        print("[WARN] No aggregates produced.")
        return

    summary = pd.concat(agg_all, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "labels_to_threshold_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[OK] wrote summary: {out_csv}")

    # optional: keep a single CSV with all per-run rows
    pd.concat(per_run_all, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "labels_to_threshold_per_run_all.csv"), index=False
    )

    # bar plot
    make_barplot(
        summary,
        os.path.join(OUT_DIR, "labels_to_threshold_bars_domainnetreal"),
        title="Labels needed to reach target accuracy"
    )

if __name__ == "__main__":
    main()
