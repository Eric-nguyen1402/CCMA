import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
})

# --------- Config ---------
BASE_DIR = "ALFM/logs/results/caltech256/dino_vit_g14"
METHODS = [
    "random",
    "uncertainty",
    "entropy",
    "margins",
    "coreset",
    "bald",
    "powerbald",
    "badge",
    "alfamix",
    "typiclust",
    # "probcover",
    "disagreement",
]
METRICS = [
    ("TEST_MulticlassAccuracy", "Accuracy"),
    ("TEST_MulticlassAUROC", "AUROC"),
    ("TEST_MulticlassF1Score", "F1"),
]
OUT_AGG_DIR = os.path.join(BASE_DIR, "aggregated")
OUT_PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(OUT_AGG_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
# --------------------------


def load_runs_for_method(method: str) -> list[pd.DataFrame]:
    """Load all CSV runs for a given method."""
    pattern = os.path.join(BASE_DIR, f"{method}-*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[WARN] No CSVs found for method '{method}' with pattern {pattern}")
        return []
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            required = {"iteration", "num_samples", "TEST_MulticlassAccuracy",
                        "TEST_MulticlassAUROC", "TEST_MulticlassF1Score"}
            missing = required - set(df.columns)
            if missing:
                print(f"[WARN] {os.path.basename(p)} missing columns: {missing}; skipping.")
                continue
            df["run_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return dfs


def interpolate_and_average(dfs, metric_cols, x_grid):
    """Interpolate each run onto common x_grid, then average across runs."""
    if not dfs:
        return pd.DataFrame()

    all_interp = {m: [] for m in metric_cols}
    for df in dfs:
        df_unique = df.groupby("num_samples", as_index=False)[metric_cols].mean()
        df_sorted = df.sort_values("num_samples")
        x = df_sorted["num_samples"].values
        for m in metric_cols:
            y = df_sorted[m].values
            f = interp1d(x, y, kind="linear",
                         bounds_error=False, fill_value="extrapolate")
            all_interp[m].append(f(x_grid))

    out = {"num_samples": x_grid}
    for m in metric_cols:
        arr = np.vstack(all_interp[m])
        out[f"{m}_mean"] = arr.mean(axis=0)
        out[f"{m}_std"] = arr.std(axis=0)
    return pd.DataFrame(out)


def aggregate_method(dfs: list[pd.DataFrame],
                     max_samples: int,
                     step: int = 100) -> pd.DataFrame:
    """Average across runs by interpolating onto a common grid."""
    if not dfs:
        return pd.DataFrame()
    metric_cols = ["TEST_MulticlassAccuracy",
                   "TEST_MulticlassAUROC",
                   "TEST_MulticlassF1Score"]
    x_grid = np.arange(step, max_samples + 1, step)
    return interpolate_and_average(dfs, metric_cols, x_grid)


def save_aggregated_csv(method: str, agg_df: pd.DataFrame):
    if agg_df.empty:
        return
    out_path = os.path.join(OUT_AGG_DIR, f"{method}_averaged.csv")
    agg_df.to_csv(out_path, index=False)


def plot_metric(all_aggs: dict, metric_key: str, metric_label: str,
                low_budget_cut: int = None):
    """Plot a single metric across all methods with mean line and std band."""
    plt.figure(figsize=(9, 6))
    for method, df in all_aggs.items():
        if df.empty:
            continue
        mean_col = f"{metric_key}_mean"
        std_col = f"{metric_key}_std"

        df_sorted = df.sort_values("num_samples")
        if low_budget_cut is not None:
            df_sorted = df_sorted[df_sorted["num_samples"] <= low_budget_cut]

        x = df_sorted["num_samples"].values
        y = df_sorted[mean_col].values
        s = df_sorted[std_col].values

        plt.plot(x, y, label=method)
        if not df_sorted[std_col].isna().all():
            plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("Number of labeled samples")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs (Domainnet-Real, dino_vit_g14)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(title="Method", ncol=2, fontsize=9)
    plt.tight_layout()

    suffix = "lowB" if low_budget_cut else "full"
    out_pgf = os.path.join(
        OUT_PLOTS_DIR, f"compare_domainnetreal_{metric_label.lower()}_{suffix}.pgf")
    out_png = os.path.join(
        OUT_PLOTS_DIR, f"compare_domainnetreal_{metric_label.lower()}_{suffix}.png")
    plt.savefig(out_pgf, dpi=200)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {out_pgf}")


def main():
    all_aggregated = {}
    # infer max_samples from Random (or any method with full budget)
    random_dfs = load_runs_for_method("random")
    if not random_dfs:
        raise RuntimeError("Random method logs not found; needed to infer max_samples.")
    max_samples = max(df["num_samples"].max() for df in random_dfs)

    for method in METHODS:
        print(f"== Processing method: {method} ==")
        dfs = load_runs_for_method(method)
        agg = aggregate_method(dfs, max_samples=max_samples, step=100)
        save_aggregated_csv(method, agg)
        all_aggregated[method] = agg

    # Plot each metric (both low budget cut and full)
    for metric_key, metric_label in METRICS:
        # Low budget regime (e.g., <= 800 samples)
        plot_metric(all_aggregated, metric_key, metric_label,
                    low_budget_cut=800)
        # Full budget regime
        plot_metric(all_aggregated, metric_key, metric_label,
                    low_budget_cut=None)


if __name__ == "__main__":
    main()
