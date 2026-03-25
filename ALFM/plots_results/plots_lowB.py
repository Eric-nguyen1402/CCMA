import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
})
# --------- Config ---------
BASE_DIR = "ALFM/logs/results/domainnetreal/dino_vit_g14"
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
    # "typiclust",
    "probcover",
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
            # normalize expected columns just in case casing/whitespace differs
            df.columns = [c.strip() for c in df.columns]
            required = {"iteration", "num_samples", "TEST_MulticlassAccuracy",
                        "TEST_MulticlassAUROC", "TEST_MulticlassF1Score"}
            missing = required - set(df.columns)
            if missing:
                print(f"[WARN] {os.path.basename(p)} missing columns: {missing}; skipping this file.")
                continue
            df["run_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return dfs

def aggregate_method(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Average across runs per num_samples (and iteration), compute mean & std."""
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    # Group by num_samples (and iteration if you want to be extra safe)
    group_cols = ["iteration", "num_samples"]
    metric_cols = ["TEST_MulticlassAccuracy", "TEST_MulticlassAUROC", "TEST_MulticlassF1Score"]
    agg = (
        all_df.groupby(group_cols, as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    # Flatten columns
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    # Keep iteration/num_samples as normal columns
    # After groupby with multiple agg, they become part of MultiIndex; reset names:
    agg = agg.reset_index()
    return agg

def save_aggregated_csv(method: str, agg_df: pd.DataFrame):
    if agg_df.empty:
        return
    out_path = os.path.join(OUT_AGG_DIR, f"{method}_averaged.csv")
    agg_df.to_csv(out_path, index=False)

def plot_metric(all_aggs: dict, metric_key: str, metric_label: str):
    """Plot a single metric across all methods with mean line and std band."""
    plt.figure(figsize=(9, 6))
    for method, df in all_aggs.items():
        if df.empty:  # skip methods with no data
            continue
        mean_col = f"{metric_key}_mean"
        std_col = f"{metric_key}_std"
        # Ensure sorting by x
        df_sorted = df.sort_values("num_samples", kind="mergesort")
        # MODIFICATION: Slice the DataFrame to only plot the first 8 points (index 0 to 7)
        df_sliced = df_sorted.iloc[:8]
        x = df_sliced["num_samples"].values
        y = df_sliced[mean_col].values
        s = df_sliced[std_col].values
        # Line
        plt.plot(x, y, label=method)
        # Std shading (only if std exists)
        if std_col in df_sliced.columns and not df_sliced[std_col].isna().all():
            plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("num_samples")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs num_samples (DomainNet-Real, dino_vit_g14)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(title="Method", ncol=2, fontsize=9)
    plt.tight_layout()
    out_pgf = os.path.join(OUT_PLOTS_DIR, f"compare_domainnetreal_{metric_label.lower()}_lowB.pgf")
    plt.savefig(out_pgf, dpi=200)
    out_png = os.path.join(OUT_PLOTS_DIR, f"compare_domainnetreal_{metric_label.lower()}_lowB.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved plot: {out_pgf}")

def main():
    all_aggregated = {}
    for method in METHODS:
        print(f"== Processing method: {method} ==")
        dfs = load_runs_for_method(method)
        agg = aggregate_method(dfs)
        # if agg.empty:
        #     print(f"[WARN] No aggregated data for {method}.")
        # else:
        #     save_aggregated_csv(method, agg)
        all_aggregated[method] = agg

    # Plot each metric on its own figure
    for metric_key, metric_label in METRICS:
        plot_metric(all_aggregated, metric_key, metric_label)

    # Optional: create a single wide table merging all methods' mean accuracy by num_samples
    # (handy for quick inspection/LaTeX tables). Comment out if you don't need it.
    # try:
    #     merged = None
    #     for method, df in all_aggregated.items():
    #         if df.empty:
    #             continue
    #         slim = df[["num_samples", f"TEST_MulticlassAccuracy_mean"]].copy()
    #         slim = slim.rename(columns={f"TEST_MulticlassAccuracy_mean": method})
    #         merged = slim if merged is None else pd.merge(merged, slim, on="num_samples", how="outer")
    #     if merged is not None:
    #         merged = merged.sort_values("num_samples")
    #         out_csv = os.path.join(OUT_AGG_DIR, "accuracy_means_by_method.csv")
    #         merged.to_csv(out_csv, index=False)
    #         print(f"[OK] Wrote table of mean accuracies: {out_csv}")
    # except Exception as e:
    #     print(f"[WARN] Could not create the merged accuracy table: {e}")

if __name__ == "__main__":
    main()
