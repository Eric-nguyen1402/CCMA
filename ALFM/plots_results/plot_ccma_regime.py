import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
})

# ------------- Config (no CLI args; mirrors your plotting script) -------------
BASE_DIRS = ["ALFM/logs/results/food101/dino_vit_g14"]
METHOD_NAME = "disagreement"
OUT_SUBDIR = "ccma_plots"
# ------------------------------------------------------------------------------

REQUIRED_BASE = {"iteration", "num_samples", "labeled"}

def load_diag_runs(base_dir: str) -> list[pd.DataFrame]:
    pattern = os.path.join(base_dir, f"{METHOD_NAME}-*-diag.csv")
    paths = sorted(glob.glob(pattern))
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            if not REQUIRED_BASE.issubset(df.columns):
                print(f"[WARN] Missing required columns in {p}; skip.")
                continue
            df["run_file"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return dfs

def aggregate_by_labeled(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Average across runs per *cumulative* labeled count, compute mean & std."""
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)

    keep_cols = [
        # x/keys
        "iteration", "num_samples", "labeled",
        # diagnostics
        "student_top1_mean", "student_entropy_mean",
        "teacher_top1_mean", "teacher_entropy_mean",
        "frac_disagree_top1", "js_full_mean",
        "CCMA_GI_mean", "CCMA_GT_mean",
        "CCMA_overlap_mean", "CCMA_symdiff_mean", "CCMA_identical_pct",
        # knobs (kept in case you want to inspect later)
        "temperature", "target_set_size_teacher", "target_set_size_student", "lam",
    ]
    have = [c for c in keep_cols if c in all_df.columns]
    all_df = all_df[have].copy()

    # Group by cumulative labeled (optionally also iteration for stability)
    group_cols = ["labeled"]  # using just 'labeled' gives clean x-axis
    metric_cols = [c for c in have if c not in ("iteration","num_samples","labeled")]

    agg = (
        all_df.groupby(group_cols, as_index=False)[metric_cols]
              .agg(["mean","std"])
    )
    # Flatten MultiIndex columns
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    agg = agg.reset_index()
    return agg

def plot_curves(agg: pd.DataFrame, title_prefix: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def draw(x, y, s, ylabel, fname):
        plt.figure(figsize=(9,6))
        plt.plot(x, y, label="mean")
        if s is not None:
            plt.fill_between(x, y - s, y + s, alpha=0.15)
        plt.xlabel("Cumulative labeled examples")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} — {ylabel} vs labeled")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.tight_layout()

        # Save PNG
        out_png = os.path.join(out_dir, fname)
        plt.savefig(out_png, dpi=200)
        print(f"[OK] saved {out_png}")
        
        # Save PGF
        fname_base = os.path.splitext(fname)[0]  # Remove .png extension
        out_pgf = os.path.join(out_dir, f"{fname_base}.pgf")
        plt.savefig(out_pgf)
        print(f"[OK] saved {out_pgf}")

    def draw_combined_confidence(x, student_y, student_s, teacher_y, teacher_s, ylabel, fname):
        plt.figure(figsize=(9,6))
        
        # Plot student line
        plt.plot(x, student_y, label="Student", color='blue')
        if student_s is not None:
            plt.fill_between(x, student_y - student_s, student_y + student_s, alpha=0.15, color='blue')
        
        # Plot teacher line
        plt.plot(x, teacher_y, label="Teacher", color='red')
        if teacher_s is not None:
            plt.fill_between(x, teacher_y - teacher_s, teacher_y + teacher_s, alpha=0.15, color='red')
            
        plt.xlabel("Cumulative labeled examples")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} — {ylabel} vs labeled")
        plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Save PNG
        out_png = os.path.join(out_dir, fname)
        plt.savefig(out_png, dpi=200)
        print(f"[OK] saved {out_png}")
        
        # Save PGF
        fname_base = os.path.splitext(fname)[0]  # Remove .png extension
        out_pgf = os.path.join(out_dir, f"{fname_base}.pgf")
        plt.savefig(out_pgf)
        print(f"[OK] saved {out_pgf}")
    
    if agg.empty:
        print("[WARN] Empty aggregate; nothing to plot.")
        return

    df = agg.sort_values("labeled", kind="mergesort")
    x = df["labeled"].values
    # Confidence
    if "student_top1_mean_mean" in df.columns and "teacher_top1_mean_mean" in df.columns:
        student_y = df["student_top1_mean_mean"].values
        student_s = df.get("student_top1_mean_std", pd.Series(np.nan, index=df.index)).values
        teacher_y = df["teacher_top1_mean_mean"].values  
        teacher_s = df.get("teacher_top1_mean_std", pd.Series(np.nan, index=df.index)).values
        
        draw_combined_confidence(x, student_y, student_s, teacher_y, teacher_s,
                               "Top-1 Confidence", "combined_top1_confidence.png")
    
    # If only one of them exists, fall back to individual plots
    elif "student_top1_mean_mean" in df.columns:
        draw(x,
             df["student_top1_mean_mean"].values,
             df.get("student_top1_mean_std", pd.Series(np.nan, index=df.index)).values,
             "Student top-1 confidence", "student_top1.png")
    
    elif "teacher_top1_mean_mean" in df.columns:
        draw(x,
             df["teacher_top1_mean_mean"].values,
             df.get("teacher_top1_mean_std", pd.Series(np.nan, index=df.index)).values,
             "Teacher top-1 confidence", "teacher_top1.png")

    # Disagreement rate
    if "frac_disagree_top1_mean" in df.columns:
        x = df["labeled"].values
        draw(x,
             df["frac_disagree_top1_mean"].values,
             df.get("frac_disagree_top1_std", pd.Series(np.nan, index=df.index)).values,
             "Frac. top-1 disagree", "frac_disagree.png")

    # JS (full support)
    if "js_full_mean_mean" in df.columns:
        x = df["labeled"].values
        draw(x,
             df["js_full_mean_mean"].values,
             df.get("js_full_mean_std", pd.Series(np.nan, index=df.index)).values,
             "JS divergence (full)", "js_full.png")

    # CCMA stats
    ccma_pairs = [
        ("CCMA_GI_mean",        "Set size |$\\Gamma_I$|"),
        ("CCMA_GT_mean",        "Set size |$\\Gamma_T$|"),
        ("CCMA_overlap_mean",   "Overlap |$\\Gamma_I \\cap \\Gamma_T$|"),
        ("CCMA_symdiff_mean",   "Symmetric diff |$\\Gamma_I \\triangle \\Gamma_T$|"),
        ("CCMA_identical_pct",  "% identical sets"),
    ]
    for key, nice in ccma_pairs:
        m, s = f"{key}_mean", f"{key}_std"
        if m in df.columns:
            x = df["labeled"].values
            draw(x, df[m].values, df.get(s, pd.Series(np.nan, index=df.index)).values,
                 nice, f"{key}.png")

def teacher_usefulness_summary(agg: pd.DataFrame) -> dict:
    """Heuristic summary to discuss 'when teacher helps' using labeled as x."""
    if agg.empty:
        return {}
    df = agg.sort_values("labeled")

    # Use about the first/last 3 labeled points (or fewer if not available)
    k = min(3, len(df))
    early = df.head(k)
    late  = df.tail(k)

    t_early = float(early.get("teacher_top1_mean_mean", pd.Series([np.nan]*k)).mean())
    s_early = float(early.get("student_top1_mean_mean", pd.Series([np.nan]*k)).mean())
    d_early = float(early.get("frac_disagree_top1_mean", pd.Series([np.nan]*k)).mean())

    t_late = float(late.get("teacher_top1_mean_mean", pd.Series([np.nan]*k)).mean())
    s_late = float(late.get("student_top1_mean_mean", pd.Series([np.nan]*k)).mean())
    d_late = float(late.get("frac_disagree_top1_mean", pd.Series([np.nan]*k)).mean())

    return {
        "teacher>student_early": (t_early - s_early),
        "teacher>student_late":  (t_late - s_late),
        "disagree_early": d_early,
        "disagree_late":  d_late,
    }

def main():
    for base in BASE_DIRS:
        print(f"\n=== Dataset/Backbone dir: {base} ===")
        dfs = load_diag_runs(base)
        if not dfs:
            print("[WARN] No diag CSVs found.")
            continue
        agg = aggregate_by_labeled(dfs)
        ds_name = Path(base).parts[-2] if len(Path(base).parts) >= 2 else base
        out = os.path.join(base, OUT_SUBDIR)
        plot_curves(agg, title_prefix=f"{ds_name} / {METHOD_NAME}", out_dir=out)
        summary = teacher_usefulness_summary(agg)
        if summary:
            print(f"[Summary] {ds_name}: {summary}")

if __name__ == "__main__":
    main()
