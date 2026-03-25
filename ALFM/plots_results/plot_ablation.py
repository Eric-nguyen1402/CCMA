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

# ----------------------- Config -----------------------
BASE_DIR   = "ALFM/logs/results/cifar100/dino_vit_g14"   # <- change per dataset
METHOD     = "disagreement"
VARIANTS   = ["V1","V2","V3","V4","V5"]             # order = ablation path
BASELINE_VARIANT = "V1"                                   # for “speedup” column
OUT_DIR    = os.path.join(BASE_DIR, "aggregated")
PLOTS_DIR  = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# timing columns (we’ll take intersection with what’s present)
TIMING_STEP_COLS = [
    "t_subpool","t_modalities","t_sets","t_score","t_topk","t_diversity","t_query_total",
    # amortized columns are fine; averaged if present
    "t_modalities_per1k","t_sets_per1k","t_score_per1k","t_topk_per1k","t_diversity_per1k_sel"
]
METRIC_COLS = ["TEST_MulticlassAccuracy","TEST_MulticlassAUROC","TEST_MulticlassF1Score"]
GROUP_COLS  = ["iteration","num_samples"]  # aggregate per round
# ------------------------------------------------------


# ---------- Load per-seed CSVs ----------
def _load_timing_files(variant: str) -> list[pd.DataFrame]:
    pat = os.path.join(BASE_DIR, f"{METHOD}-{variant}-timing-*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        print(f"[WARN] No timing files for {variant} with pattern {pat}")
        return []
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            df["__src__"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return dfs

def _load_metric_files(variant: str) -> list[pd.DataFrame]:
    pat = os.path.join(BASE_DIR, f"{METHOD}_{variant}-*.csv")
    paths = sorted(glob.glob(pat))
    if not paths:
        print(f"[WARN] No metric files for {variant} with pattern {pat}")
        return []
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            need = set(GROUP_COLS) | set(METRIC_COLS)
            if not need.issubset(df.columns):
                print(f"[WARN] {os.path.basename(p)} missing {need - set(df.columns)}; skipping.")
                continue
            df["__src__"] = os.path.basename(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    return dfs

# ---------- Aggregate helpers ----------
def _agg_mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    have = [c for c in value_cols if c in df.columns]
    if not have:
        return pd.DataFrame()
    g = (
        df.groupby(group_cols, as_index=False)[have]
          .agg(["mean","std"])
          .reset_index()
    )
    g.columns = ['_'.join(col).strip('_') for col in g.columns.to_flat_index()]
    return g

def _save(df: pd.DataFrame, path: str):
    if df is not None and not df.empty:
        df.to_csv(path, index=False)
        print(f"[OK] wrote {path}")
    else:
        print(f"[WARN] nothing to write for {path}")

def _overall_step_means(timing_agg: pd.DataFrame) -> pd.DataFrame:
    if timing_agg is None or timing_agg.empty:
        return pd.DataFrame()
    step_mean_cols = [c for c in timing_agg.columns if c.endswith("_mean") and any(c.startswith(s) for s in TIMING_STEP_COLS)]
    tq_mean = "t_query_total_mean" if "t_query_total_mean" in timing_agg.columns else None

    row = {}
    for col in step_mean_cols:
        step = col.replace("_mean","")
        row[step] = float(timing_agg[col].mean())

    overall = pd.DataFrame([row]).T.reset_index()
    overall.columns = ["step","mean_seconds"]

    # total for percentages
    if tq_mean is not None:
        total = float(timing_agg[tq_mean].mean())
    else:
        core = [s for s in ["t_subpool","t_modalities","t_sets","t_score","t_topk","t_diversity"] if f"{s}_mean" in timing_agg.columns]
        total = float(sum(timing_agg[f"{s}_mean"].mean() for s in core)) if core else float(overall["mean_seconds"].sum())

    overall["percent_of_total"] = np.where(total > 0, 100.0 * overall["mean_seconds"]/total, np.nan)
    overall = overall.sort_values("mean_seconds", ascending=False).reset_index(drop=True)
    return overall

# AULC over rounds (normalized 0..1). Uses accuracy mean per iteration.
def _aulc_over_rounds(metrics_agg: pd.DataFrame, acc_col="TEST_MulticlassAccuracy_mean") -> float:
    if metrics_agg.empty or acc_col not in metrics_agg.columns:
        return np.nan
    df = metrics_agg.sort_values("iteration").copy()
    it = df["iteration"].values.astype(float)
    y  = df[acc_col].values.astype(float)
    if len(it) < 2:
        return float(y[-1]) if len(y) else np.nan
    x = (it - it.min()) / (it.max() - it.min())  # normalize 0..1
    return float(np.trapz(y, x))

# Mean query time/round (average across iterations of total query time)
def _mean_query_time_per_round(timing_agg: pd.DataFrame) -> float:
    if timing_agg.empty:
        return np.nan
    if "t_query_total_mean" in timing_agg.columns:
        return float(timing_agg["t_query_total_mean"].mean())
    # else: sum core steps for each iteration then average
    core = [c for c in ["t_subpool_mean","t_modalities_mean","t_sets_mean","t_score_mean","t_topk_mean","t_diversity_mean"] if c in timing_agg.columns]
    if not core:
        return np.nan
    tot = timing_agg[core].sum(axis=1)
    return float(tot.mean())

def aggregate_one_variant(variant: str):
    print(f"\n=== Aggregating {variant} ===")
    # try reusing already-aggregated files (idempotent)
    timing_agg_path = os.path.join(OUT_DIR, f"{variant}_timing_agg.csv")
    metrics_agg_path = os.path.join(OUT_DIR, f"{variant}_metrics_agg.csv")

    if os.path.exists(timing_agg_path):
        timing_agg = pd.read_csv(timing_agg_path)
    else:
        tdfs = _load_timing_files(variant)
        timing_agg = pd.DataFrame()
        if tdfs:
            T = pd.concat(tdfs, ignore_index=True)
            value_cols = [c for c in TIMING_STEP_COLS if c in T.columns]
            if value_cols:
                timing_agg = _agg_mean_std(T, GROUP_COLS, value_cols)
                _save(timing_agg, timing_agg_path)

    if os.path.exists(metrics_agg_path):
        metrics_agg = pd.read_csv(metrics_agg_path)
    else:
        mdfs = _load_metric_files(variant)
        metrics_agg = pd.DataFrame()
        if mdfs:
            M = pd.concat(mdfs, ignore_index=True)
            metrics_agg = _agg_mean_std(M, GROUP_COLS, METRIC_COLS)
            _save(metrics_agg, metrics_agg_path)

    # step shares
    timing_steps_path = os.path.join(OUT_DIR, f"{variant}_timing_step_means.csv")
    if not os.path.exists(timing_steps_path) and not timing_agg.empty:
        overall_steps = _overall_step_means(timing_agg)
        _save(overall_steps, timing_steps_path)

    return timing_agg, metrics_agg

# ---------- Cross-variant summary & plots ----------
def build_cross_variant_summary():
    rows = []
    per_variant_steps = {}

    for v in VARIANTS:
        t_agg, m_agg = aggregate_one_variant(v)

        aulc = _aulc_over_rounds(m_agg, acc_col="TEST_MulticlassAccuracy_mean") if not m_agg.empty else np.nan
        final_acc = np.nan
        if not m_agg.empty and "TEST_MulticlassAccuracy_mean" in m_agg.columns:
            final_acc = float(m_agg.sort_values("iteration")["TEST_MulticlassAccuracy_mean"].iloc[-1])

        mean_t = _mean_query_time_per_round(t_agg)
        aulc_per_sec = (aulc / mean_t) if (np.isfinite(aulc) and np.isfinite(mean_t) and mean_t>0) else np.nan

        rows.append({"variant": v, "AULC": aulc, "FinalAcc": final_acc, "MeanTimePerRound": mean_t, "AULC_per_sec": aulc_per_sec})

        # time shares for stacked bar
        steps_path = os.path.join(OUT_DIR, f"{v}_timing_step_means.csv")
        if os.path.exists(steps_path):
            per_variant_steps[v] = pd.read_csv(steps_path)

    summary = pd.DataFrame(rows)

    # speedup vs baseline
    if BASELINE_VARIANT in summary["variant"].values:
        t0 = float(summary.loc[summary["variant"]==BASELINE_VARIANT, "MeanTimePerRound"].values[0])
        if np.isfinite(t0) and t0>0:
            summary["Speedup_vs_"+BASELINE_VARIANT] = t0 / summary["MeanTimePerRound"]

    # save
    sum_path = os.path.join(OUT_DIR, "summary_variants.csv")
    summary.to_csv(sum_path, index=False)
    print(f"[OK] wrote {sum_path}")

    return summary, per_variant_steps

def plot_pareto(summary: pd.DataFrame):
    df = summary.dropna(subset=["AULC","MeanTimePerRound"]).copy()
    if df.empty:
        print("[WARN] Pareto plot skipped (missing data).")
        return
    plt.figure(figsize=(5.5,4.5))
    plt.scatter(df["MeanTimePerRound"], df["AULC"])
    # labels
    for _, r in df.iterrows():
        plt.annotate(r["variant"], (r["MeanTimePerRound"], r["AULC"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    plt.xlabel("Mean query time per round (s)")
    plt.ylabel("AULC (over rounds)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pareto_aulc_vs_time.png")
    plt.savefig(out, dpi=220)
    out_pgf = os.path.join(PLOTS_DIR, "pareto_aulc_vs_time.pgf")
    plt.savefig(out_pgf, dpi=220)
    plt.close()
    print(f"[OK] saved {out}")

def plot_ablation_ladder(summary: pd.DataFrame):
    # use the VARIANTS order as the ablation path
    df = summary.set_index("variant").reindex(VARIANTS)
    if df["AULC"].isna().all() or df["MeanTimePerRound"].isna().all():
        print("[WARN] Ablation ladder skipped (missing data).")
        return
    deltas = []
    for i in range(1, len(VARIANTS)):
        v_prev, v_now = VARIANTS[i-1], VARIANTS[i]
        if v_prev not in df.index or v_now not in df.index:
            continue
        a_prev, a_now = df.loc[v_prev, "AULC"], df.loc[v_now, "AULC"]
        t_prev, t_now = df.loc[v_prev, "MeanTimePerRound"], df.loc[v_now, "MeanTimePerRound"]
        if np.isfinite(a_prev) and np.isfinite(a_now) and np.isfinite(t_prev) and np.isfinite(t_now):
            deltas.append((f"{v_prev}→{v_now}", a_now - a_prev, t_now - t_prev))
    if not deltas:
        print("[WARN] No deltas for ablation ladder.")
        return

    labels, da, dt = zip(*deltas)
    x = np.arange(len(labels))
    w = 0.38

    plt.figure(figsize=(5.0,3.6))
    plt.bar(x - w/2, da, width=w, label="ΔAULC")
    plt.bar(x + w/2, dt, width=w, label="ΔTime (s)")
    plt.xticks(x, labels, rotation=0)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "ablation_ladder.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] saved {out}")

def plot_time_composition(per_variant_steps: dict):
    if not per_variant_steps:
        print("[WARN] Time composition skipped (no step files).")
        return
    # core steps we care about in stacked bars (order)
    core = ["t_subpool","t_modalities","t_sets","t_score","t_topk","t_diversity"]
    # build a table of % shares per variant
    table = []
    for v in VARIANTS:
        if v not in per_variant_steps: 
            continue
        df = per_variant_steps[v]
        row = {"variant": v}
        # prefer percent_of_total if given; else compute from mean_seconds
        if "percent_of_total" in df.columns.values and "step" in df.columns and "mean_seconds" in df.columns:
            # ensure we have entries for each core step
            for c in core:
                pct = float(df.loc[df["step"]==c, "percent_of_total"].values[0]) if (df["step"]==c).any() else 0.0
                row[c] = pct
        else:
            # fallback: compute % from mean seconds if present
            if "step" in df.columns and "mean_seconds" in df.columns:
                total = float(df["mean_seconds"].sum()) if df["mean_seconds"].sum() > 0 else 1.0
                for c in core:
                    ms = float(df.loc[df["step"]==c, "mean_seconds"].values[0]) if (df["step"]==c).any() else 0.0
                    row[c] = 100.0 * ms / total
        table.append(row)
    if not table:
        print("[WARN] No data for stacked bars.")
        return
    T = pd.DataFrame(table).set_index("variant").reindex(VARIANTS).fillna(0.0)

    # stacked bar
    plt.figure(figsize=(5.0,3.6))
    bottom = np.zeros(len(T))
    x = np.arange(len(T))
    for c in core:
        if c in T.columns:
            plt.bar(x, T[c].values, bottom=bottom, label=c.replace("t_",""))
            bottom += T[c].values
    plt.xticks(x, T.index)
    plt.ylabel("% of query time")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "time_composition.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] saved {out}")

def write_compact_latex_table(summary: pd.DataFrame, path: str):
    df = summary.copy()
    cols = ["variant","AULC","FinalAcc","MeanTimePerRound","AULC_per_sec"]
    sp_col = [c for c in df.columns if c.startswith("Speedup_vs_")]
    if sp_col:
        cols.append(sp_col[0])
    df = df[cols]
    # format to 3 decimals (or as you like)
    def fmt(x):
        if pd.isna(x): return "--"
        return f"{x:.3f}"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.8pt}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    head = ["Variant","AULC","Final@R","Time/rd [s]","AULC/s", ("Speedup" if sp_col else "")]
    head = [h for h in head if h!=""]
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        row = [str(r["variant"]), fmt(r["AULC"]), fmt(r["FinalAcc"]), fmt(r["MeanTimePerRound"]), fmt(r["AULC_per_sec"])]
        if sp_col:
            row.append(fmt(r[sp_col[0]]))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Mean over 5 seeds. AULC computed over rounds.}")
    lines.append(r"\label{tab:ablation-compact}")
    lines.append(r"\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] wrote {path}")

def main():
    summary, per_variant_steps = build_cross_variant_summary()
    # plots
    plot_pareto(summary)
    plot_ablation_ladder(summary)
    plot_time_composition(per_variant_steps)
    # latex
    write_compact_latex_table(summary, os.path.join(OUT_DIR, "summary_variants_table.tex"))

if __name__ == "__main__":
    main()
