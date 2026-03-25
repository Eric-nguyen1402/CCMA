# save as: tools/teacher_quality_regimes.py
# No CLI args. Edit BASE_DIR, REGIMES, and METRICS titles to your dataset/backbone.

import os, glob, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
})

# ========= CONFIG (mirror your AULC plotting style) =========
BASE_DIR = "ALFM/logs/results/food101/dino_vit_g14"

# Map of human-readable regime name -> filename glob pattern (without directory)
# Adjust patterns to your actual CSV naming.
REGIMES = {
    "teacher_high":   "disagreement_high-*.csv",    # τ≈0.03, T_set≈3–4
    "teacher_weak":   "disagreement_weak-*.csv",    # τ≈0.3–0.5, T_set≈12–16
    "student_only":   "disagreement_student-*.csv", # teacher disabled
}

# Which metrics to plot for "standard curves"
METRIC_KEY = "TEST_MulticlassAccuracy"
METRIC_LABEL = "Accuracy"

# Where to put outputs
OUT_DIR = os.path.join(BASE_DIR, "teacher_regimes")
os.makedirs(OUT_DIR, exist_ok=True)
# ============================================================

# Optional: CCMA columns if you already log them into the CSV each round.
# If they’re absent, we will try to parse .log files with regex.
CCMA_COLS = [
    "CCMA_GI_mean",       # mean |Γ_I|
    "CCMA_GT_mean",       # mean |Γ_T|
    "CCMA_overlap_mean",  # mean overlap
    "CCMA_symdiff_mean",  # mean symmetric difference
    "CCMA_identical_pct", # % identical sets (0..1 or 0..100)
]

def _find_sidecar_log(csv_path: str) -> str | None:
    """
    Try to find a same-named .log next to the CSV, or in BASE_DIR/logs/
    (Adjust if your logs are elsewhere.)
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    # same folder
    candidate1 = os.path.join(os.path.dirname(csv_path), stem + ".log")
    if os.path.isfile(candidate1):
        return candidate1
    # logs subdir
    candidate2 = os.path.join(os.path.dirname(csv_path), "logs", stem + ".log")
    if os.path.isfile(candidate2):
        return candidate2
    # BASE_DIR/logs/
    candidate3 = os.path.join(BASE_DIR, "logs", stem + ".log")
    if os.path.isfile(candidate3):
        return candidate3
    return None

_ccma_line = re.compile(
    r"\[CCMA\]\s+\|Γ_I\|\s+mean=(?P<gi>[\d\.]+).*?\|Γ_T\|\s+mean=(?P<gt>[\d\.]+).*?"
    r"overlap\s+mean=(?P<ov>[\d\.]+),\s+symdiff\s+mean=(?P<sd>[\d\.]+),\s+%identical=(?P<id>\d+(\.\d+)?%)"
)

def _parse_ccma_from_log(log_path: str) -> pd.DataFrame:
    """
    Grep CCMA snapshots from a log. We won't know the exact iteration/num_samples,
    so we return a simple row-indexed sequence; caller can align by count.
    """
    rows = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _ccma_line.search(line)
                if m:
                    gi = float(m.group("gi"))
                    gt = float(m.group("gt"))
                    ov = float(m.group("ov"))
                    sd = float(m.group("sd"))
                    pct = m.group("id")
                    val = float(pct.strip("%"))
                    # normalize to 0..1
                    if val > 1.0 + 1e-6:
                        val = val / 100.0
                    rows.append({
                        "CCMA_GI_mean": gi,
                        "CCMA_GT_mean": gt,
                        "CCMA_overlap_mean": ov,
                        "CCMA_symdiff_mean": sd,
                        "CCMA_identical_pct": val,
                    })
    except Exception as e:
        print(f"[WARN] Failed parsing CCMA from {log_path}: {e}")
    return pd.DataFrame(rows)

def _load_csvs_for_pattern(pattern: str) -> list[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(BASE_DIR, pattern)))
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            if not {"iteration", "num_samples", METRIC_KEY}.issubset(df.columns):
                print(f"[WARN] {os.path.basename(p)} missing required columns; skip")
                continue

            # try to attach CCMA columns:
            have_ccma = all(c in df.columns for c in CCMA_COLS)
            if not have_ccma:
                # attempt to parse a sidecar .log and append CCMA in order found
                logp = _find_sidecar_log(p)
                if logp:
                    cc = _parse_ccma_from_log(logp)
                    if not cc.empty:
                        # align by length; trim/pad to df length
                        k = min(len(cc), len(df))
                        for c in CCMA_COLS:
                            df[c] = np.nan
                        for c in CCMA_COLS:
                            df.loc[:k-1, c] = cc[c].values[:k]
                else:
                    # leave CCMA as NaN columns
                    for c in CCMA_COLS:
                        if c not in df.columns:
                            df[c] = np.nan

            df["run_file"] = os.path.basename(p)
            dfs.append(df.sort_values("num_samples", kind="mergesort"))
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")
    if not dfs:
        print(f"[WARN] No CSVs matched {pattern}")
    return dfs

def _mean_std_agg(dfs: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    """Aggregate mean±std per (iteration, num_samples) for the requested cols."""
    all_df = pd.concat(dfs, ignore_index=True)
    group_cols = ["iteration", "num_samples"]
    agg = all_df.groupby(group_cols, as_index=False)[cols].agg(["mean", "std"])
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns.values]
    agg = agg.reset_index()
    return agg

def _plot_curve(regime_aggs: dict, y_key: str, y_label: str, title: str, out_png: str):
    plt.figure(figsize=(9, 6))
    for regime, df in regime_aggs.items():
        if df is None or df.empty:
            continue
        df = df.sort_values("num_samples", kind="mergesort")
        x = df["num_samples"].to_numpy()
        y = df[f"{y_key}_mean"].to_numpy()
        s = df[f"{y_key}_std"].to_numpy() if f"{y_key}_std" in df.columns else None
        plt.plot(x, y, label=regime)
        if s is not None and not np.all(np.isnan(s)):
            plt.fill_between(x, y - s, y + s, alpha=0.15)
    plt.xlabel("num_samples")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(title="Regime")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] saved plot: {out_png}")

def _compute_aulc(df: pd.DataFrame, x_key="num_samples", y_key=METRIC_KEY):
    """Trapezoid AULC per run_file; returns per-run and aggregated mean±std."""
    if df.empty:
        return pd.DataFrame(), math.nan, math.nan
    rows = []
    for rf, g in df.groupby("run_file"):
        g = g.sort_values(x_key)
        x = g[x_key].to_numpy()
        y = g[y_key].to_numpy().astype(float)
        if y.max() <= 1.0 + 1e-6:
            y = y * 100.0
        a = float(np.trapz(y, x))
        rows.append({"run_file": rf, "AULC": a, "x_min": x.min(), "x_max": x.max()})
    per_run = pd.DataFrame(rows)
    return per_run, float(per_run["AULC"].mean()), float(per_run["AULC"].std(ddof=1) if len(per_run)>1 else np.nan)

def main():
    # 1) Load per-regime seeds
    regime_runs = {}   # regime -> list[DataFrame]
    regime_aggs = {}   # regime -> aggregated mean/std df
    for regime, pat in REGIMES.items():
        print(f"== Regime: {regime} ({pat}) ==")
        dfs = _load_csvs_for_pattern(pat)
        regime_runs[regime] = dfs
        if dfs:
            # aggregate standard accuracy metric
            cols = [METRIC_KEY] + [c for c in CCMA_COLS if c in dfs[0].columns]
            agg = _mean_std_agg(dfs, cols)
            regime_aggs[regime] = agg
            # write aggregates
            agg.to_csv(os.path.join(OUT_DIR, f"{regime}_aggregate.csv"), index=False)
        else:
            regime_aggs[regime] = pd.DataFrame()

    # 2) Plot test accuracy curves (mean±std)
    _plot_curve(
        regime_aggs,
        y_key=METRIC_KEY,
        y_label=METRIC_LABEL,
        title=f"{METRIC_LABEL} vs num_samples (Teacher regimes)",
        out_png=os.path.join(OUT_DIR, "accuracy_curves_regimes.pgf"),
    )

    # 3) AULC per regime
    aulc_rows = []
    for regime, dfs in regime_runs.items():
        if not dfs:
            continue
        all_df = pd.concat(dfs, ignore_index=True)
        per_run, mu, sd = _compute_aulc(all_df, x_key="num_samples", y_key=METRIC_KEY)
        per_run["regime"] = regime
        per_run.to_csv(os.path.join(OUT_DIR, f"{regime}_AULC_per_run.csv"), index=False)
        aulc_rows.append({"regime": regime, "AULC_mean": mu, "AULC_std": sd, "n": len(per_run)})
    if aulc_rows:
        aulc_df = pd.DataFrame(aulc_rows).sort_values("AULC_mean", ascending=False)
        aulc_df.to_csv(os.path.join(OUT_DIR, "AULC_summary.csv"), index=False)
        # quick bar
        plt.figure(figsize=(7.5, 4.8))
        x = np.arange(len(aulc_df))
        plt.bar(x, aulc_df["AULC_mean"].to_numpy(),
                yerr=aulc_df["AULC_std"].to_numpy(), capsize=3)
        plt.xticks(x, aulc_df["regime"].tolist(), rotation=20, ha="right")
        plt.ylabel("AULC (area under accuracy curve)")
        plt.title("AULC by teacher regime")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "AULC_bars_regimes.png"), dpi=200)
        plt.savefig(os.path.join(OUT_DIR, "AULC_bars_regimes.pgf"), dpi=200)
        plt.close()
        print(f"[OK] saved plot: {os.path.join(OUT_DIR, 'AULC_bars_regimes.png')}")

    # 4) CCMA diagnostics curves (if available)
    ccma_found_any = False
    for cc, label in [
        ("CCMA_GI_mean",      "|Γ_I| (student set size)"),
        ("CCMA_GT_mean",      "|Γ_T| (teacher set size)"),
        ("CCMA_overlap_mean", "overlap(Γ_I, Γ_T)"),
        ("CCMA_symdiff_mean", "symdiff(Γ_I, Γ_T)"),
        ("CCMA_identical_pct","% identical sets"),
    ]:
        # only plot if at least one regime has this column
        if any(cc in (df.columns if df is not None else []) for df in regime_aggs.values()):
            _plot_curve(
                regime_aggs,
                y_key=cc,
                y_label=label,
                title=f"CCMA diagnostic • {label}",
                out_png=os.path.join(OUT_DIR, f"ccma_{cc}.png"),
            )
            ccma_found_any = True

    if not ccma_found_any:
        print("[INFO] No CCMA columns found in CSVs or logs; diagnostics plots skipped."
              " (You can add those five CCMA fields to your CSV per iteration, or ensure .log exists.)")

if __name__ == "__main__":
    main()
