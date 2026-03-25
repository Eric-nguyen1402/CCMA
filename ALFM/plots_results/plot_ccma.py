# save as: tools/plot_teacher_regimes_from_csv.py

import os, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "ALFM/logs/results/domainnetreal/dino_vit_g14"
OUT_DIR  = os.path.join(BASE_DIR, "teacher_regimes")
os.makedirs(OUT_DIR, exist_ok=True)

REGIMES = {
    "teacher_high":   "disagreement_high-*.csv",
    "teacher_weak":   "disagreement_weak-*.csv",
    "student_only":   "disagreement_student-*.csv",
}

METRIC_KEY = "TEST_MulticlassAccuracy"
CCMA_KEYS = [
    ("CCMA_GI_mean",      "|Γ_I|"),
    ("CCMA_GT_mean",      "|Γ_T|"),
    ("CCMA_overlap_mean", "overlap"),
    ("CCMA_symdiff_mean", "symdiff"),
    ("CCMA_identical_pct","% identical"),
]

def _load_runs(pattern):
    paths = sorted(glob.glob(os.path.join(BASE_DIR, pattern)))
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            req = {"iteration","num_samples",METRIC_KEY}
            if not req.issubset(df.columns): continue
            dfs.append(df.sort_values("num_samples", kind="mergesort"))
        except Exception as e:
            print(f"[WARN] read fail {p}: {e}")
    if not dfs:
        print(f"[WARN] no files for {pattern}")
    return dfs

def _agg_mean_std(dfs, cols):
    all_df = pd.concat(dfs, ignore_index=True)
    g = all_df.groupby(["iteration","num_samples"], as_index=False)[cols].agg(["mean","std"])
    g.columns = ["_".join(c).strip("_") for c in g.columns.to_flat_index()]
    return g.reset_index()

def _plot_curves(regime_aggs, y_key, y_label, fname):
    plt.figure(figsize=(9,6))
    for name, df in regime_aggs.items():
        if df.empty: continue
        df = df.sort_values("num_samples", kind="mergesort")
        x = df["num_samples"].to_numpy()
        y = df[f"{y_key}_mean"].to_numpy()
        s = df.get(f"{y_key}_std", pd.Series(np.nan, index=df.index)).to_numpy()
        plt.plot(x, y, label=name)
        if not np.all(np.isnan(s)):
            plt.fill_between(x, y - s, y + s, alpha=0.15)
    plt.xlabel("num_samples")
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs num_samples (teacher regimes)")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(title="Regime")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=200); plt.close()
    print(f"[OK] {out}")

def _aulc(per_run_df, x="num_samples", y=METRIC_KEY):
    rows = []
    for rf, g in per_run_df.groupby("run_file", dropna=False):
        g = g.sort_values(x)
        xv = g[x].to_numpy()
        yv = g[y].to_numpy().astype(float)
        if yv.max() <= 1 + 1e-6: yv = yv * 100.0
        rows.append({"run_file": rf, "AULC": float(np.trapz(yv, xv))})
    out = pd.DataFrame(rows)
    mu = float(out["AULC"].mean()) if not out.empty else math.nan
    sd = float(out["AULC"].std(ddof=1)) if len(out) > 1 else math.nan
    return out, mu, sd

def main():
    regime_runs = {name: _load_runs(pat) for name, pat in REGIMES.items()}
    # aggregate
    regime_aggs = {}
    for name, dfs in regime_runs.items():
        if not dfs:
            regime_aggs[name] = pd.DataFrame()
            continue
        cols = [METRIC_KEY] + [k for k,_ in CCMA_KEYS if k in dfs[0].columns]
        agg = _agg_mean_std(dfs, cols)
        agg.to_csv(os.path.join(OUT_DIR, f"{name}_aggregate.csv"), index=False)
        regime_aggs[name] = agg

    # accuracy curves
    _plot_curves(regime_aggs, METRIC_KEY, "Accuracy", "regime_accuracy.png")

    # AULC bars
    bars = []
    for name, dfs in regime_runs.items():
        if not dfs: continue
        all_df = pd.concat(dfs, ignore_index=True)
        pr, mu, sd = _aulc(all_df, y=METRIC_KEY)
        pr["regime"] = name
        pr.to_csv(os.path.join(OUT_DIR, f"{name}_AULC_per_run.csv"), index=False)
        bars.append({"regime": name, "AULC_mean": mu, "AULC_std": sd, "n": len(pr)})
    if bars:
        bdf = pd.DataFrame(bars).sort_values("AULC_mean", ascending=False)
        bdf.to_csv(os.path.join(OUT_DIR, "AULC_summary.csv"), index=False)
        plt.figure(figsize=(7.5,4.4))
        x = np.arange(len(bdf))
        plt.bar(x, bdf["AULC_mean"].to_numpy(),
                yerr=bdf["AULC_std"].to_numpy(), capsize=3)
        plt.xticks(x, bdf["regime"].tolist(), rotation=15, ha="right")
        plt.ylabel("AULC (area under accuracy curve)")
        plt.title("AULC by teacher regime")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, "regime_AULC_bars.png")
        plt.savefig(out, dpi=200); plt.close()
        print(f"[OK] {out}")

    # CCMA diagnostics
    for key, label in CCMA_KEYS:
        # plot only if any regime has it
        if not any((not df.empty and f"{key}_mean" in df.columns) for df in regime_aggs.values()):
            continue
        _plot_curves(regime_aggs, key, label, f"ccma_{key}.png")

if __name__ == "__main__":
    main()
