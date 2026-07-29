"""noise_stats.py — paired CLUSTER bootstrap and McNemar over saved noise predictions.

Reads the per-clip CSVs written by the noise evaluators. No model is loaded and nothing is
re-run, so the whole comparison is cheap and can be redone whenever the analysis changes.

WHY CLUSTERED. Windows are not independent observations. Several windows come from one recording,
and every recording of the same (instrument, note) belongs to one pitch group -- the unit step3
splits on. Resampling individual windows would treat near-duplicates as fresh evidence and report
confidence intervals that are too narrow. The bootstrap here resamples whole CLUSTERS with
replacement, which keeps that correlation intact. Window-level point estimates are still reported;
only the uncertainty around them changes.

PAIRED. Both the bootstrap and McNemar compare two conditions (or two models) on the SAME windows,
so the shared noise realization cancels out and what remains is the difference of interest. This
is why noise_sweep.py generates one shared noisy set rather than letting each model make its own.

Usage:
    python -m instrument_robustness.noise_stats --a clean --b white_20
    python -m instrument_robustness.noise_stats --a clean --b natural_0 --cluster source_path
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from instrument_robustness.config import DATA_ROOT

DEFAULT_DIR = DATA_ROOT / "artifacts" / "panns" / "noise"


def load_condition(condition, directory=None, prefix="panns_ft_test_"):
    d = Path(directory) if directory else DEFAULT_DIR
    p = d / f"{prefix}{condition}.csv"
    if not p.exists():
        raise SystemExit(f"no predictions for condition {condition!r} at {p}")
    return pd.read_csv(p)


def macro_f1(df):
    return f1_score(df["true_label"], df["predicted_label"], average="macro", zero_division=0)


def cluster_bootstrap(df_a, df_b, cluster="pitch_group", n_boot=2000, seed=0):
    """Paired cluster bootstrap of (metric_b - metric_a). Returns point estimate and 95% CI.

    Resamples clusters, not rows. Both conditions are indexed by the same resampled clusters, so
    each replicate compares the two conditions on identical windows.
    """
    if cluster not in df_a.columns:
        raise SystemExit(f"cluster column {cluster!r} not in the predictions CSV; "
                         f"available: {[c for c in df_a.columns if c in ('source_path','pitch_group')]}")
    a = df_a.sort_values("window_id").reset_index(drop=True)
    b = df_b.sort_values("window_id").reset_index(drop=True)
    if not a["window_id"].equals(b["window_id"]):
        raise SystemExit("conditions are not paired: window_id sets differ")

    groups = a[cluster].to_numpy()
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    point = macro_f1(b) - macro_f1(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in picked])
        diffs[i] = macro_f1(b.iloc[rows]) - macro_f1(a.iloc[rows])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_macro_f1": float(point), "ci95": [float(lo), float(hi)],
            "n_clusters": int(len(uniq)), "n_windows": int(len(a)), "cluster": cluster,
            "n_boot": n_boot}


def mcnemar(df_a, df_b, cluster=None):
    """Paired accuracy test. Exact binomial on the discordant pairs (no chi-square approximation).

    If `cluster` is given, the discordant counts are aggregated per cluster first, because the
    plain window-level test also assumes independent observations.
    """
    from scipy import stats
    a = df_a.sort_values("window_id").reset_index(drop=True)
    b = df_b.sort_values("window_id").reset_index(drop=True)
    ca, cb = a["correct"].to_numpy(bool), b["correct"].to_numpy(bool)
    if cluster:
        g = a[cluster].to_numpy()
        n01 = sum(((~ca) & cb)[g == u].sum() > ((ca & ~cb)[g == u]).sum() for u in np.unique(g))
        n10 = sum(((ca) & ~cb)[g == u].sum() > ((~ca & cb)[g == u]).sum() for u in np.unique(g))
    else:
        n01 = int(((~ca) & cb).sum())     # a wrong, b right
        n10 = int((ca & (~cb)).sum())     # a right, b wrong
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"b_better": int(n01), "a_better": int(n10), "discordant": int(n),
            "p_value": p, "unit": cluster or "window"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline condition, e.g. clean")
    ap.add_argument("--b", required=True, help="comparison condition, e.g. white_20")
    ap.add_argument("--cluster", default="pitch_group", choices=["pitch_group", "source_path"])
    ap.add_argument("--dir", default=None, help="directory of predictions CSVs")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    A, B = load_condition(args.a, args.dir), load_condition(args.b, args.dir)
    boot = cluster_bootstrap(A, B, cluster=args.cluster, n_boot=args.n_boot)
    mc = mcnemar(A, B, cluster=args.cluster)
    print(f"{args.a}  macro-F1 {macro_f1(A):.4f}")
    print(f"{args.b}  macro-F1 {macro_f1(B):.4f}")
    print(f"\npaired cluster bootstrap over {boot['n_clusters']} {args.cluster} clusters "
          f"({boot['n_windows']} windows, {args.n_boot} replicates):")
    print(f"  delta macro-F1 = {boot['delta_macro_f1']:+.4f}  "
          f"95% CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}]")
    print(f"\nMcNemar (unit = {mc['unit']}): discordant {mc['discordant']}, "
          f"p = {mc['p_value']:.3g}")
    print(json.dumps({"bootstrap": boot, "mcnemar": mc}, indent=2))


if __name__ == "__main__":
    main()
