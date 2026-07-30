"""Summarise a degradation curve, and control the false-discovery rate across many comparisons.

Three reporting hazards that only appear once a sweep has actually been run.

AUDIT ITEM 3 -- REPLICATES. Multiple draws are useful only if the reporting keeps their identity.
Curves therefore average macro-F1 at each SNR across replicates, report the spread, and model
comparisons pair the same replicate number before taking a difference. Treating every draw as an
unrelated curve point creates duplicate SNRs; comparing unmatched draws confounds model differences
with noise-sampling luck.

AUDIT ITEM 18 -- SPACING. `config.SNRS` uses 10 dB steps from 60 through 0 and 5 dB steps from 0
through -15. An unweighted mean would therefore give the densely sampled lower tail too much weight.
The dB-weighted integral handles the actual spacing:

  * adding two levels where a model happens to do well moves the unweighted mean a lot and the
    weighted integral almost not at all. Measured, same SVM curve, adding 55 and 45 dB:
    mean 0.4059 -> 0.4986 (moves 0.093), AUC 0.3902 -> 0.3898 (moves 0.0004).
  * `--snrs` and `snr_range` can produce other non-uniform selections.
  * any future grid retune can change density again.

So: report `robustness_auc`, which is invariant to how densely the curve was sampled.
`mean_retention` is returned alongside it only so the two can be compared and the gap seen.

AUDIT ITEM 19 -- MULTIPLICITY. A full sweep is 90 noisy condition-replicates per model. Comparing
several models, and optionally 12 instruments within each, produces hundreds of hypothesis tests.
`noise_stats` returns a correct p-value for ONE comparison; nothing corrected across the family, so
at alpha = 0.05 roughly one comparison in twenty looks significant by construction.
`benjamini_hochberg` controls the expected proportion of false positives among the rejections, and
requires the family to be named explicitly rather than inferred.

Torch-free; numpy and scipy only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurvePoint:
    """One measured point on a degradation curve."""

    snr_db: float
    macro_f1: float


def _sorted_points(points: list[CurvePoint]) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("a curve needs at least two points")
    snrs = np.asarray([point.snr_db for point in points], dtype=np.float64)
    scores = np.asarray([point.macro_f1 for point in points], dtype=np.float64)
    if len(np.unique(snrs)) != len(snrs):
        raise ValueError(f"duplicate SNR values in the curve: {snrs.tolist()}")
    order = np.argsort(snrs)
    return snrs[order], scores[order]


def mean_retention(points: list[CurvePoint], clean_macro_f1: float) -> float:
    """Unweighted mean retention across the measured points.

    Provided for comparison ONLY. This is not a curve summary: it is the average of however many
    points happen to have been measured, so adding levels in the region a model handles well raises
    it without the model having changed (measured: +0.093 from adding two easy levels, against
    +0.0004 for the integral). Prefer `robustness_auc`.
    """
    if clean_macro_f1 <= 0:
        raise ValueError("clean_macro_f1 must be positive")
    _, scores = _sorted_points(points)
    return float(np.mean(scores / clean_macro_f1))


def robustness_auc(
    points: list[CurvePoint],
    clean_macro_f1: float,
    *,
    snr_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Area under the retention-vs-SNR curve, integrated over dB and normalised by dB span.

    Preconditions: at least two distinct SNRs; `clean_macro_f1` positive.
    Postcondition: `{"auc", "snr_low_db", "snr_high_db", "span_db", "mean_retention", "n_points"}`.
    `auc` is dimensionless and lies in [0, 1] for retentions in [0, 1]: 1.0 means no degradation
    anywhere in the span, 0.0 means total collapse everywhere.

    Trapezoidal in dB:

        auc = (1 / (s_max - s_min)) * integral of retention(s) ds

    Normalising by the span is what makes two models comparable when they were measured over the
    same range, and makes the number independent of how many intermediate points were sampled.

    `snr_range` restricts the integration, which is the honest way to compare against a study that
    used a different grid -- integrate both over the overlapping span rather than comparing areas
    computed over different ranges.
    """
    if clean_macro_f1 <= 0:
        raise ValueError("clean_macro_f1 must be positive")
    snrs, scores = _sorted_points(points)
    retention = scores / clean_macro_f1
    if snr_range is not None:
        low, high = sorted(snr_range)
        inside = (snrs >= low) & (snrs <= high)
        if inside.sum() < 2:
            raise ValueError(
                f"snr_range {snr_range} contains fewer than two measured points"
            )
        snrs, retention = snrs[inside], retention[inside]
    span = float(snrs[-1] - snrs[0])
    if span <= 0:
        raise ValueError("SNR span must be positive")
    area = float(np.trapezoid(retention, snrs)) if hasattr(np, "trapezoid") else float(
        np.trapz(retention, snrs)
    )
    return {
        "auc": area / span,
        "snr_low_db": float(snrs[0]),
        "snr_high_db": float(snrs[-1]),
        "span_db": span,
        "mean_retention": float(np.mean(retention)),
        "n_points": int(len(snrs)),
    }


def snr_at_retention(
    points: list[CurvePoint],
    clean_macro_f1: float,
    *,
    target: float = 0.5,
) -> float | None:
    """The SNR at which retention crosses `target`, linearly interpolated in dB.

    Postcondition: a float in the measured range, or None when the curve never crosses the target.

    Often a more legible headline than an area: "this model holds half its clean macro-F1 down to
    42 dB" is directly comparable across models and needs no explanation of weighting.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target retention must lie strictly between 0 and 1")
    if clean_macro_f1 <= 0:
        raise ValueError("clean_macro_f1 must be positive")
    snrs, scores = _sorted_points(points)
    retention = scores / clean_macro_f1
    # Walk from the highest SNR downwards: find the first pair bracketing the target.
    for index in range(len(snrs) - 1, 0, -1):
        high_r, low_r = retention[index], retention[index - 1]
        if (high_r >= target) and (low_r < target):
            high_s, low_s = snrs[index], snrs[index - 1]
            if high_r == low_r:
                return float(low_s)
            weight = (target - low_r) / (high_r - low_r)
            return float(low_s + weight * (high_s - low_s))
    return None


def benjamini_hochberg(
    p_values: dict[str, float],
    *,
    alpha: float = 0.05,
    family: str,
) -> dict[str, object]:
    """Benjamini-Hochberg FDR control over a named family of comparisons.

    Preconditions: `p_values` maps a comparison label to a p-value in [0, 1]; `family` names the set
    being corrected and is required, not optional.
    Postcondition: `{"family", "alpha", "n_comparisons", "n_rejected", "results": [...]}` where each
    result carries `label`, `p_value`, `rank`, `critical_value`, `q_value` and `rejected`.

    WHY `family` IS MANDATORY. FDR is meaningless without stating what was corrected over. Correcting
    21 conditions for one model is a different claim from correcting 21 conditions x 5 models, and a
    reader cannot check the arithmetic without knowing which was done. Forcing the caller to name it
    makes the choice appear in the output.

    BH controls the expected PROPORTION OF FALSE POSITIVES AMONG REJECTIONS, not the probability of
    any false positive -- that would be Bonferroni, which is far more conservative and inappropriate
    here, where the comparisons are positively correlated (the same test windows at neighbouring
    SNRs). Reported q-values are the standard monotone (step-up) adjusted values.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not family:
        raise ValueError("family must be a non-empty description of what is being corrected")
    if not p_values:
        raise ValueError("no p-values supplied")
    for label, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"p-value for {label!r} is outside [0, 1]: {value}")

    labels = list(p_values)
    raw = np.asarray([float(p_values[label]) for label in labels], dtype=np.float64)
    count = len(raw)
    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    ranks = np.arange(1, count + 1, dtype=np.float64)

    critical = alpha * ranks / count
    below = np.nonzero(ranked <= critical)[0]
    cutoff_rank = int(below.max()) + 1 if below.size else 0

    # Step-up adjusted values, enforced monotone from the largest p downwards.
    adjusted = ranked * count / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    results = []
    for position, index in enumerate(order):
        rank = position + 1
        results.append(
            {
                "label": labels[index],
                "p_value": float(raw[index]),
                "rank": rank,
                "critical_value": float(critical[position]),
                "q_value": float(adjusted[position]),
                "rejected": bool(rank <= cutoff_rank),
            }
        )
    return {
        "family": family,
        "method": "benjamini_hochberg_fdr",
        "alpha": float(alpha),
        "n_comparisons": count,
        "n_rejected": cutoff_rank,
        "results": results,
    }


def _replicate_number(row: dict[str, object]) -> int:
    """Return a non-negative integer replicate, treating an absent legacy value as replicate 0."""
    value = row.get("replicate", 0)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        replicate = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid replicate value: {value!r}") from error
    if replicate < 0 or float(value) != replicate:
        raise ValueError(f"invalid replicate value: {value!r}")
    return replicate


def _noisy_metric_map(
    rows: list[dict[str, object]],
    *,
    metric: str,
) -> dict[tuple[str, float, int], float]:
    """Index noisy rows by (noise type, SNR, replicate), refusing duplicate conditions."""
    indexed: dict[tuple[str, float, int], float] = {}
    for row in rows:
        noise_type = str(row.get("noise_type", ""))
        snr = row.get("snr_db")
        if noise_type == "clean" or snr is None or (
            isinstance(snr, float) and np.isnan(snr)
        ):
            continue
        if metric not in row:
            raise ValueError(f"summary row is missing metric {metric!r}")
        key = (noise_type, float(snr), _replicate_number(row))
        if key in indexed:
            raise ValueError(
                "duplicate summary condition "
                f"noise_type={key[0]!r}, snr_db={key[1]}, replicate={key[2]}"
            )
        indexed[key] = float(row[metric])
    if not indexed:
        raise ValueError("summary contains no noisy conditions")
    return indexed


def replicate_spread(
    rows: list[dict[str, object]],
    *,
    metric: str = "macro_f1",
) -> list[dict[str, object]]:
    """Summarise one metric across noise draws at each (noise type, SNR).

    Every SNR for a noise type must contain the same replicate numbers. This makes an interrupted
    or partially evaluated sweep fail rather than silently averaging unequal numbers of draws.
    Standard deviation is the population spread across the enumerated draws and is 0 for one draw.
    """
    indexed = _noisy_metric_map(rows, metric=metric)
    replicate_sets: dict[tuple[str, float], set[int]] = {}
    for noise_type, snr, replicate in indexed:
        replicate_sets.setdefault((noise_type, snr), set()).add(replicate)
    for noise_type in {key[0] for key in replicate_sets}:
        sets = {
            tuple(sorted(replicates))
            for (kind, _), replicates in replicate_sets.items()
            if kind == noise_type
        }
        if len(sets) != 1:
            raise ValueError(
                f"inconsistent replicate sets across SNRs for {noise_type!r}: "
                f"{sorted(sets)}"
            )

    records: list[dict[str, object]] = []
    for (noise_type, snr), replicates in sorted(replicate_sets.items()):
        ordered = sorted(replicates)
        values = np.asarray(
            [indexed[(noise_type, snr, replicate)] for replicate in ordered],
            dtype=np.float64,
        )
        records.append(
            {
                "noise_type": noise_type,
                "snr_db": snr,
                "metric": metric,
                "n_replicates": len(ordered),
                "replicates": ordered,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return records


def paired_replicate_differences(
    reference_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    *,
    metric: str = "macro_f1",
) -> list[dict[str, object]]:
    """Summarise candidate-minus-reference differences after pairing identical noise draws.

    The two summaries must have exactly the same (noise type, SNR, replicate) keys. A positive mean
    therefore means the candidate scored higher on the same generated audio, not merely on a
    different random draw.
    """
    reference = _noisy_metric_map(reference_rows, metric=metric)
    candidate = _noisy_metric_map(candidate_rows, metric=metric)
    if reference.keys() != candidate.keys():
        missing_candidate = sorted(reference.keys() - candidate.keys())
        missing_reference = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            "model summaries do not contain the same replicate conditions: "
            f"missing from candidate={missing_candidate[:5]}, "
            f"missing from reference={missing_reference[:5]}"
        )

    grouped: dict[tuple[str, float], list[tuple[int, float]]] = {}
    for (noise_type, snr, replicate), reference_value in reference.items():
        difference = candidate[(noise_type, snr, replicate)] - reference_value
        grouped.setdefault((noise_type, snr), []).append((replicate, difference))

    records: list[dict[str, object]] = []
    for (noise_type, snr), pairs in sorted(grouped.items()):
        pairs.sort()
        values = np.asarray([difference for _, difference in pairs], dtype=np.float64)
        records.append(
            {
                "noise_type": noise_type,
                "snr_db": snr,
                "metric": metric,
                "direction": "candidate_minus_reference",
                "n_replicates": len(pairs),
                "replicates": [replicate for replicate, _ in pairs],
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return records


def summarise_sweep(summary_csv, *, alpha: float = 0.05) -> dict[str, object]:
    """Summarise one model's `noise_sweep_summary.csv`: a curve per noise type, plus the clean score.

    Postcondition: `{"clean_macro_f1", "curves": {noise_type: {...auc fields..., "snr_at_50pct",
    "snr_at_90pct"}}, "replicate_spread": [...]}`. Curves use the replicate mean at each SNR.
    Raises: ValueError if the clean row is missing -- retention is meaningless without it.
    """
    import pandas as pd

    frame = pd.read_csv(summary_csv)
    clean_rows = frame[frame["noise_type"] == "clean"]
    if clean_rows.empty:
        raise ValueError(f"{summary_csv} has no clean row to normalise against")
    clean = float(clean_rows["macro_f1"].iloc[0])
    rows = frame.to_dict("records")
    curves: dict[str, object] = {}
    for noise_type in sorted(set(frame["noise_type"]) - {"clean"}):
        points = curve_from_summary(rows, noise_type=noise_type)
        record = robustness_auc(points, clean)
        record["snr_at_50pct"] = snr_at_retention(points, clean, target=0.5)
        record["snr_at_90pct"] = snr_at_retention(points, clean, target=0.9)
        curves[noise_type] = record
    return {
        "clean_macro_f1": clean,
        "alpha": alpha,
        "curves": curves,
        "replicate_spread": replicate_spread(rows),
    }


def main() -> None:
    """CLI: summarise a completed sweep's curves, or FDR-correct a set of p-values."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Robustness-curve summaries (dB-weighted) and FDR correction."
    )
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="artifacts/<model>/noise/noise_sweep_summary.csv",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = summarise_sweep(args.summary_csv, alpha=args.alpha)
    print(f"clean macro-F1 {result['clean_macro_f1']:.6f}\n")
    print(f"{'noise':<12}{'AUC':>8}{'mean':>8}{'span':>8}{'50%@':>8}{'90%@':>8}")
    for noise_type, record in result["curves"].items():
        half = record["snr_at_50pct"]
        ninety = record["snr_at_90pct"]
        print(
            f"{noise_type:<12}{record['auc']:>8.4f}{record['mean_retention']:>8.4f}"
            f"{record['span_db']:>7.0f}dB"
            f"{'  n/a' if half is None else f'{half:>7.1f}'}"
            f"{'  n/a' if ninety is None else f'{ninety:>7.1f}'}"
        )
    print(
        "\nAUC is the dB-weighted integral of retention, normalised by span -- prefer it over "
        "`mean`,\nwhich changes with how densely the curve was sampled."
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


def curve_from_summary(
    rows: list[dict[str, object]],
    *,
    noise_type: str,
    replicate: int | None = None,
) -> list[CurvePoint]:
    """Extract one noise type's curve from `noise_sweep_summary.csv` rows.

    Preconditions: each row has `noise_type`, `snr_db` and `macro_f1`; the clean row (snr_db null) is
    ignored, since the clean score is the normaliser rather than a point on the curve.
    Postcondition: points sorted ascending by SNR. With `replicate=None`, rows at the same SNR are
    averaged across replicates; otherwise only the requested replicate is used.
    Raises: ValueError if the selection yields fewer than two points.
    """
    selected: dict[float, list[float]] = {}
    for row in rows:
        if row.get("noise_type") != noise_type:
            continue
        snr = row.get("snr_db")
        if snr is None or (isinstance(snr, float) and np.isnan(snr)):
            continue
        if replicate is not None and _replicate_number(row) != replicate:
            continue
        selected.setdefault(float(snr), []).append(float(row["macro_f1"]))
    points = [
        CurvePoint(snr, float(np.mean(scores)))
        for snr, scores in selected.items()
    ]
    if len(points) < 2:
        raise ValueError(
            f"fewer than two points for noise_type={noise_type!r}"
            + ("" if replicate is None else f", replicate={replicate}")
        )
    return sorted(points, key=lambda point: point.snr_db)


if __name__ == "__main__":
    main()
