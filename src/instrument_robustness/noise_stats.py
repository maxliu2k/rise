"""Paired cluster-aware comparisons over saved noise predictions.

The cluster bootstrap estimates the macro-F1 difference and its interval while keeping all twelve
labels fixed in every replicate. The cluster sign test is deliberately named as such; it is not
McNemar's test. An ordinary exact McNemar test is available only as an explicitly requested
window-level sensitivity analysis because windows within a recording/pitch group are correlated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from instrument_robustness.config import ARTIFACTS, TARGET_LABELS

DEFAULT_DIR = ARTIFACTS / "panns" / "noise"
PAIR_COLUMNS = (
    "window_id",
    "source_path",
    "pitch_group",
    "true_label",
)

# Clustering units, in increasing looseness. `pitch_group` remains the default and the conservative
# choice for comparing two CONDITIONS of one model. `noise_source` is the right unit when the
# correlation of concern runs through the NOISE rather than the instrument: several windows can be
# corrupted by crops of the same ESC-50 recording, so if that recording is unusually destructive they
# fail together. It is only available when the prediction CSV carries the column, which
# noise_eval_common attaches at scoring time.
CLUSTER_COLUMNS = ("pitch_group", "source_path", "noise_source")


def load_condition(
    condition: str,
    directory: str | Path | None = None,
    prefix: str = "panns_ft_test_",
) -> pd.DataFrame:
    root = DEFAULT_DIR if directory is None else Path(directory)
    path = root / f"{prefix}{condition}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No predictions for {condition!r} at {path}")
    frame = pd.read_csv(path)
    required = set(PAIR_COLUMNS) | {"predicted_label", "correct"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["window_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate window IDs")
    observed = set(frame["true_label"]) | set(frame["predicted_label"])
    unexpected = sorted(observed - set(TARGET_LABELS))
    if unexpected:
        raise ValueError(f"{path} contains unexpected labels: {unexpected}")
    return frame


def paired_frames(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = frame_a.sort_values("window_id").reset_index(drop=True)
    b = frame_b.sort_values("window_id").reset_index(drop=True)
    for column in PAIR_COLUMNS:
        if not a[column].equals(b[column]):
            raise ValueError(
                f"Predictions are not paired: column {column!r} differs"
            )
    return a, b


def macro_f1(frame: pd.DataFrame) -> float:
    return float(
        f1_score(
            frame["true_label"],
            frame["predicted_label"],
            labels=TARGET_LABELS,
            average="macro",
            zero_division=0,
        )
    )


def cluster_bootstrap(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    cluster: str = "pitch_group",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    if n_boot <= 0:
        raise ValueError("n_boot must be greater than zero")
    a, b = paired_frames(frame_a, frame_b)
    if cluster not in a.columns:
        raise ValueError(f"Missing cluster column {cluster!r}")
    if a[cluster].isna().any():
        raise ValueError(f"Cluster column {cluster!r} contains missing values")

    groups = a[cluster].to_numpy()
    unique_groups = np.unique(groups)
    indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    random = np.random.default_rng(seed)
    differences = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = random.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        rows = np.concatenate([indices[group] for group in sampled])
        differences[index] = macro_f1(b.iloc[rows]) - macro_f1(a.iloc[rows])
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "delta_macro_f1": macro_f1(b) - macro_f1(a),
        "ci95": [float(low), float(high)],
        "n_clusters": int(len(unique_groups)),
        "n_windows": int(len(a)),
        "cluster": cluster,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "fixed_label_order": TARGET_LABELS,
    }


def _correct_values(frame: pd.DataFrame) -> np.ndarray:
    values = frame["correct"]
    if values.dtype == bool:
        return values.to_numpy()
    mapped = values.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("correct column must contain booleans")
    return mapped.to_numpy(dtype=bool)


def exact_mcnemar(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
) -> dict[str, object]:
    """Ordinary exact McNemar over windows; use only as a sensitivity analysis."""
    a, b = paired_frames(frame_a, frame_b)
    correct_a = _correct_values(a)
    correct_b = _correct_values(b)
    b_better = int(((~correct_a) & correct_b).sum())
    a_better = int((correct_a & (~correct_b)).sum())
    discordant = b_better + a_better
    p_value = (
        1.0
        if discordant == 0
        else float(
            stats.binomtest(
                min(b_better, a_better),
                discordant,
                0.5,
            ).pvalue
        )
    )
    return {
        "test": "exact_mcnemar",
        "unit": "window",
        "b_better": b_better,
        "a_better": a_better,
        "discordant": discordant,
        "p_value": p_value,
        "warning": "windows are correlated; cluster-aware analysis is primary",
    }


def cluster_sign_test(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    cluster: str = "pitch_group",
) -> dict[str, object]:
    """Exact sign test on each cluster's net paired accuracy difference."""
    a, b = paired_frames(frame_a, frame_b)
    if cluster not in a.columns:
        raise ValueError(f"Missing cluster column {cluster!r}")
    correct_a = _correct_values(a)
    correct_b = _correct_values(b)
    groups = a[cluster].to_numpy()
    b_better = 0
    a_better = 0
    ties = 0
    for group in np.unique(groups):
        rows = groups == group
        difference = int(correct_b[rows].sum()) - int(correct_a[rows].sum())
        if difference > 0:
            b_better += 1
        elif difference < 0:
            a_better += 1
        else:
            ties += 1
    non_ties = b_better + a_better
    p_value = (
        1.0
        if non_ties == 0
        else float(
            stats.binomtest(
                min(b_better, a_better),
                non_ties,
                0.5,
            ).pvalue
        )
    )
    return {
        "test": "exact_cluster_sign_test",
        "unit": cluster,
        "b_better": b_better,
        "a_better": a_better,
        "ties": ties,
        "non_ties": non_ties,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--dir", default=None, help="shared directory for both inputs")
    parser.add_argument("--dir-a", default=None)
    parser.add_argument("--dir-b", default=None)
    parser.add_argument("--prefix-a", default="panns_ft_test_")
    parser.add_argument("--prefix-b", default="panns_ft_test_")
    parser.add_argument(
        "--cluster",
        default="pitch_group",
        choices=CLUSTER_COLUMNS,
        help=(
            "resampling unit. pitch_group (default) is conservative for instrument correlation; "
            "noise_source clusters by the ESC-50 recording drawn, for correlation running through "
            "the noise instead (audit item 8)"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--window-mcnemar",
        action="store_true",
        help="also report ordinary window-level McNemar as a sensitivity analysis",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    directory_a = args.dir_a or args.dir
    directory_b = args.dir_b or args.dir
    frame_a = load_condition(args.a, directory_a, args.prefix_a)
    frame_b = load_condition(args.b, directory_b, args.prefix_b)
    result: dict[str, object] = {
        "a": {
            "condition": args.a,
            "directory": str(DEFAULT_DIR if directory_a is None else directory_a),
            "prefix": args.prefix_a,
            "macro_f1": macro_f1(frame_a),
        },
        "b": {
            "condition": args.b,
            "directory": str(DEFAULT_DIR if directory_b is None else directory_b),
            "prefix": args.prefix_b,
            "macro_f1": macro_f1(frame_b),
        },
        "cluster_bootstrap": cluster_bootstrap(
            frame_a,
            frame_b,
            cluster=args.cluster,
            n_boot=args.n_boot,
            seed=args.seed,
        ),
        "cluster_sign_test": cluster_sign_test(
            frame_a,
            frame_b,
            cluster=args.cluster,
        ),
    }
    if args.window_mcnemar:
        result["window_mcnemar_sensitivity"] = exact_mcnemar(frame_a, frame_b)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
