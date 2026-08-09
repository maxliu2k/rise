"""Paired CNN-vs-CRNN comparison at one SNR, as a committed artifact.

    python -m instrument_robustness.paired_model_comparison

Writes `artifacts/failure_analysis/paired_model_comparison.csv` and
`paired_model_comparison.json`: one row per (noise_type, replicate), each carrying the paired
macro-F1 difference, its cluster-bootstrap interval, the exact cluster sign test, and the
Benjamini-Hochberg q-value over the declared family of six.

WHY THIS EXISTS. `noise_stats.py` has implemented `cluster_bootstrap` and `cluster_sign_test`
since the noise sweep landed, and `tests/test_noise.py` covers both -- but nothing ever called
them from a script, so the paper's paired-comparison numbers lived only as prose in
`docs/POSTER_REVIEW.md`. Every other number in the paper regenerates from a committed artifact;
that subsection did not. A tested function nobody runs produces no results.

DIRECTION. Differences are B minus A with A=CRNN and B=CNN, so a positive value favours CNN.
This matches the paper. Changing `--model-a`/`--model-b` changes the sign; the direction is
recorded in the manifest rather than left to the reader.

WHAT IT REFUSES TO DO. It will not report over whatever conditions happen to be on disk: the
family is declared up front and every member must be present, because a BH correction over four
comparisons that silently claims to be over six is a wrong q-value, not a missing one.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from instrument_robustness.config import (
    ARTIFACTS,
    NOISE_TYPES,
    N_REPLICATES,
    assert_fingerprint,
)
from instrument_robustness.noise_stats import (
    cluster_bootstrap,
    cluster_sign_test,
    load_condition,
    macro_f1,
    paired_frames,
)
from instrument_robustness.robustness_curve import benjamini_hochberg

OUTPUT_DIR = ARTIFACTS / "failure_analysis"
MODEL_PREFIXES = {
    "cnn": "cnn_test_",
    "crnn": "crnn_test_",
    "svm": "svm_test_",
    "ast": "ast_test_",
    "panns": "panns_test_",
    "mert_ft": "mert_ft_test_",
}
DEFAULT_SNR_DB = 20


@dataclass(frozen=True)
class Condition:
    """One member of the declared comparison family."""

    noise_type: str
    snr_db: int
    replicate: int

    @property
    def tag(self) -> str:
        return f"{self.noise_type}_{self.snr_db}_r{self.replicate}"


def declared_family(snr_db: int) -> list[Condition]:
    """Every (noise_type, replicate) pair at one SNR, in a fixed order.

    Postcondition: length is `len(NOISE_TYPES) * N_REPLICATES`. The order is deterministic so the
    output CSV and the BH ranking are reproducible run to run.
    """
    return [
        Condition(noise_type, snr_db, replicate)
        for noise_type in NOISE_TYPES
        for replicate in range(N_REPLICATES)
    ]


def _condition_dataset_identity(model: str, condition: Condition) -> str:
    """Assert this condition's metrics match the current config; return its dataset identity.

    The prediction CSVs carry no fingerprint, so the sibling metrics JSON is what stands between
    this analysis and a stale prediction file from an older cache.

    Returns the canonical JSON of the metrics file's `dataset` block. Read from the artifact
    rather than re-derived from the local tree on purpose: what matters is the corpus the
    predictions were SCORED against, not whatever happens to sit on this machine. The audio root
    is not committed, so re-deriving it would also make this analysis un-runnable off the cluster.

    Raises: FileNotFoundError if the metrics file is absent; StaleArtifactError if it disagrees
    with the current config; ValueError if it records no dataset identity.
    """
    path = ARTIFACTS / model / "noise" / f"metrics_{condition.tag}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics for {model} {condition.tag}: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert_fingerprint(metrics.get("config_fingerprint"), str(path))
    dataset = metrics.get("dataset")
    if not dataset:
        raise ValueError(f"{path} records no dataset identity, so the corpus cannot be checked")
    return _canonical_json(dataset)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _assert_same_noise(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    condition: Condition,
) -> None:
    """Crash unless both models saw the same noise realization on every window.

    `paired_frames` already proves the two frames cover the same windows in the same order. This
    proves the mixtures were the same ones, which is the claim the paired interval rests on. If
    the column is absent from either side the check cannot be made and says so, rather than
    passing silently.
    """
    if "noise_source" not in frame_a.columns or "noise_source" not in frame_b.columns:
        raise ValueError(
            f"{condition.tag}: a prediction file lacks noise_source, so pairing cannot be "
            f"verified. Re-score with the current noise_eval_common."
        )
    mismatched = int((frame_a["noise_source"] != frame_b["noise_source"]).sum())
    if mismatched:
        raise ValueError(
            f"{condition.tag}: {mismatched} of {len(frame_a)} windows were scored against "
            f"different noise sources by the two models. These predictions are not paired."
        )


def compare_condition(
    model_a: str,
    model_b: str,
    condition: Condition,
    *,
    cluster: str,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap interval and sign test for one condition.

    Preconditions: both models have a prediction CSV and a metrics JSON for `condition`.
    Postcondition: the returned mapping reports `delta_macro_f1` as B minus A.
    Raises: FileNotFoundError, ValueError, StaleArtifactError -- see the helpers above.
    """
    identities = {
        model: _condition_dataset_identity(model, condition)
        for model in (model_a, model_b)
    }

    frame_a = load_condition(
        condition.tag, ARTIFACTS / model_a / "noise", MODEL_PREFIXES[model_a]
    )
    frame_b = load_condition(
        condition.tag, ARTIFACTS / model_b / "noise", MODEL_PREFIXES[model_b]
    )
    aligned_a, aligned_b = paired_frames(frame_a, frame_b)
    _assert_same_noise(aligned_a, aligned_b, condition)

    boot = cluster_bootstrap(
        aligned_a, aligned_b, cluster=cluster, n_boot=n_boot, seed=seed
    )
    sign = cluster_sign_test(aligned_a, aligned_b, cluster=cluster)
    low, high = boot["ci95"]
    return {
        "noise_type": condition.noise_type,
        "snr_db": condition.snr_db,
        "replicate": condition.replicate,
        f"macro_f1_{model_a}": macro_f1(aligned_a),
        f"macro_f1_{model_b}": macro_f1(aligned_b),
        "delta_macro_f1": boot["delta_macro_f1"],
        "ci95_low": low,
        "ci95_high": high,
        "ci95_excludes_zero": bool(low > 0.0 or high < 0.0),
        "sign_test_p": sign["p_value"],
        "sign_clusters_favouring_b": sign["b_better"],
        "sign_clusters_favouring_a": sign["a_better"],
        "sign_clusters_tied": sign["ties"],
        "n_clusters": boot["n_clusters"],
        "n_windows": boot["n_windows"],
        "_dataset_identity": identities[model_a],
        "_dataset_identity_b": identities[model_b],
    }


def run(
    model_a: str,
    model_b: str,
    *,
    snr_db: int,
    cluster: str,
    n_boot: int,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare two models over the declared family and write both artifacts.

    Postcondition: `output_dir` contains `paired_model_comparison.csv` with exactly
    `len(NOISE_TYPES) * N_REPLICATES` rows, and a manifest naming the direction, the family, and
    the BH correction applied to the sign tests.
    """
    family = declared_family(snr_db)
    rows = [
        compare_condition(
            model_a, model_b, condition, cluster=cluster, n_boot=n_boot, seed=seed
        )
        for condition in family
    ]
    if len(rows) != len(family):
        raise RuntimeError("condition count changed mid-run")

    # Every condition, both models, one corpus. REPOSITORY_AUDIT.md records a real case of models
    # being compared across different corpora, and a paired interval computed across two corpora
    # is not an interval at all -- it is a comparison of two different experiments.
    identities = {row.pop("_dataset_identity") for row in rows} | {
        row.pop("_dataset_identity_b") for row in rows
    }
    if len(identities) != 1:
        raise ValueError(
            f"{model_a} and {model_b} were scored against {len(identities)} distinct dataset "
            f"identities across the family. These predictions cannot be paired."
        )
    dataset_identity = json.loads(next(iter(identities)))

    correction = benjamini_hochberg(
        {condition.tag: row["sign_test_p"] for condition, row in zip(family, rows)},
        family=(
            f"{model_b} vs {model_a} exact cluster sign tests at {snr_db} dB, "
            f"{len(NOISE_TYPES)} noise categories x {N_REPLICATES} replicates"
        ),
    )
    q_values = {result["label"]: result for result in correction["results"]}
    for condition, row in zip(family, rows):
        result = q_values[condition.tag]
        row["sign_test_q"] = result["q_value"]
        row["sign_test_rejected"] = result["rejected"]

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "paired_model_comparison.csv", index=False)
    manifest = {
        "state": "complete",
        "comparison": f"{model_b} minus {model_a}",
        "sign_convention": f"positive favours {model_b}",
        "snr_db": snr_db,
        "cluster_unit": cluster,
        "n_boot": n_boot,
        "seed": seed,
        "declared_family": [condition.tag for condition in family],
        "n_intervals_excluding_zero": int(frame["ci95_excludes_zero"].sum()),
        "n_sign_tests_rejected_after_bh": int(correction["n_rejected"]),
        "benjamini_hochberg": correction,
        "dataset": dataset_identity,
    }
    (output_dir / "paired_model_comparison.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default="crnn", choices=sorted(MODEL_PREFIXES))
    parser.add_argument("--model-b", default="cnn", choices=sorted(MODEL_PREFIXES))
    parser.add_argument("--snr-db", type=int, default=DEFAULT_SNR_DB)
    parser.add_argument("--cluster", default="pitch_group")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.model_a == args.model_b:
        parser.error("model-a and model-b must differ")

    frame = run(
        args.model_a,
        args.model_b,
        snr_db=args.snr_db,
        cluster=args.cluster,
        n_boot=args.n_boot,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"{args.model_b} minus {args.model_a} at {args.snr_db} dB "
          f"(positive favours {args.model_b}):\n")
    columns = [
        "noise_type", "replicate", "delta_macro_f1", "ci95_low", "ci95_high",
        "ci95_excludes_zero", "sign_test_p", "sign_test_q", "sign_test_rejected",
    ]
    print(frame[columns].to_string(index=False))
    print(f"\nintervals excluding zero      : {int(frame['ci95_excludes_zero'].sum())}"
          f" of {len(frame)}")
    print(f"sign tests rejected after BH  : {int(frame['sign_test_rejected'].sum())}"
          f" of {len(frame)}")
    print(f"\nwrote {args.output_dir}/paired_model_comparison.{{csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
