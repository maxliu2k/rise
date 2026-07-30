"""Shared, fail-closed evaluation loop for clean-trained model noise sweeps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from instrument_robustness.config import (
    ARTIFACTS,
    MANIFEST_FINGERPRINT,
    MANIFEST_IN,
    MANIFEST_LABELED,
    N_REPLICATES,
    ROOT,
    TARGET_LABELS,
    WINDOWS_CSV,
    assert_artifact_fingerprint,
    assert_fingerprint,
    config_fingerprint,
)
from instrument_robustness.noise_sweep import (
    NOISE_MANIFEST_NAME,
    NOISE_PROVENANCE_NAME,
    NOISE_TYPES,
    NOISY_DIR,
    SNRS,
    out_path,
    sha256_file,
    test_windows,
    validate_noise_manifest,
    window_id_of,
)

CLEAN_PARITY_TOLERANCE = 1e-3
LABEL_TO_INDEX = {label: index for index, label in enumerate(TARGET_LABELS)}


@dataclass(frozen=True)
class NoiseCondition:
    tag: str
    noise_type: str
    snr_db: int | None
    replicate: int | None = None


def noise_conditions() -> list[NoiseCondition]:
    """Every condition a model is scored on: clean, plus each (noise type, SNR, replicate).

    Replicates are SEPARATE conditions rather than being averaged here. Aggregating them at scoring
    time would throw away exactly the quantity they exist to provide -- the spread of a model's score
    across independent noise draws, which is what tells you whether a gap between two models is
    larger than the noise in the measurement. `noise_stats` aggregates; this does not.

    The `_r{k}` suffix appears only when N_REPLICATES > 1, so a single-replicate build keeps the
    condition names that appear in figures and filenames short.
    """
    conditions = [NoiseCondition("clean", "clean", None, None)]
    for noise_type in NOISE_TYPES:
        for snr in SNRS:
            for replicate in range(N_REPLICATES):
                suffix = f"_r{replicate}" if N_REPLICATES > 1 else ""
                conditions.append(
                    NoiseCondition(
                        f"{noise_type}_{snr}{suffix}", noise_type, snr, replicate
                    )
                )
    return conditions


def load_test_frame(
    *,
    windows_csv: str | Path = WINDOWS_CSV,
    manifest_labeled: str | Path = MANIFEST_LABELED,
) -> pd.DataFrame:
    """Load test windows and attach the pitch-group cluster key without guessing."""
    frame = test_windows(windows_csv=windows_csv)
    manifest_labeled = Path(manifest_labeled)
    assert_artifact_fingerprint(manifest_labeled, "step0_filter")
    manifest = pd.read_csv(manifest_labeled)
    required = {"path", "label", "note"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"{manifest_labeled} is missing columns: {sorted(missing)}"
        )
    if manifest["path"].duplicated().any():
        raise ValueError(f"{manifest_labeled} contains duplicate source paths")
    notes = manifest[["path", "label", "note"]].rename(
        columns={"path": "source_path", "label": "manifest_label"}
    )
    frame = frame.merge(
        notes,
        on="source_path",
        how="left",
        validate="many_to_one",
    )
    if frame["note"].isna().any():
        count = int(frame["note"].isna().sum())
        raise ValueError(f"{count} test windows have no pitch-group note")
    if not frame["label"].equals(frame["manifest_label"]):
        raise ValueError("windows.csv labels disagree with manifest_labeled.csv")
    unexpected = sorted(set(frame["label"]) - set(TARGET_LABELS))
    if unexpected:
        raise ValueError(f"Unexpected test labels: {unexpected}")
    missing_labels = sorted(set(TARGET_LABELS) - set(frame["label"]))
    if missing_labels:
        raise ValueError(f"Test split is missing labels: {missing_labels}")

    frame["window_id"] = frame["window_path"].map(window_id_of)
    frame["pitch_group"] = (
        frame["label"].astype(str) + "_" + frame["note"].astype(str)
    )
    return frame.drop(columns=["manifest_label"])


def noise_source_lookup(
    condition: NoiseCondition,
    *,
    noisy_dir: str | Path = NOISY_DIR,
) -> dict[str, str]:
    """window_id -> the noise recording used, for one condition. Empty dict for clean.

    WHY PREDICTIONS NEED THIS (audit item 8). Several clean test windows can be corrupted with crops
    from the SAME ESC-50 recording, so their errors are not independent -- if that one recording
    happens to be especially destructive, every window that drew it fails together. `noise_stats`
    resamples clusters to handle exactly this kind of correlation, but it can only cluster on a
    column that is present in the prediction CSV. Attaching it here is what makes
    `--cluster noise_source` possible; recovering it afterwards would mean re-joining every
    prediction file against provenance by hand.
    """
    if condition.noise_type == "clean":
        return {}
    provenance_path = Path(noisy_dir) / NOISE_PROVENANCE_NAME
    if not provenance_path.is_file():
        return {}
    provenance = pd.read_csv(provenance_path)
    required = {"window_id", "noise_type", "snr_db", "noise_source"}
    if not required <= set(provenance.columns):
        return {}
    selected = provenance[
        (provenance["noise_type"] == condition.noise_type)
        & (provenance["snr_db"].astype(float) == float(condition.snr_db))
    ]
    if "replicate" in provenance.columns and condition.replicate is not None:
        selected = selected[
            selected["replicate"].astype(int) == int(condition.replicate)
        ]
    return dict(zip(selected["window_id"].astype(str), selected["noise_source"].astype(str)))


def condition_paths(
    frame: pd.DataFrame,
    condition: NoiseCondition,
    *,
    data_root: str | Path = ROOT,
    noisy_dir: str | Path = NOISY_DIR,
) -> list[Path]:
    if condition.noise_type == "clean":
        return [Path(data_root) / path for path in frame["window_path"]]
    return [
        out_path(
            condition.noise_type,
            int(condition.snr_db),
            window_id,
            replicate=int(condition.replicate or 0),
            noisy_dir=noisy_dir,
        )
        for window_id in frame["window_id"]
    ]


def load_official_summary(
    path: str | Path,
    *,
    expected_model_path: str | Path | None = None,
) -> dict[str, object]:
    path = Path(path)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read official clean summary at {path}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    assert_fingerprint(summary.get("config_fingerprint"), str(path))
    if summary.get("label_order") != TARGET_LABELS:
        raise ValueError(f"Unexpected label order in {path}")
    if expected_model_path is not None:
        expected_model_path = Path(expected_model_path)
        recorded = summary.get("output_files", {}).get("model", {}).get("sha256")
        if recorded != sha256_file(expected_model_path):
            raise ValueError(
                f"{expected_model_path} does not match the official clean summary"
            )
    return summary


def assert_clean_parity(
    measured_macro_f1: float,
    *,
    official_macro_f1: float | None,
    measured_examples: int,
    official_examples: int | None,
    tolerance: float = CLEAN_PARITY_TOLERANCE,
) -> None:
    """A missing official result is a hard failure, not a skipped gate."""
    if official_macro_f1 is None:
        raise ValueError("Official clean macro-F1 is required for the parity gate")
    if official_examples is None:
        raise ValueError("Official clean test-example count is required for parity")
    if measured_examples != official_examples:
        raise ValueError(
            f"Clean parity failed: evaluated {measured_examples} examples, "
            f"official result used {official_examples}"
        )
    difference = abs(measured_macro_f1 - official_macro_f1)
    if difference > tolerance:
        raise ValueError(
            "Clean parity failed: "
            f"measured macro-F1={measured_macro_f1:.6f}, "
            f"official={official_macro_f1:.6f}, "
            f"difference={difference:.6f} > {tolerance}"
        )


def _existing_output_paths(output_dir: Path, file_prefix: str) -> list[Path]:
    paths = [output_dir / "noise_sweep_summary.csv"]
    for condition in noise_conditions():
        paths.extend(
            [
                output_dir / f"{file_prefix}{condition.tag}.csv",
                output_dir / f"metrics_{condition.tag}.json",
            ]
        )
    return [path for path in paths if path.exists()]


def run_noise_evaluation(
    *,
    model_name: str,
    file_prefix: str,
    predict_scores: Callable[[list[Path]], np.ndarray],
    official_macro_f1: float | None,
    official_examples: int | None,
    model_sha256: str,
    score_type: str | None = "score",
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    windows_csv: str | Path = WINDOWS_CSV,
    manifest_labeled: str | Path = MANIFEST_LABELED,
    data_root: str | Path = ROOT,
    noisy_dir: str | Path = NOISY_DIR,
    manifest_csv: str | Path = MANIFEST_IN,
    manifest_fingerprint: str | Path = MANIFEST_FINGERPRINT,
) -> pd.DataFrame:
    """Evaluate clean first, enforce parity, then evaluate every frozen noise condition."""
    output_dir = (
        ARTIFACTS / model_name / "noise"
        if output_dir is None
        else Path(output_dir)
    )
    existing = _existing_output_paths(output_dir, file_prefix)
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing noise results: "
            + ", ".join(str(path) for path in existing[:5])
        )

    manifest = validate_noise_manifest(
        noisy_dir=noisy_dir,
        data_root=data_root,
        windows_csv=windows_csv,
        manifest_csv=manifest_csv,
        manifest_fingerprint=manifest_fingerprint,
    )
    frame = load_test_frame(
        windows_csv=windows_csv,
        manifest_labeled=manifest_labeled,
    )
    y_true = np.asarray(
        [LABEL_TO_INDEX[label] for label in frame["label"]],
        dtype=np.int64,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    clean_macro_f1: float | None = None

    for condition in noise_conditions():
        paths = condition_paths(
            frame,
            condition,
            data_root=data_root,
            noisy_dir=noisy_dir,
        )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{condition.tag}: missing {len(missing)} windows; first: {missing[0]}"
            )
        scores = np.asarray(predict_scores(paths))
        expected_shape = (len(frame), len(TARGET_LABELS))
        if scores.shape != expected_shape:
            raise ValueError(
                f"{condition.tag}: expected scores {expected_shape}, got {scores.shape}"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{condition.tag}: model produced non-finite scores")
        predictions = scores.argmax(axis=1)
        accuracy = float(accuracy_score(y_true, predictions))
        macro_f1 = float(
            f1_score(
                y_true,
                predictions,
                labels=range(len(TARGET_LABELS)),
                average="macro",
                zero_division=0,
            )
        )
        if condition.tag == "clean":
            assert_clean_parity(
                macro_f1,
                official_macro_f1=official_macro_f1,
                measured_examples=len(frame),
                official_examples=official_examples,
            )
            clean_macro_f1 = macro_f1

        # Which noise recording each window drew, so noise_stats can cluster on it (item 8).
        # "clean" for the clean condition, and "unknown" only if provenance is unreadable -- both
        # are single-valued, so clustering on them degrades to the ungrouped case rather than
        # silently splitting windows into meaningless groups.
        sources = noise_source_lookup(condition, noisy_dir=noisy_dir)
        predictions_frame = pd.DataFrame(
            {
                "window_id": frame["window_id"],
                "source_path": frame["source_path"],
                "pitch_group": frame["pitch_group"],
                "noise_source": [
                    "clean"
                    if condition.noise_type == "clean"
                    else sources.get(str(window_id), "unknown")
                    for window_id in frame["window_id"]
                ],
                "true_label": frame["label"],
                "predicted_label": [
                    TARGET_LABELS[index] for index in predictions
                ],
                "correct": y_true == predictions,
            }
        )
        if score_type is not None:
            for index, label in enumerate(TARGET_LABELS):
                predictions_frame[f"{score_type}_{label}"] = scores[:, index]
        predictions_frame.to_csv(
            output_dir / f"{file_prefix}{condition.tag}.csv",
            index=False,
        )

        report = classification_report(
            y_true,
            predictions,
            labels=range(len(TARGET_LABELS)),
            target_names=TARGET_LABELS,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "condition": condition.tag,
            "noise_type": condition.noise_type,
            "snr_db": condition.snr_db,
            "replicate": condition.replicate,
            "n": int(len(frame)),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "config_fingerprint": config_fingerprint(),
            "dataset": manifest["dataset"],
            "noise_manifest": {
                "path": str((Path(noisy_dir) / NOISE_MANIFEST_NAME).resolve()),
                "sha256": sha256_file(Path(noisy_dir) / NOISE_MANIFEST_NAME),
            },
            "model_sha256": model_sha256,
            "score_type": score_type,
            # How concentrated the noise draw was: 1.0 means every window drew a distinct
            # recording, 0.1 means ten windows shared each one. Low values mean window-level
            # errors are correlated through the noise, not only through the instrument.
            "noise_source_distinct_fraction": (
                None
                if condition.noise_type == "clean"
                else round(
                    predictions_frame["noise_source"].nunique() / max(len(frame), 1), 6
                )
            ),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                y_true,
                predictions,
                labels=range(len(TARGET_LABELS)),
            ).tolist(),
        }
        (output_dir / f"metrics_{condition.tag}.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "noise_type": condition.noise_type,
                "snr_db": condition.snr_db,
                "replicate": condition.replicate,
                "condition": condition.tag,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
        print(
            f"{condition.tag:<16} accuracy={accuracy:.4f} "
            f"macro-F1={macro_f1:.4f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    if clean_macro_f1 is None:
        raise AssertionError("Clean condition was not evaluated")
    summary["macro_f1_drop"] = clean_macro_f1 - summary["macro_f1"]
    summary["macro_f1_retention"] = summary["macro_f1"] / clean_macro_f1
    summary.to_csv(output_dir / "noise_sweep_summary.csv", index=False)
    return summary
