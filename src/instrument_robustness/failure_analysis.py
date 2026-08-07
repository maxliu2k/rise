"""Post-rerun analyses of instrument failures and noise-induced confusions.

This module consumes only frozen training features and saved test predictions. It never loads
audio, trains a model, or runs inference. Analysis 1 is an explicitly exploratory instrument-level
recall-loss analysis. Analysis 2 is the predeclared primary mechanism analysis relating acoustic
distance to noise-induced pairwise confusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import f1_score

from instrument_robustness.config import (
    ARTIFACTS,
    FEATURES,
    NOISE_TYPES,
    N_REPLICATES,
    SNRS,
    STATS_NPZ,
    TARGET_LABELS,
    assert_fingerprint,
    config_fingerprint,
)
from instrument_robustness.featurelib import SVM_FEATURE_NAMES
from instrument_robustness.noise_eval_common import NoiseCondition, noise_conditions
from instrument_robustness.noise_eval_svm import load_training_statistics
from instrument_robustness.noise_sweep import dataset_build_identity, sha256_file
from instrument_robustness.robustness_curve import benjamini_hochberg
from instrument_robustness.svm_model import load_svm_feature_names, load_svm_split

# Directory -> prediction-file prefix. The two are NOT the same string for every model: PANNs
# reports its fine-tune (`panns_ft_test_`) and MERT now reports its fine-tune from a separate
# directory, so a reader cannot infer one from the other.
#
# MERT IS THE FINE-TUNE HERE. The frozen probe made MERT the only pretrained model not
# fine-tuned, so every MERT-vs-AST and MERT-vs-PANNs association in this analysis was confounded
# by adaptation method rather than isolating architecture. Pointing at artifacts/mert_ft/ fixes
# that; it also means these outputs must be regenerated, because the committed ones were
# computed from the probe's predictions.
MODEL_PREFIXES = {
    "svm": "svm_test_",
    "cnn": "cnn_test_",
    "crnn": "crnn_test_",
    "mert_ft": "mert_ft_test_",
    "ast": "ast_test_",
    "panns": "panns_ft_test_",
}
IDENTITY_COLUMNS = (
    "window_id",
    "source_path",
    "pitch_group",
    "noise_source",
    "true_label",
)
N_PERMUTATIONS = 100_000
PERMUTATION_SEED = 0
ANALYSIS_FAMILY = (
    f"{len(MODEL_PREFIXES)} models x {len(NOISE_TYPES)} noise categories: "
    "acoustic distance vs confusion AUC"
)
# The sealed SCC build used for the corrected six-model rerun. Failure analysis is intentionally
# narrower than the reusable model evaluators: it must never accept the retired local build or an
# earlier SCC run merely because all six result folders happen to agree with one another.
CANONICAL_DATASET_FINGERPRINT = (
    "97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf"
)


@dataclass(frozen=True)
class ModelSweep:
    model: str
    frames: dict[str, pd.DataFrame]
    dataset: dict[str, object]
    noise_manifest_sha256: str
    model_sha256: str
    input_hashes: dict[str, str]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash_file_set(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _same_column(left: pd.Series, right: pd.Series) -> bool:
    return left.astype(str).reset_index(drop=True).equals(
        right.astype(str).reset_index(drop=True)
    )


def load_prediction_frame(path: str | Path) -> pd.DataFrame:
    """Load one prediction CSV and enforce the common paired-analysis schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    frame = pd.read_csv(path)
    required = set(IDENTITY_COLUMNS) | {"predicted_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no predictions")
    if frame["window_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate window IDs")
    if frame[list(IDENTITY_COLUMNS) + ["predicted_label"]].isna().any().any():
        raise ValueError(f"{path} contains missing identity or label values")
    observed = set(frame["true_label"]) | set(frame["predicted_label"])
    unexpected = sorted(observed - set(TARGET_LABELS))
    if unexpected:
        raise ValueError(f"{path} contains unexpected labels: {unexpected}")
    missing_truth = sorted(set(TARGET_LABELS) - set(frame["true_label"]))
    if missing_truth:
        raise ValueError(f"{path} is missing true classes: {missing_truth}")
    return frame.sort_values("window_id").reset_index(drop=True)


def _check_metric_condition(
    metrics: dict[str, object],
    condition: NoiseCondition,
    path: Path,
    frame: pd.DataFrame,
) -> None:
    assert_fingerprint(metrics.get("config_fingerprint"), str(path))
    expected = {
        "condition": condition.tag,
        "noise_type": condition.noise_type,
        "snr_db": condition.snr_db,
        "replicate": condition.replicate,
        "n": len(frame),
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"{path} has {key}={metrics.get(key)!r}; expected {value!r}"
            )
    report = metrics.get("classification_report")
    if not isinstance(report, dict) or not set(TARGET_LABELS) <= set(report):
        raise ValueError(f"{path} has no complete 12-class classification report")
    true = frame["true_label"].to_numpy()
    predicted = frame["predicted_label"].to_numpy()
    measured_accuracy = float(np.mean(true == predicted))
    measured_macro_f1 = float(
        f1_score(
            true,
            predicted,
            labels=TARGET_LABELS,
            average="macro",
            zero_division=0,
        )
    )
    if not np.isclose(float(metrics.get("accuracy", np.nan)), measured_accuracy):
        raise ValueError(f"{path} accuracy disagrees with its prediction CSV")
    if not np.isclose(float(metrics.get("macro_f1", np.nan)), measured_macro_f1):
        raise ValueError(f"{path} macro-F1 disagrees with its prediction CSV")
    expected_confusion = pd.crosstab(
        pd.Categorical(true, categories=TARGET_LABELS),
        pd.Categorical(predicted, categories=TARGET_LABELS),
        dropna=False,
    ).to_numpy()
    if not np.array_equal(np.asarray(metrics.get("confusion_matrix")), expected_confusion):
        raise ValueError(f"{path} confusion matrix disagrees with its prediction CSV")


def load_model_sweep(
    model: str,
    *,
    directory: str | Path,
    prefix: str,
    conditions: list[NoiseCondition] | None = None,
) -> ModelSweep:
    """Load one complete model sweep, rejecting mixed or partially written results."""
    directory = Path(directory)
    expected_conditions = noise_conditions() if conditions is None else conditions
    frames: dict[str, pd.DataFrame] = {}
    prediction_paths: list[Path] = []
    metric_paths: list[Path] = []
    reference_identity: pd.DataFrame | None = None
    dataset_json: str | None = None
    dataset: dict[str, object] | None = None
    manifest_sha: str | None = None
    model_sha: str | None = None
    metric_values: dict[str, tuple[float, float]] = {}

    for condition in expected_conditions:
        prediction_path = directory / f"{prefix}{condition.tag}.csv"
        metric_path = directory / f"metrics_{condition.tag}.json"
        frame = load_prediction_frame(prediction_path)
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read metrics file {metric_path}") from error
        if not isinstance(metrics, dict):
            raise ValueError(f"{metric_path} must contain a JSON object")
        _check_metric_condition(metrics, condition, metric_path, frame)

        current_identity = frame[list(IDENTITY_COLUMNS)]
        if reference_identity is None:
            reference_identity = current_identity
        else:
            for column in IDENTITY_COLUMNS:
                if column == "noise_source":
                    continue
                if not _same_column(reference_identity[column], current_identity[column]):
                    raise ValueError(
                        f"{model} condition {condition.tag} is not paired: {column} differs"
                    )

        current_dataset = metrics.get("dataset")
        if not isinstance(current_dataset, dict):
            raise ValueError(f"{metric_path} is missing the dataset identity")
        current_dataset_json = _canonical_json(current_dataset)
        current_manifest_sha = metrics.get("noise_manifest", {}).get("sha256")
        current_model_sha = metrics.get("model_sha256")
        if not isinstance(current_manifest_sha, str) or not current_manifest_sha:
            raise ValueError(f"{metric_path} is missing the noise-manifest hash")
        if not isinstance(current_model_sha, str) or not current_model_sha:
            raise ValueError(f"{metric_path} is missing the model hash")
        if dataset_json is None:
            dataset_json, dataset = current_dataset_json, current_dataset
            manifest_sha, model_sha = current_manifest_sha, current_model_sha
        elif (
            current_dataset_json != dataset_json
            or current_manifest_sha != manifest_sha
            or current_model_sha != model_sha
        ):
            raise ValueError(f"{model} mixes dataset, noise-manifest, or model identities")

        frames[condition.tag] = frame
        metric_values[condition.tag] = (
            float(metrics["accuracy"]),
            float(metrics["macro_f1"]),
        )
        prediction_paths.append(prediction_path)
        metric_paths.append(metric_path)

    summary_path = directory / "noise_sweep_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing sweep summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    required_summary = {"condition", "accuracy", "macro_f1"}
    missing_summary = required_summary - set(summary.columns)
    if missing_summary:
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing_summary)}")
    if summary["condition"].duplicated().any():
        raise ValueError(f"{summary_path} contains duplicate conditions")
    if set(summary["condition"]) != set(metric_values):
        raise ValueError(f"{summary_path} does not contain the complete condition set")
    indexed_summary = summary.set_index("condition")
    for tag, (accuracy, macro_f1) in metric_values.items():
        row = indexed_summary.loc[tag]
        if not np.isclose(float(row["accuracy"]), accuracy) or not np.isclose(
            float(row["macro_f1"]), macro_f1
        ):
            raise ValueError(f"{summary_path} disagrees with metrics for {tag}")
    assert dataset is not None and manifest_sha is not None and model_sha is not None
    return ModelSweep(
        model=model,
        frames=frames,
        dataset=dataset,
        noise_manifest_sha256=manifest_sha,
        model_sha256=model_sha,
        input_hashes={
            "predictions_sha256": _hash_file_set(prediction_paths),
            "metrics_sha256": _hash_file_set(metric_paths),
            "summary_sha256": sha256_file(summary_path),
        },
    )


def validate_shared_sweeps(sweeps: dict[str, ModelSweep]) -> None:
    """Require the same dataset, noise corpus, conditions, truth, and draws for every model."""
    if set(sweeps) != set(MODEL_PREFIXES):
        raise ValueError(
            f"Expected exactly the six models {sorted(MODEL_PREFIXES)}, got {sorted(sweeps)}"
        )
    reference = next(iter(sweeps.values()))
    for model, sweep in sweeps.items():
        if _canonical_json(sweep.dataset) != _canonical_json(reference.dataset):
            raise ValueError(f"{model} uses a different dataset build")
        if sweep.noise_manifest_sha256 != reference.noise_manifest_sha256:
            raise ValueError(f"{model} uses a different noise manifest")
        if sweep.frames.keys() != reference.frames.keys():
            raise ValueError(f"{model} does not contain the same conditions")
        for tag, frame in sweep.frames.items():
            reference_frame = reference.frames[tag]
            for column in IDENTITY_COLUMNS:
                if not _same_column(frame[column], reference_frame[column]):
                    raise ValueError(
                        f"{model} condition {tag} differs in paired column {column}"
                    )


def validate_analysis_dataset(
    sweeps: dict[str, ModelSweep],
    current_dataset: dict[str, object],
) -> None:
    """Accept only results and local inputs from the frozen canonical SCC build."""
    if not sweeps:
        raise ValueError("No model sweeps were provided")
    result_dataset = next(iter(sweeps.values())).dataset
    result_fingerprint = result_dataset.get("dataset_fingerprint")
    if result_fingerprint != CANONICAL_DATASET_FINGERPRINT:
        raise ValueError(
            "Failure analysis accepts only the canonical SCC dataset build "
            f"{CANONICAL_DATASET_FINGERPRINT}; result files identify "
            f"{result_fingerprint!r}. These results are stale or from a different build."
        )
    current_fingerprint = current_dataset.get("dataset_fingerprint")
    if current_fingerprint != CANONICAL_DATASET_FINGERPRINT:
        raise ValueError(
            "The active data root is not the canonical SCC dataset build: expected "
            f"{CANONICAL_DATASET_FINGERPRINT}, found {current_fingerprint!r}. Run this "
            "analysis against the sealed SCC data root, not a local or retired build."
        )
    if _canonical_json(current_dataset) != _canonical_json(result_dataset):
        raise ValueError(
            "The result files and active data root claim the canonical fingerprint but their "
            "full dataset identities differ"
        )


def validate_svm_training_features(
    feature_path: str | Path,
    test_summary_path: str | Path,
) -> None:
    """Require the analysis features to be the exact train array used by the final SVM."""
    feature_path = Path(feature_path)
    test_summary_path = Path(test_summary_path)
    try:
        summary = json.loads(test_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read final SVM summary at {test_summary_path}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"{test_summary_path} must contain a JSON object")
    recorded_hash = summary.get("input_files", {}).get("train", {}).get("sha256")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise ValueError(f"{test_summary_path} does not record the final SVM train-feature hash")
    current_hash = sha256_file(feature_path)
    if current_hash != recorded_hash:
        raise ValueError(
            "The SVM training features selected for failure analysis are not the exact array "
            "used by the final SVM model"
        )


def normalized_curve_area(snrs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Trapezoidal mean over SNR in dB; the last axis(s) may hold classes or pairs."""
    snrs = np.asarray(snrs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if snrs.ndim != 1 or len(snrs) < 2:
        raise ValueError("a curve needs at least two SNR values")
    if values.shape[0] != len(snrs):
        raise ValueError("the first value dimension must match the SNR count")
    if not np.all(np.isfinite(snrs)) or not np.all(np.isfinite(values)):
        raise ValueError("curve values must be finite")
    if len(np.unique(snrs)) != len(snrs):
        raise ValueError("curve SNR values must be distinct")
    order = np.argsort(snrs)
    ordered_snrs = snrs[order]
    span = ordered_snrs[-1] - ordered_snrs[0]
    if span <= 0:
        raise ValueError("curve SNR span must be positive")
    ordered_values = values[order]
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(ordered_values, ordered_snrs, axis=0)
    else:  # numpy < 2.0
        area = np.trapz(ordered_values, ordered_snrs, axis=0)
    return np.asarray(area / span)


def recalls(frame: pd.DataFrame) -> np.ndarray:
    """Recall for every class in the fixed label order."""
    true = frame["true_label"].to_numpy()
    predicted = frame["predicted_label"].to_numpy()
    values = np.empty(len(TARGET_LABELS), dtype=np.float64)
    for index, label in enumerate(TARGET_LABELS):
        actual = true == label
        if not actual.any():
            raise ValueError(f"prediction frame has no true examples for {label}")
        values[index] = np.mean(predicted[actual] == label)
    return values


def row_normalized_confusion(frame: pd.DataFrame) -> np.ndarray:
    """P(predicted=b | true=a) in the fixed class order."""
    label_to_index = {label: index for index, label in enumerate(TARGET_LABELS)}
    true = np.asarray([label_to_index[value] for value in frame["true_label"]])
    predicted = np.asarray([label_to_index[value] for value in frame["predicted_label"]])
    matrix = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=np.float64)
    np.add.at(matrix, (true, predicted), 1)
    supports = matrix.sum(axis=1)
    if np.any(supports == 0):
        raise ValueError("confusion matrix is missing a true class")
    return matrix / supports[:, None]


def pair_values(matrix: np.ndarray) -> np.ndarray:
    """Symmetric confusion for all 66 unordered class pairs."""
    matrix = np.asarray(matrix, dtype=np.float64)
    expected = (len(TARGET_LABELS), len(TARGET_LABELS))
    if matrix.shape != expected:
        raise ValueError(f"expected a {expected} matrix, got {matrix.shape}")
    rows, columns = np.triu_indices(len(TARGET_LABELS), k=1)
    return 0.5 * (matrix[rows, columns] + matrix[columns, rows])


def training_feature_tables(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return standardized class centroids, five fixed acoustic summaries, and distances."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.shape != (len(y), len(feature_names)):
        raise ValueError("training features, labels, and feature names are incompatible")
    if feature_names != list(SVM_FEATURE_NAMES):
        raise ValueError("unexpected 88-feature order")
    if set(y.tolist()) != set(range(len(TARGET_LABELS))):
        raise ValueError("training features must contain every class and no unknown classes")
    raw = X * np.asarray(raw_std) + np.asarray(raw_mean)
    centroids = np.vstack([X[y == index].mean(axis=0) for index in range(len(TARGET_LABELS))])
    global_mfcc = X[:, :40].mean(axis=0)

    centroid_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    feature_index = {name: index for index, name in enumerate(feature_names)}
    contrast_indices = [feature_index[f"contrast{index}_mean"] for index in range(7)]
    for index, label in enumerate(TARGET_LABELS):
        selected = y == index
        centroid_rows.append(
            {
                "label": label,
                "n_train": int(selected.sum()),
                **{
                    name: float(centroids[index, column])
                    for column, name in enumerate(feature_names)
                },
            }
        )
        profile_rows.append(
            {
                "label": label,
                "n_train": int(selected.sum()),
                "spectral_centroid_hz": float(
                    raw[selected, feature_index["centroid_mean"]].mean()
                ),
                "spectral_bandwidth_hz": float(
                    raw[selected, feature_index["bandwidth_mean"]].mean()
                ),
                "spectral_rolloff_hz": float(
                    raw[selected, feature_index["rolloff_mean"]].mean()
                ),
                "spectral_contrast_db": float(
                    raw[selected][:, contrast_indices].mean()
                ),
                "mfcc_profile_distance": float(
                    np.linalg.norm(centroids[index, :40] - global_mfcc)
                ),
            }
        )

    differences = centroids[:, None, :] - centroids[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return pd.DataFrame(centroid_rows), pd.DataFrame(profile_rows), distances


def instrument_recall_analysis(
    sweeps: dict[str, ModelSweep],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, object]] = []
    area_rows: list[dict[str, object]] = []
    for model, sweep in sweeps.items():
        clean = recalls(sweep.frames["clean"])
        for noise_type in NOISE_TYPES:
            for replicate in range(N_REPLICATES):
                losses = []
                for snr in SNRS:
                    tag = f"{noise_type}_{snr}_r{replicate}"
                    noisy = recalls(sweep.frames[tag])
                    loss = clean - noisy
                    losses.append(loss)
                    for index, label in enumerate(TARGET_LABELS):
                        condition_rows.append(
                            {
                                "model": model,
                                "label": label,
                                "noise_type": noise_type,
                                "snr_db": snr,
                                "replicate": replicate,
                                "clean_recall": clean[index],
                                "noisy_recall": noisy[index],
                                "recall_loss": loss[index],
                            }
                        )
                areas = normalized_curve_area(np.asarray(SNRS), np.vstack(losses))
                for index, label in enumerate(TARGET_LABELS):
                    area_rows.append(
                        {
                            "model": model,
                            "label": label,
                            "noise_type": noise_type,
                            "replicate": replicate,
                            "recall_loss_auc": float(areas[index]),
                        }
                    )
    area_frame = pd.DataFrame(area_rows)
    summary = (
        area_frame.groupby(["model", "label", "noise_type"], sort=False)[
            "recall_loss_auc"
        ]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_recall_loss_auc",
                "min": "min_recall_loss_auc",
                "max": "max_recall_loss_auc",
                "count": "n_replicates",
            }
        )
    )
    return pd.DataFrame(condition_rows), area_frame, summary


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = stats.rankdata(np.asarray(left, dtype=np.float64))
    right_rank = stats.rankdata(np.asarray(right, dtype=np.float64))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)


def scalar_permutation_spearman(
    predictor: np.ndarray, outcome: np.ndarray, permutations: np.ndarray
) -> dict[str, float | bool]:
    """Two-sided identity permutation for the 12-instrument exploratory association."""
    predictor_rank = stats.rankdata(np.asarray(predictor, dtype=np.float64))
    outcome_rank = stats.rankdata(np.asarray(outcome, dtype=np.float64))
    predictor_rank -= predictor_rank.mean()
    outcome_rank -= outcome_rank.mean()
    denominator = np.linalg.norm(predictor_rank) * np.linalg.norm(outcome_rank)
    if denominator == 0:
        return {"spearman_rho": float("nan"), "permutation_p": 1.0, "degenerate": True}
    observed = float(np.dot(predictor_rank, outcome_rank) / denominator)
    null = outcome_rank[permutations] @ predictor_rank / denominator
    p_value = (np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15) + 1) / (
        len(permutations) + 1
    )
    return {
        "spearman_rho": observed,
        "permutation_p": float(p_value),
        "degenerate": False,
    }


def exploratory_acoustic_associations(
    recall_summary: pd.DataFrame,
    acoustic_profiles: pd.DataFrame,
    permutations: np.ndarray,
) -> pd.DataFrame:
    predictors = [
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_hz",
        "spectral_contrast_db",
        "mfcc_profile_distance",
    ]
    rows: list[dict[str, object]] = []
    profiles = acoustic_profiles.set_index("label").loc[TARGET_LABELS]
    for (model, noise_type), selected in recall_summary.groupby(
        ["model", "noise_type"], sort=False
    ):
        outcome = selected.set_index("label").loc[TARGET_LABELS][
            "mean_recall_loss_auc"
        ].to_numpy()
        for predictor in predictors:
            result = scalar_permutation_spearman(
                profiles[predictor].to_numpy(), outcome, permutations
            )
            rows.append(
                {
                    "model": model,
                    "noise_type": noise_type,
                    "predictor": predictor,
                    **result,
                    "n_instruments": len(TARGET_LABELS),
                    "status": "exploratory; uncorrected",
                }
            )
    return pd.DataFrame(rows)


def pair_distance_table(distances: np.ndarray) -> pd.DataFrame:
    rows = []
    for first, second in combinations(range(len(TARGET_LABELS)), 2):
        rows.append(
            {
                "instrument_a": TARGET_LABELS[first],
                "instrument_b": TARGET_LABELS[second],
                "acoustic_distance": float(distances[first, second]),
            }
        )
    return pd.DataFrame(rows)


def pair_confusion_analysis(
    sweeps: dict[str, ModelSweep],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, object]] = []
    area_rows: list[dict[str, object]] = []
    pairs = list(combinations(range(len(TARGET_LABELS)), 2))
    for model, sweep in sweeps.items():
        clean = pair_values(row_normalized_confusion(sweep.frames["clean"]))
        for noise_type in NOISE_TYPES:
            for replicate in range(N_REPLICATES):
                deltas = []
                for snr in SNRS:
                    tag = f"{noise_type}_{snr}_r{replicate}"
                    noisy = pair_values(row_normalized_confusion(sweep.frames[tag]))
                    delta = noisy - clean
                    deltas.append(delta)
                    for pair_index, (first, second) in enumerate(pairs):
                        condition_rows.append(
                            {
                                "model": model,
                                "instrument_a": TARGET_LABELS[first],
                                "instrument_b": TARGET_LABELS[second],
                                "noise_type": noise_type,
                                "snr_db": snr,
                                "replicate": replicate,
                                "clean_symmetric_confusion": clean[pair_index],
                                "noisy_symmetric_confusion": noisy[pair_index],
                                "confusion_increase": delta[pair_index],
                            }
                        )
                areas = normalized_curve_area(np.asarray(SNRS), np.vstack(deltas))
                for pair_index, (first, second) in enumerate(pairs):
                    area_rows.append(
                        {
                            "model": model,
                            "instrument_a": TARGET_LABELS[first],
                            "instrument_b": TARGET_LABELS[second],
                            "noise_type": noise_type,
                            "replicate": replicate,
                            "confusion_increase_auc": float(areas[pair_index]),
                        }
                    )
    area_frame = pd.DataFrame(area_rows)
    summary = (
        area_frame.groupby(
            ["model", "instrument_a", "instrument_b", "noise_type"], sort=False
        )["confusion_increase_auc"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_confusion_increase_auc",
                "min": "min_confusion_increase_auc",
                "max": "max_confusion_increase_auc",
                "count": "n_replicates",
            }
        )
    )
    return pd.DataFrame(condition_rows), area_frame, summary


def matrix_permutation_spearman(
    distance_matrix: np.ndarray,
    outcome_matrix: np.ndarray,
    permutations: np.ndarray,
) -> dict[str, float | bool]:
    """Spearman test that permutes whole instrument rows/columns, preserving pair dependence."""
    count = len(TARGET_LABELS)
    expected = (count, count)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    outcome_matrix = np.asarray(outcome_matrix, dtype=np.float64)
    if distance_matrix.shape != expected or outcome_matrix.shape != expected:
        raise ValueError(f"both matrices must have shape {expected}")
    if permutations.ndim != 2 or permutations.shape[1] != count:
        raise ValueError(f"permutations must have shape (N, {count})")
    upper_a, upper_b = np.triu_indices(count, k=1)
    distance = distance_matrix[upper_a, upper_b]
    outcome = outcome_matrix[upper_a, upper_b]
    observed = _rank_correlation(distance, outcome)
    if not np.isfinite(observed):
        return {"spearman_rho": float("nan"), "permutation_p": 1.0, "degenerate": True}

    pair_index = np.full((count, count), -1, dtype=np.int64)
    pair_index[upper_a, upper_b] = np.arange(len(upper_a))
    pair_index[upper_b, upper_a] = np.arange(len(upper_a))
    mapped = pair_index[permutations[:, upper_a], permutations[:, upper_b]]
    distance_rank = stats.rankdata(distance)
    outcome_rank = stats.rankdata(outcome)
    distance_rank -= distance_rank.mean()
    outcome_rank -= outcome_rank.mean()
    denominator = np.linalg.norm(distance_rank) * np.linalg.norm(outcome_rank)
    null = outcome_rank[mapped] @ distance_rank / denominator
    p_value = (np.count_nonzero(np.abs(null) >= abs(observed) - 1e-15) + 1) / (
        len(permutations) + 1
    )
    return {
        "spearman_rho": observed,
        "permutation_p": float(p_value),
        "degenerate": False,
    }


def distance_confusion_tests(
    pair_summary: pd.DataFrame,
    distance_matrix: np.ndarray,
    permutations: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    count = len(TARGET_LABELS)
    index = {label: position for position, label in enumerate(TARGET_LABELS)}
    rows: list[dict[str, object]] = []
    p_values: dict[str, float] = {}
    for model in MODEL_PREFIXES:
        for noise_type in NOISE_TYPES:
            selected = pair_summary[
                (pair_summary["model"] == model)
                & (pair_summary["noise_type"] == noise_type)
            ]
            if len(selected) != count * (count - 1) // 2:
                raise ValueError(
                    f"{model}/{noise_type} does not contain all 66 instrument pairs"
                )
            outcome = np.zeros((count, count), dtype=np.float64)
            for row in selected.itertuples(index=False):
                first = index[row.instrument_a]
                second = index[row.instrument_b]
                outcome[first, second] = outcome[second, first] = (
                    row.mean_confusion_increase_auc
                )
            result = matrix_permutation_spearman(
                distance_matrix, outcome, permutations
            )
            label = f"{model}:{noise_type}"
            p_values[label] = float(result["permutation_p"])
            rows.append(
                {
                    "comparison": label,
                    "model": model,
                    "noise_type": noise_type,
                    **result,
                    "n_pairs": count * (count - 1) // 2,
                    "n_permutations": len(permutations),
                    "alternative": "two-sided",
                }
            )

    correction = benjamini_hochberg(
        p_values,
        alpha=0.05,
        family=ANALYSIS_FAMILY,
    )
    corrected = {record["label"]: record for record in correction["results"]}
    for row in rows:
        record = corrected[row["comparison"]]
        row["bh_q_value"] = record["q_value"]
        row["bh_rejected_at_0.05"] = record["rejected"]
    return pd.DataFrame(rows), correction


def _permutations(seed: int, n_permutations: int) -> np.ndarray:
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    random = np.random.default_rng(seed)
    return np.vstack(
        [random.permutation(len(TARGET_LABELS)) for _ in range(n_permutations)]
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def run_analysis(
    *,
    artifacts_root: str | Path = ARTIFACTS,
    feature_dir: str | Path = FEATURES / "svm",
    stats_path: str | Path = STATS_NPZ,
    output_dir: str | Path = ARTIFACTS / "failure_analysis",
    n_permutations: int = N_PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
    overwrite: bool = False,
) -> dict[str, object]:
    artifacts_root = Path(artifacts_root)
    feature_dir = Path(feature_dir)
    stats_path = Path(stats_path)
    output_dir = Path(output_dir)
    output_names = [
        "training_feature_centroids.csv",
        "training_acoustic_profiles.csv",
        "acoustic_distances.csv",
        "instrument_recall_loss_by_condition.csv",
        "instrument_recall_loss_auc_by_replicate.csv",
        "instrument_recall_loss_summary.csv",
        "instrument_acoustic_associations_exploratory.csv",
        "pair_confusion_by_condition.csv",
        "pair_confusion_auc_by_replicate.csv",
        "pair_confusion_summary.csv",
        "distance_confusion_tests.csv",
        "analysis_manifest.json",
    ]
    existing = [output_dir / name for name in output_names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing analysis outputs: "
            + ", ".join(str(path) for path in existing[:5])
        )

    sweeps = {
        model: load_model_sweep(
            model,
            directory=artifacts_root / model / "noise",
            prefix=prefix,
        )
        for model, prefix in MODEL_PREFIXES.items()
    }
    validate_shared_sweeps(sweeps)
    validate_analysis_dataset(sweeps, dataset_build_identity())

    train_feature_path = feature_dir / "train.npz"
    validate_svm_training_features(
        train_feature_path,
        artifacts_root / "svm" / "test_summary.json",
    )

    X, y = load_svm_split("train", feature_dir=feature_dir)
    feature_names = load_svm_feature_names("train", feature_dir=feature_dir)
    if feature_names is None:
        raise ValueError(f"{feature_dir / 'train.npz'} has no saved feature names")
    raw_mean, raw_std = load_training_statistics(stats_path)
    centroids, acoustic_profiles, distances = training_feature_tables(
        X, y, feature_names, raw_mean, raw_std
    )
    permutations = _permutations(seed, n_permutations)

    recall_conditions, recall_areas, recall_summary = instrument_recall_analysis(sweeps)
    exploratory = exploratory_acoustic_associations(
        recall_summary, acoustic_profiles, permutations
    )
    pair_conditions, pair_areas, pair_summary = pair_confusion_analysis(sweeps)
    primary_tests, correction = distance_confusion_tests(
        pair_summary, distances, permutations
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "training_feature_centroids.csv": centroids,
        "training_acoustic_profiles.csv": acoustic_profiles,
        "acoustic_distances.csv": pair_distance_table(distances),
        "instrument_recall_loss_by_condition.csv": recall_conditions,
        "instrument_recall_loss_auc_by_replicate.csv": recall_areas,
        "instrument_recall_loss_summary.csv": recall_summary,
        "instrument_acoustic_associations_exploratory.csv": exploratory,
        "pair_confusion_by_condition.csv": pair_conditions,
        "pair_confusion_auc_by_replicate.csv": pair_areas,
        "pair_confusion_summary.csv": pair_summary,
        "distance_confusion_tests.csv": primary_tests,
    }
    for name, frame in tables.items():
        _write_csv(frame, output_dir / name)

    first_sweep = next(iter(sweeps.values()))
    manifest: dict[str, object] = {
        "state": "complete",
        "analysis_version": 1,
        "config_fingerprint": config_fingerprint(),
        "dataset": first_sweep.dataset,
        "noise_manifest_sha256": first_sweep.noise_manifest_sha256,
        "models": {
            model: {
                "model_sha256": sweep.model_sha256,
                **sweep.input_hashes,
            }
            for model, sweep in sweeps.items()
        },
        "training_inputs": {
            "features": str(train_feature_path.resolve()),
            "features_sha256": sha256_file(train_feature_path),
            "statistics": str(stats_path.resolve()),
            "statistics_sha256": sha256_file(stats_path),
        },
        "protocol": {
            "analysis_1_status": "exploratory",
            "instrument_failure_metric": "recall loss relative to clean",
            "curve_summary": "trapezoidal integral over dB divided by measured SNR span",
            "analysis_1_predictors": [
                "spectral_centroid_hz",
                "spectral_bandwidth_hz",
                "spectral_rolloff_hz",
                "spectral_contrast_db",
                "mfcc_profile_distance",
            ],
            "analysis_1_p_values": "two-sided identity permutation; exploratory and uncorrected",
            "analysis_2_status": "primary failure-mechanism analysis",
            "acoustic_distance": "Euclidean distance between train-only standardized 88D class centroids",
            "pair_confusion": "mean of the two row-normalized directional confusion rates",
            "clean_subtraction": True,
            "replicate_test_value": "mean of replicate-specific confusion-increase AUCs",
            "matrix_permutation": "complete instrument row/column identity permutation",
            "alternative": "two-sided",
            "multiple_testing": correction,
            "n_permutations": n_permutations,
            "permutation_seed": seed,
            "snrs_db": SNRS,
            "noise_types": NOISE_TYPES,
            "n_replicates": N_REPLICATES,
            "fixed_label_order": TARGET_LABELS,
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "outputs": {
            name: sha256_file(output_dir / name)
            for name in tables
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen instrument-failure and acoustic-confusion analyses."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ARTIFACTS / "failure_analysis"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = run_analysis(output_dir=args.output_dir, overwrite=args.overwrite)
    print(
        f"completed Analysis 1 and Analysis 2 for {len(manifest['models'])} models; "
        f"wrote {args.output_dir}"
    )


if __name__ == "__main__":
    main()
