"""Evaluate the frozen final SVM on the shared noise sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from instrument_robustness.config import (
    ARTIFACTS,
    STATS_NPZ,
    assert_serialized_fingerprint,
)
from instrument_robustness.featurelib import SVM_FEATURE_NAMES, svm_vector
from instrument_robustness.noise_eval_common import (
    load_official_summary,
    run_noise_evaluation,
)
from instrument_robustness.noise_sweep import read_audio_window, sha256_file
from instrument_robustness.svm_model import load_svm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ARTIFACTS / "svm" / "final_model.joblib",
    )
    parser.add_argument(
        "--clean-summary",
        type=Path,
        default=ARTIFACTS / "svm" / "test_summary.json",
    )
    parser.add_argument("--stats", type=Path, default=STATS_NPZ)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_training_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        assert_serialized_fingerprint(
            data["config_fingerprint"]
            if "config_fingerprint" in data
            else None,
            str(path),
        )
        if data["computed_on"].item() != "train":
            raise ValueError(f"{path} was not computed from train only")
        feature_names = data["svm_feature_names"].astype(str).tolist()
        if feature_names != SVM_FEATURE_NAMES:
            raise ValueError(f"{path} has an unexpected SVM feature order")
        mean = np.asarray(data["svm_mean"], dtype=np.float32)
        std = np.asarray(data["svm_std"], dtype=np.float32)
    expected_shape = (len(SVM_FEATURE_NAMES),)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            f"{path} has incompatible SVM statistics: {mean.shape}, {std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError(f"{path} contains non-finite SVM statistics")
    if np.any(std <= 0):
        raise ValueError(f"{path} contains non-positive SVM standard deviations")
    return mean, std


def main() -> None:
    args = parse_args()
    model = load_svm(args.model)
    summary = load_official_summary(
        args.clean_summary,
        expected_model_path=args.model,
    )
    mean, std = load_training_statistics(args.stats)

    def predict_scores(paths: list[Path]) -> np.ndarray:
        raw = np.vstack(
            [svm_vector(read_audio_window(path)) for path in paths]
        ).astype(np.float32, copy=False)
        standardized = (raw - mean) / std
        return np.asarray(model.decision_function(standardized), dtype=np.float64)

    run_noise_evaluation(
        model_name="svm",
        file_prefix="svm_test_",
        predict_scores=predict_scores,
        official_macro_f1=float(summary["test_metrics"]["macro_f1"]),
        official_examples=int(summary["test_examples"]),
        model_sha256=sha256_file(args.model),
        score_type="score",
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
