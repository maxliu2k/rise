from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.svm import SVC

from instrument_robustness.config import (
    TARGET_LABELS,
    StaleArtifactError,
    config_fingerprint,
    config_fingerprint_json,
)
from instrument_robustness.finalize_svm import main as finalize_main
from instrument_robustness.svm_model import (
    SVMConfig,
    build_svm,
    load_svm,
    load_svm_split,
)
from instrument_robustness.train_svm import candidate_configs, main

N_CLASSES = len(TARGET_LABELS)


def write_split(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    np.savez(
        path,
        X=X,
        y=y,
        feature_names=np.array([f"feature_{index}" for index in range(88)]),
        label_names=np.array(TARGET_LABELS),
        config_fingerprint=np.asarray(config_fingerprint_json()),
    )


class SVMTests(unittest.TestCase):
    def test_loader_does_not_standardize_features_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            feature_dir = Path(temporary_dir)
            X = np.arange(N_CLASSES * 88, dtype=np.float32).reshape(N_CLASSES, 88)
            y = np.arange(N_CLASSES, dtype=np.int64)
            write_split(feature_dir / "train.npz", X, y)

            loaded_X, loaded_y = load_svm_split(
                "train",
                feature_dir=feature_dir,
            )

            np.testing.assert_array_equal(loaded_X, X)
            np.testing.assert_array_equal(loaded_y, y)

    def test_loader_rejects_an_unfingerprinted_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            feature_dir = Path(temporary_dir)
            np.savez(
                feature_dir / "train.npz",
                X=np.zeros((1, 88), dtype=np.float32),
                y=np.zeros(1, dtype=np.int64),
                label_names=np.asarray(TARGET_LABELS),
            )

            with self.assertRaises(StaleArtifactError):
                load_svm_split("train", feature_dir=feature_dir)

    def test_grid_and_model_are_rbf_only(self) -> None:
        configs = candidate_configs(
            c_values=[0.1, 1.0],
            gamma_values=["scale", 0.01],
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {(config.C, config.gamma) for config in configs},
            {(0.1, "scale"), (0.1, 0.01), (1.0, "scale"), (1.0, 0.01)},
        )
        for config in configs:
            model = build_svm(config)
            self.assertIsInstance(model, SVC)
            self.assertEqual(model.kernel, "rbf")
            self.assertEqual(model.C, config.C)
            self.assertEqual(model.gamma, config.gamma)

    def test_training_saves_ranked_results_without_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            feature_dir = root / "features"
            output_dir = root / "outputs"
            feature_dir.mkdir()

            y_train = np.repeat(np.arange(N_CLASSES, dtype=np.int64), 2)
            X_train = np.zeros((len(y_train), 88), dtype=np.float32)
            X_train[np.arange(len(y_train)), y_train] = 3.0
            X_train[1::2, N_CLASSES] = 0.1

            y_val = np.arange(N_CLASSES, dtype=np.int64)
            X_val = np.zeros((len(y_val), 88), dtype=np.float32)
            X_val[np.arange(len(y_val)), y_val] = 3.0

            write_split(feature_dir / "train.npz", X_train, y_train)
            write_split(feature_dir / "val.npz", X_val, y_val)
            self.assertFalse((feature_dir / "test.npz").exists())

            argv = [
                "train_svm",
                "--feature-dir",
                str(feature_dir),
                "--output-dir",
                str(output_dir),
                "--C-values",
                "0.1",
                "1.0",
                "--gamma-values",
                "scale",
            ]
            with patch.object(sys, "argv", argv):
                main()

            with (output_dir / "validation_search.csv").open(
                newline="",
                encoding="utf-8",
            ) as file:
                results = list(csv.DictReader(file))
            with (output_dir / "validation_summary.json").open(
                encoding="utf-8"
            ) as file:
                summary = json.load(file)
            model = load_svm(output_dir / "best_model.joblib")
            with (output_dir / "validation_confusion_matrix.csv").open(
                newline="",
                encoding="utf-8",
            ) as file:
                confusion_rows = list(csv.reader(file))

            self.assertEqual(len(results), 2)
            self.assertTrue(all(row["kernel"] == "rbf" for row in results))
            self.assertGreaterEqual(
                float(results[0]["validation_macro_f1"]),
                float(results[1]["validation_macro_f1"]),
            )
            self.assertEqual(summary["selection_metric"], "validation_macro_f1")
            self.assertEqual(
                summary["validation_metrics"]["macro_f1"],
                float(results[0]["validation_macro_f1"]),
            )
            self.assertFalse(summary["test_evaluated"])
            self.assertEqual(
                summary["validation_metrics"]["confusion_matrix"],
                np.eye(N_CLASSES, dtype=int).tolist(),
            )
            self.assertEqual(len(confusion_rows), N_CLASSES + 1)
            self.assertEqual(confusion_rows[0][0], "actual")
            self.assertEqual(confusion_rows[0][1:], TARGET_LABELS)
            self.assertEqual(
                summary["feature_schema"]["names"],
                [f"feature_{index}" for index in range(88)],
            )
            self.assertEqual(summary["model_fit_splits"], ["train"])
            self.assertEqual(
                summary["final_test_policy"]["final_fit_splits"],
                ["train"],
            )
            self.assertEqual(
                summary["final_test_policy"]["test_evaluations_allowed"],
                1,
            )
            self.assertEqual(
                summary["input_files"]["train"]["sha256"],
                hashlib.sha256(
                    (feature_dir / "train.npz").read_bytes()
                ).hexdigest(),
            )
            self.assertIn("scikit_learn", summary["software_versions"])
            self.assertEqual(len(summary["output_files"]["model"]["sha256"]), 64)
            self.assertEqual(summary["config_fingerprint"], config_fingerprint())
            self.assertEqual(model.kernel, "rbf")
            self.assertEqual(model.n_features_in_, 88)

    def test_finalization_fits_train_only_and_only_runs_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            feature_dir = root / "features"
            output_dir = root / "outputs"
            feature_dir.mkdir()
            output_dir.mkdir()

            y_train = np.repeat(np.arange(N_CLASSES, dtype=np.int64), 2)
            X_train = np.zeros((len(y_train), 88), dtype=np.float32)
            X_train[np.arange(len(y_train)), y_train] = 3.0
            y_val = np.arange(N_CLASSES, dtype=np.int64)
            X_val = np.zeros((len(y_val), 88), dtype=np.float32)
            X_val[np.arange(len(y_val)), y_val] = 3.0
            y_test = np.arange(N_CLASSES, dtype=np.int64)
            X_test = np.zeros((len(y_test), 88), dtype=np.float32)
            X_test[np.arange(len(y_test)), y_test] = 3.0

            write_split(feature_dir / "train.npz", X_train, y_train)
            write_split(feature_dir / "val.npz", X_val, y_val)
            write_split(feature_dir / "test.npz", X_test, y_test)

            validation_summary = {
                "best_config": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
                "label_order": TARGET_LABELS,
                "config_fingerprint": config_fingerprint(),
                "test_evaluated": False,
                "feature_schema": {
                    "dimension": 88,
                    "names": [f"feature_{index}" for index in range(88)],
                    "label_order": TARGET_LABELS,
                    "standardization": "training-set statistics",
                },
                "input_files": {
                    "train": {
                        "sha256": hashlib.sha256(
                            (feature_dir / "train.npz").read_bytes()
                        ).hexdigest()
                    },
                    "val": {
                        "sha256": hashlib.sha256(
                            (feature_dir / "val.npz").read_bytes()
                        ).hexdigest()
                    },
                },
            }
            with (output_dir / "validation_summary.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(validation_summary, file)

            argv = [
                "finalize_svm",
                "--feature-dir",
                str(feature_dir),
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(sys, "argv", argv):
                finalize_main()

            with (output_dir / "test_summary.json").open(
                encoding="utf-8"
            ) as file:
                summary = json.load(file)
            with (output_dir / "final_evaluation_status.json").open(
                encoding="utf-8"
            ) as file:
                status = json.load(file)
            model = load_svm(output_dir / "final_model.joblib")

            # Train only: val is loaded for hash/feature-order checks but must not be fit on.
            # If this flips back to ["train", "val"] / 3 * N_CLASSES, SVM has regained a data
            # advantage over the CNN/CRNN/AST/PANNs families and the comparison is unfair.
            self.assertEqual(summary["model_fit_splits"], ["train"])
            self.assertEqual(summary["final_fit_examples"], 2 * N_CLASSES)
            self.assertEqual(summary["test_examples"], N_CLASSES)
            self.assertEqual(summary["test_metrics"]["macro_f1"], 1.0)
            self.assertTrue(summary["test_evaluated"])
            self.assertEqual(summary["test_evaluation_count"], 1)
            self.assertEqual(status["state"], "complete")
            self.assertEqual(model.n_features_in_, 88)

            with (
                patch.object(sys, "argv", argv),
                patch(
                    "instrument_robustness.finalize_svm.load_svm_split"
                ) as load_split,
            ):
                with self.assertRaises(FileExistsError):
                    finalize_main()
                load_split.assert_not_called()


if __name__ == "__main__":
    unittest.main()
