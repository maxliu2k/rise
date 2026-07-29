from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is optional; install the ast extra to run AST tests")

from instrument_robustness.ast_data import (
    ASTWindowDataset,
    _load_window,
    resolve_ast_labels,
    validate_ast_window_files,
)
from instrument_robustness.config import (
    SR,
    TARGET_LABELS,
    WINDOW_S,
    artifact_fingerprint_path,
    write_artifact_fingerprint,
)
from instrument_robustness.pretrained_extractors import build_ast_model
from instrument_robustness.train_ast import (
    _balanced_accuracy,
    _balanced_class_weights,
    _matthews_correlation,
    _write_test_reports,
)


NEW_LABELS = {"double-bass", "french-horn", "oboe"}
TWELVE_LABELS = list(TARGET_LABELS)


def write_manifest(path: Path, labels, counts=None) -> None:
    counts = counts or {label: 1 for label in labels}
    with path.open("w", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=["window_path", "label", "split"],
        )
        writer.writeheader()
        for split in ("train", "val", "test"):
            for label in reversed(labels):
                for index in range(counts[label] if split == "train" else 1):
                    writer.writerow(
                        {
                            "window_path": f"{split}/{label}/{index}.wav",
                            "label": label,
                            "split": split,
                        }
                    )
    write_artifact_fingerprint(path, "step5_normalize")


class ASTLabelTests(unittest.TestCase):
    def test_resolves_new_instruments_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)

            self.assertEqual(resolve_ast_labels(manifest_path), TWELVE_LABELS)

    def test_dataset_uses_configured_label_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)
            dataset = ASTWindowDataset(
                "train",
                extractor=object(),
                manifest_path=manifest_path,
                root=Path(temporary_dir),
            )

            self.assertEqual(dataset.label_names, TWELVE_LABELS)
            self.assertEqual(set(dataset.labels), set(range(12)))

    def test_rejects_stale_nine_class_manifest(self) -> None:
        old_labels = [label for label in TWELVE_LABELS if label not in NEW_LABELS]
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "windows.csv"
            write_manifest(manifest_path, old_labels)

            with self.assertRaisesRegex(ValueError, "configured 12-class dataset"):
                resolve_ast_labels(manifest_path)

    def test_rejects_class_missing_from_a_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)
            with manifest_path.open(newline="") as manifest:
                rows = list(csv.DictReader(manifest))
            rows = [
                row
                for row in rows
                if not (row["split"] == "test" and row["label"] == "french-horn")
            ]
            with manifest_path.open("w", newline="") as manifest:
                writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            write_artifact_fingerprint(manifest_path, "step5_normalize")

            with self.assertRaisesRegex(ValueError, "french-horn"):
                resolve_ast_labels(manifest_path)

    def test_rejects_windows_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)
            artifact_fingerprint_path(manifest_path).unlink()

            with self.assertRaisesRegex(RuntimeError, "no provenance sidecar"):
                resolve_ast_labels(manifest_path)

    def test_rejects_manifest_with_missing_window_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path = root / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)

            with self.assertRaisesRegex(FileNotFoundError, "AST window file"):
                validate_ast_window_files(manifest_path, root)

    def test_validates_all_manifest_window_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path = root / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)
            with manifest_path.open(newline="") as manifest:
                rows = list(csv.DictReader(manifest))
            for row in rows:
                window_path = root / row["window_path"]
                window_path.parent.mkdir(parents=True, exist_ok=True)
                window_path.touch()

            self.assertEqual(
                validate_ast_window_files(manifest_path, root),
                len(rows),
            )

    def test_rejects_window_with_wrong_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            window_path = Path(temporary_dir) / "short.wav"
            sf.write(window_path, np.ones(SR, dtype=np.float32), SR)

            with self.assertRaisesRegex(ValueError, "Expected exactly"):
                _load_window(window_path)

    def test_imbalance_metrics_reward_per_class_recall(self) -> None:
        true_labels = np.asarray([0, 0, 0, 1, 2])
        perfect_predictions = true_labels.copy()
        collapsed_predictions = np.zeros_like(true_labels)

        self.assertEqual(
            _balanced_accuracy(true_labels, perfect_predictions, 3),
            1.0,
        )
        self.assertEqual(
            _matthews_correlation(true_labels, perfect_predictions, 3),
            1.0,
        )
        self.assertAlmostEqual(
            _balanced_accuracy(true_labels, collapsed_predictions, 3),
            1 / 3,
        )
        self.assertEqual(
            _matthews_correlation(true_labels, collapsed_predictions, 3),
            0.0,
        )

    def test_balanced_weights_favor_underrepresented_classes(self) -> None:
        labels = [0] * 8 + [1] * 2 + [2]
        weights = _balanced_class_weights(labels, 3)

        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])

    def test_builds_classifier_head_for_all_resolved_labels(self) -> None:
        fake_transformers = types.ModuleType("transformers")

        class FakeConfig:
            @classmethod
            def from_pretrained(cls, _model_name):
                return types.SimpleNamespace()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, _model_name, *, config, ignore_mismatched_sizes):
                self.assertTrue(ignore_mismatched_sizes)
                return types.SimpleNamespace(config=config)

        fake_transformers.ASTConfig = FakeConfig
        fake_transformers.ASTForAudioClassification = FakeModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = build_ast_model(TWELVE_LABELS)

        self.assertEqual(model.config.num_labels, 12)
        self.assertEqual(model.config.label2id["double-bass"], 3)
        self.assertEqual(model.config.label2id["french-horn"], 5)
        self.assertEqual(model.config.label2id["oboe"], 6)

    def test_reports_include_new_instruments_and_families(self) -> None:
        true_labels = np.arange(len(TWELVE_LABELS))
        predicted_labels = true_labels.copy()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            reports = _write_test_reports(
                output_dir,
                true_labels,
                predicted_labels,
                TWELVE_LABELS,
            )

            self.assertEqual(len(reports["per_instrument"]), 12)
            self.assertEqual(
                {row["family"] for row in reports["per_family"]},
                {"strings", "woodwinds", "brass"},
            )
            self.assertEqual(reports["summary"]["macro_f1"], 1.0)
            self.assertEqual(reports["summary"]["balanced_accuracy"], 1.0)
            self.assertEqual(reports["summary"]["mcc"], 1.0)


if __name__ == "__main__":
    unittest.main()
