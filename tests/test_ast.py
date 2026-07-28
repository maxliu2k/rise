from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from instrument_robustness.ast_data import ASTWindowDataset, resolve_ast_labels
from instrument_robustness.config import TARGET_LABELS, config_fingerprint
from instrument_robustness.pretrained_extractors import build_ast_model
from instrument_robustness.train_ast import _balanced_class_weights, _write_test_reports


NEW_LABELS = {"double-bass", "french-horn", "oboe"}
TWELVE_LABELS = list(TARGET_LABELS)


def write_manifest(path: Path, labels, counts=None) -> None:
    counts = counts or {label: 1 for label in labels}
    row_count = 0
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
                    row_count += 1
    with path.with_name("windows_fingerprint.json").open("w") as fingerprint:
        json.dump(
            {
                "fingerprint": config_fingerprint(),
                "n_rows": row_count,
            },
            fingerprint,
        )


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
            manifest_path.with_name("windows_fingerprint.json").write_text(
                json.dumps(
                    {
                        "fingerprint": config_fingerprint(),
                        "n_rows": len(rows),
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "french-horn"):
                resolve_ast_labels(manifest_path)

    def test_rejects_windows_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "windows.csv"
            write_manifest(manifest_path, TWELVE_LABELS)
            manifest_path.with_name("windows_fingerprint.json").unlink()

            with self.assertRaisesRegex(RuntimeError, "predates config fingerprinting"):
                resolve_ast_labels(manifest_path)

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


if __name__ == "__main__":
    unittest.main()
