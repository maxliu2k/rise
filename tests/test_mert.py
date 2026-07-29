from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from instrument_robustness.config import (
    SR,
    TARGET_LABELS,
    config_fingerprint_json,
    write_artifact_fingerprint,
)
from instrument_robustness.extract_mert import extract_mert_splits
from instrument_robustness.finalize_mert import main as finalize_mert
from instrument_robustness.mert_data import (
    MERT_HIDDEN_SIZE,
    MERT_NUM_LAYERS,
    load_mert_embedding_metadata,
    load_mert_embeddings,
    load_mert_examples,
    validate_mert_windows,
)
from instrument_robustness.pretrained_extractors import mert_batch_input
from instrument_robustness.train_mert import (
    class_weight_vector,
    main as train_mert,
    train_candidate,
)


class FakeProcessor:
    def __init__(self) -> None:
        self.sampling_rate = None
        self.waveforms = None

    def __call__(self, waveforms, *, sampling_rate, return_tensors, padding):
        self.sampling_rate = sampling_rate
        self.waveforms = waveforms
        return {
            "input_values": np.stack(waveforms),
            "return_tensors": return_tensors,
            "padding": padding,
        }


def write_embedding_split(path: Path, labels: np.ndarray) -> None:
    np.savez(
        path,
        X=np.zeros(
            (len(labels), MERT_NUM_LAYERS, MERT_HIDDEN_SIZE),
            dtype=np.float32,
        ),
        y=labels.astype(np.int64),
        label_names=np.asarray(TARGET_LABELS),
        model_id=np.asarray("m-a-p/MERT-v1-95M"),
        model_revision=np.asarray("test-revision"),
        pooling=np.asarray("mean_over_time_per_hidden_layer"),
        config_fingerprint=np.asarray(config_fingerprint_json()),
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MERTTests(unittest.TestCase):
    def test_examples_follow_authoritative_window_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            window_dir = root / "work" / "windows"
            window_dir.mkdir(parents=True)
            train_window = window_dir / "train.wav"
            val_window = window_dir / "val.wav"
            train_window.touch()
            val_window.touch()
            windows_csv = root / "windows.csv"
            pd.DataFrame(
                [
                    {
                        "window_path": "work/windows/train.wav",
                        "source_path": "source/train.mp3",
                        "label": "violin",
                        "split": "train",
                    },
                    {
                        "window_path": "work/windows/val.wav",
                        "source_path": "source/val.mp3",
                        "label": "cello",
                        "split": "val",
                    },
                ]
            ).to_csv(windows_csv, index=False)
            write_artifact_fingerprint(windows_csv, "step5_normalize")

            examples = load_mert_examples(
                "train", windows_csv=windows_csv, data_root=root
            )

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].label, "violin")
            self.assertEqual(examples[0].target, TARGET_LABELS.index("violin"))
            self.assertEqual(examples[0].source_path, "source/train.mp3")

    def test_batch_input_resamples_to_mert_rate(self) -> None:
        processor = FakeProcessor()
        waveforms = [np.zeros(SR * 3, dtype=np.float32) for _ in range(2)]

        result = mert_batch_input(waveforms, processor)

        self.assertEqual(processor.sampling_rate, 24000)
        self.assertEqual(len(processor.waveforms), 2)
        self.assertEqual(processor.waveforms[0].shape, (24000 * 3,))
        self.assertEqual(result["input_values"].shape, (2, 24000 * 3))
        self.assertEqual(result["return_tensors"], "pt")
        self.assertTrue(result["padding"])

    def test_window_preflight_requires_all_twelve_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rows = []
            for label in TARGET_LABELS:
                path = root / "work" / "windows" / f"{label}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                rows.append(
                    {
                        "window_path": str(path.relative_to(root)),
                        "source_path": f"source/{label}.mp3",
                        "label": label,
                        "split": "train",
                    }
                )
            windows_csv = root / "windows.csv"
            pd.DataFrame(rows).to_csv(windows_csv, index=False)
            write_artifact_fingerprint(windows_csv, "step5_normalize")

            counts = validate_mert_windows(
                splits=("train",),
                windows_csv=windows_csv,
                data_root=root,
            )

            self.assertEqual(counts, {"train": len(TARGET_LABELS)})

    def test_embedding_loader_validates_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            feature_dir = Path(temporary_dir)
            X = np.zeros(
                (len(TARGET_LABELS), MERT_NUM_LAYERS, MERT_HIDDEN_SIZE),
                dtype=np.float32,
            )
            y = np.arange(len(TARGET_LABELS), dtype=np.int64)
            np.savez(
                feature_dir / "train.npz",
                X=X,
                y=y,
                label_names=np.asarray(TARGET_LABELS),
                model_id=np.asarray("m-a-p/MERT-v1-95M"),
                model_revision=np.asarray("test-revision"),
                pooling=np.asarray("mean_over_time_per_hidden_layer"),
                config_fingerprint=np.asarray(config_fingerprint_json()),
            )

            loaded_X, loaded_y = load_mert_embeddings(
                "train", feature_dir=feature_dir
            )

            np.testing.assert_array_equal(loaded_X, X)
            np.testing.assert_array_equal(loaded_y, y)
            self.assertEqual(
                load_mert_embedding_metadata("train", feature_dir=feature_dir),
                {
                    "model_id": "m-a-p/MERT-v1-95M",
                    "model_revision": "test-revision",
                    "pooling": "mean_over_time_per_hidden_layer",
                },
            )

    def test_class_weights_follow_the_current_twelve_class_distribution(self) -> None:
        balanced = np.repeat(np.arange(len(TARGET_LABELS)), 2)
        self.assertIsNone(class_weight_vector(balanced))

        imbalanced = np.concatenate([balanced, np.asarray([0, 0])])
        weights = class_weight_vector(imbalanced)

        self.assertEqual(weights.shape, (len(TARGET_LABELS),))
        self.assertLess(weights[0], weights[1])

    def test_command_line_extraction_cannot_access_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with self.assertRaisesRegex(ValueError, "Test extraction is sealed"):
                extract_mert_splits(
                    splits=("test",),
                    data_root=root,
                    windows_csv=root / "windows.csv",
                    output_dir=root / "features",
                    batch_size=2,
                    model_id="test-model",
                    revision="test-revision",
                    device="cpu",
                )

    def test_extraction_refuses_to_overwrite_cached_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            feature_dir = root / "features"
            feature_dir.mkdir()
            (feature_dir / "train.npz").touch()

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                extract_mert_splits(
                    splits=("train",),
                    data_root=root,
                    windows_csv=root / "windows.csv",
                    output_dir=feature_dir,
                    batch_size=2,
                    model_id="test-model",
                    revision="test-revision",
                    device="cpu",
                )

    def test_training_refuses_to_overwrite_validation_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            (output_dir / "validation_summary.json").touch()

            with (
                patch.object(
                    sys,
                    "argv",
                    ["train_mert", "--output-dir", str(output_dir)],
                ),
                self.assertRaisesRegex(FileExistsError, "existing MERT validation run"),
            ):
                train_mert()

    def test_finalization_refuses_a_second_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            feature_dir = root / "features"
            output_dir = root / "outputs"
            feature_dir.mkdir()
            output_dir.mkdir()
            (feature_dir / "test.npz").touch()

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "finalize_mert",
                        "--feature-dir",
                        str(feature_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                self.assertRaisesRegex(FileExistsError, "another test access"),
            ):
                finalize_mert()

    def test_finalization_requires_frozen_validation_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            feature_dir = root / "features"
            output_dir = root / "outputs"
            feature_dir.mkdir()
            output_dir.mkdir()

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "finalize_mert",
                        "--feature-dir",
                        str(feature_dir),
                        "--output-dir",
                        str(output_dir),
                    ],
                ),
                self.assertRaisesRegex(FileNotFoundError, "Run train_mert"),
            ):
                finalize_mert()

    def test_finalization_writes_one_complete_test_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            feature_dir = root / "features"
            output_dir = root / "outputs"
            feature_dir.mkdir()
            output_dir.mkdir()
            labels = np.arange(len(TARGET_LABELS), dtype=np.int64)
            write_embedding_split(feature_dir / "train.npz", labels)
            write_embedding_split(feature_dir / "val.npz", labels)

            validation_search = output_dir / "validation_search.csv"
            validation_confusion = output_dir / "validation_confusion_matrix.csv"
            best_probe = output_dir / "best_probe.pt"
            validation_search.write_text("rank,learning_rate\n1,0.001\n")
            validation_confusion.write_text("actual\n")
            best_probe.write_bytes(b"validation checkpoint")
            metadata = {
                "model_id": "m-a-p/MERT-v1-95M",
                "model_revision": "test-revision",
                "pooling": "mean_over_time_per_hidden_layer",
            }
            validation_summary = {
                "config_fingerprint": json.loads(config_fingerprint_json()),
                "label_order": TARGET_LABELS,
                "test_evaluated": False,
                "selection_metric": "validation_macro_f1",
                "embedding_schema": metadata,
                "best_config": {
                    "learning_rate": 0.001,
                    "batch_size": 4,
                    "best_epoch": 2,
                    "seed": 0,
                },
                "input_files": {
                    split: {
                        "sha256": file_sha256(feature_dir / f"{split}.npz")
                    }
                    for split in ("train", "val")
                },
                "output_files": {
                    "validation_search": {
                        "sha256": file_sha256(validation_search)
                    },
                    "validation_confusion_matrix": {
                        "sha256": file_sha256(validation_confusion)
                    },
                    "model": {"sha256": file_sha256(best_probe)},
                },
            }
            (output_dir / "validation_summary.json").write_text(
                json.dumps(validation_summary)
            )

            class FakeModel:
                def state_dict(self):
                    return {}

                def layer_weights(self):
                    return [1 / MERT_NUM_LAYERS] * MERT_NUM_LAYERS

            fake_torch = types.ModuleType("torch")
            fake_torch.__version__ = "test"
            fake_torch.save = lambda _value, path: Path(path).write_bytes(
                b"final checkpoint"
            )
            fake_probe_module = types.ModuleType("instrument_robustness.mert_probe")
            fake_probe_module.MERTProbe = object

            def extract_test(**kwargs):
                self.assertTrue(kwargs["allow_test"])
                write_embedding_split(feature_dir / "test.npz", labels)
                return {"test": feature_dir / "test.npz"}

            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": fake_torch,
                        "instrument_robustness.mert_probe": fake_probe_module,
                    },
                ),
                patch(
                    "instrument_robustness.finalize_mert.choose_device",
                    return_value="cpu",
                ),
                patch(
                    "instrument_robustness.finalize_mert.train_fixed_epochs",
                    return_value=FakeModel(),
                ),
                patch(
                    "instrument_robustness.finalize_mert.extract_mert_splits",
                    side_effect=extract_test,
                ),
                patch(
                    "instrument_robustness.finalize_mert.predict",
                    return_value=labels,
                ),
                patch.object(
                    sys,
                    "argv",
                    [
                        "finalize_mert",
                        "--data-root",
                        str(root),
                        "--windows-csv",
                        str(root / "windows.csv"),
                        "--feature-dir",
                        str(feature_dir),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cpu",
                    ],
                ),
            ):
                finalize_mert()

            status = json.loads(
                (output_dir / "final_evaluation_status.json").read_text()
            )
            summary = json.loads((output_dir / "test_summary.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["test_access_count"], 1)
            self.assertEqual(status["test_evaluation_count"], 1)
            self.assertEqual(summary["model_fit_splits"], ["train", "val"])
            self.assertEqual(summary["test_metrics"]["macro_f1"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_candidate_training_supports_twelve_class_weights(self) -> None:
        import torch

        from instrument_robustness.mert_probe import MERTProbe

        labels = np.arange(len(TARGET_LABELS), dtype=np.int64)
        X = np.zeros(
            (len(labels), MERT_NUM_LAYERS, MERT_HIDDEN_SIZE),
            dtype=np.float32,
        )
        weights = np.ones(len(TARGET_LABELS), dtype=np.float32)
        _, _, metrics, best_epoch, epochs_run = train_candidate(
            X,
            labels,
            X,
            labels,
            learning_rate=0.001,
            batch_size=len(labels),
            max_epochs=1,
            patience=1,
            seed=0,
            device="cpu",
            torch=torch,
            MERTProbe=MERTProbe,
            class_weights=weights,
        )

        self.assertEqual(best_epoch, 1)
        self.assertEqual(epochs_run, 1)
        self.assertIn("macro_f1", metrics)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_probe_returns_one_logit_per_class_and_normalized_layer_weights(self) -> None:
        import torch

        from instrument_robustness.mert_probe import MERTProbe

        model = MERTProbe(len(TARGET_LABELS))
        embeddings = torch.zeros(2, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)

        logits = model(embeddings)

        self.assertEqual(tuple(logits.shape), (2, len(TARGET_LABELS)))
        self.assertAlmostEqual(sum(model.layer_weights()), 1.0, places=6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_saved_probe_loader_checks_current_dataset_identity(self) -> None:
        import torch

        from instrument_robustness.config import config_fingerprint
        from instrument_robustness.mert_probe import MERTProbe, load_mert_probe

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "probe.pt"
            model = MERTProbe(len(TARGET_LABELS))
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": len(TARGET_LABELS),
                    "label_order": TARGET_LABELS,
                    "config_fingerprint": config_fingerprint(),
                },
                path,
            )

            loaded, checkpoint = load_mert_probe(path)

            self.assertEqual(checkpoint["label_order"], TARGET_LABELS)
            self.assertEqual(
                tuple(loaded.classifier.weight.shape),
                (len(TARGET_LABELS), MERT_HIDDEN_SIZE),
            )


if __name__ == "__main__":
    unittest.main()
