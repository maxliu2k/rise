from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_robustness.config import SR, TARGET_LABELS
from instrument_robustness.mert_data import (
    MERT_HIDDEN_SIZE,
    MERT_NUM_LAYERS,
    load_mert_embedding_metadata,
    load_mert_embeddings,
    load_mert_examples,
)
from instrument_robustness.pretrained_extractors import mert_batch_input


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

            examples = load_mert_examples(
                "train", windows_csv=windows_csv, data_root=root
            )

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].label, "violin")
            self.assertEqual(examples[0].target, 0)
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

    def test_embedding_loader_validates_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            feature_dir = Path(temporary_dir)
            X = np.zeros(
                (9, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE),
                dtype=np.float32,
            )
            y = np.arange(9, dtype=np.int64)
            np.savez(
                feature_dir / "train.npz",
                X=X,
                y=y,
                label_names=np.asarray(TARGET_LABELS),
                model_id=np.asarray("m-a-p/MERT-v1-95M"),
                model_revision=np.asarray("test-revision"),
                pooling=np.asarray("mean_over_time_per_hidden_layer"),
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
    def test_probe_returns_nine_logits_and_normalized_layer_weights(self) -> None:
        import torch

        from instrument_robustness.mert_probe import MERTProbe

        model = MERTProbe(len(TARGET_LABELS))
        embeddings = torch.zeros(2, MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)

        logits = model(embeddings)

        self.assertEqual(tuple(logits.shape), (2, 9))
        self.assertAlmostEqual(sum(model.layer_weights()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
