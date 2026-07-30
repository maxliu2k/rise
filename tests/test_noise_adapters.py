"""Tests for the CNN/CRNN/AST noise adapters and the SNR pilot.

Split from test_noise.py because these cover the model-side of the sweep rather than the generator:
the representation path a noisy waveform travels, the ensemble score encoding, and the grid pilot.

The log-mel tests are the important ones. A noise adapter that quietly read the cached CLEAN feature
array would report the clean score under a noisy label -- a wrong number with no crash and no
warning. `test_matches_the_cached_clean_features` pins that down against the real Step-7 output when
it is present.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from instrument_robustness.config import (
    FEATURES,
    N_FRAMES,
    N_MELS,
    STATS_NPZ,
    TARGET_LABELS,
    WINDOWS_CSV,
    config_fingerprint,
)
from instrument_robustness.ensemble_scores import combiner_scores
from instrument_robustness.logmel_input import (
    cnn_batch_from_waveforms,
    cnn_input_from_waveform,
    load_logmel_statistics,
)
from instrument_robustness.snr_pilot import (
    CANDIDATE_SNRS,
    USABLE_RETENTION_HIGH,
    USABLE_RETENTION_LOW,
    recommend,
    resolve_mert_embedding_schema,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
CNN_TEST_NPZ = FEATURES / "cnn" / "val.npz"
REAL_BUILD = STATS_NPZ.is_file() and CNN_TEST_NPZ.is_file() and WINDOWS_CSV.is_file()


class LogMelInputTests(unittest.TestCase):
    def test_shape_and_dtype(self) -> None:
        mean = np.zeros((N_MELS, 1), dtype=np.float32)
        std = np.ones((N_MELS, 1), dtype=np.float32)
        waveform = np.random.default_rng(0).standard_normal(66150).astype(np.float32)
        single = cnn_input_from_waveform(waveform, mean, std)
        self.assertEqual(single.shape, (1, N_MELS, N_FRAMES))
        self.assertEqual(single.dtype, np.float32)
        batch = cnn_batch_from_waveforms([waveform, waveform], mean, std)
        self.assertEqual(batch.shape, (2, 1, N_MELS, N_FRAMES))
        np.testing.assert_allclose(batch[0], batch[1])

    def test_empty_batch_keeps_the_contract_shape(self) -> None:
        mean = np.zeros((N_MELS, 1), dtype=np.float32)
        std = np.ones((N_MELS, 1), dtype=np.float32)
        self.assertEqual(
            cnn_batch_from_waveforms([], mean, std).shape, (0, 1, N_MELS, N_FRAMES)
        )

    def test_standardization_is_applied_per_mel_bin(self) -> None:
        waveform = np.random.default_rng(1).standard_normal(66150).astype(np.float32)
        zero_mean = np.zeros((N_MELS, 1), dtype=np.float32)
        unit_std = np.ones((N_MELS, 1), dtype=np.float32)
        raw = cnn_input_from_waveform(waveform, zero_mean, unit_std)
        # A distinct offset per bin must land on that bin only, which is what "per mel bin" means.
        offsets = np.arange(N_MELS, dtype=np.float32)[:, None]
        shifted = cnn_input_from_waveform(waveform, offsets, unit_std)
        np.testing.assert_allclose(raw[0] - offsets, shifted[0], rtol=1e-5, atol=1e-4)

    def test_rejects_a_wrong_length_window(self) -> None:
        mean = np.zeros((N_MELS, 1), dtype=np.float32)
        std = np.ones((N_MELS, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            cnn_input_from_waveform(np.zeros(1000, dtype=np.float32), mean, std)

    @unittest.skipUnless(REAL_BUILD, "needs a generated build (norm_stats + features/cnn)")
    def test_statistics_bundle_loads_and_is_train_only(self) -> None:
        mean, std = load_logmel_statistics(STATS_NPZ)
        self.assertEqual(mean.shape, (N_MELS, 1))
        self.assertEqual(std.shape, (N_MELS, 1))
        self.assertTrue(np.all(std > 0))

    @unittest.skipUnless(REAL_BUILD, "needs a generated build (norm_stats + features/cnn)")
    def test_matches_the_cached_clean_features(self) -> None:
        """Recomputing a clean window from audio must reproduce the Step-7 array exactly.

        This is the invariant that makes noisy features comparable to clean ones: the adapter and
        Step 7 must be the same transform. If this drifts, every noise result silently compares two
        different representations.
        """
        import pandas as pd

        from instrument_robustness.noise_sweep import load_clean

        windows = pd.read_csv(WINDOWS_CSV)
        windows = windows.loc[windows["split"] == "val"].reset_index(drop=True)
        with np.load(CNN_TEST_NPZ, allow_pickle=True) as data:
            cached = data["X"]
        mean, std = load_logmel_statistics(STATS_NPZ)
        for row in range(min(3, len(windows))):
            waveform = load_clean(str(windows.loc[row, "window_path"]))
            rebuilt = cnn_input_from_waveform(waveform, mean, std)
            # step7 stores (N, mels, frames, 1); the adapter emits (1, mels, frames).
            expected = np.transpose(cached[row], (2, 0, 1))
            np.testing.assert_allclose(rebuilt, expected, rtol=1e-4, atol=1e-3)


def reference_hard_vote(per_seed_probs: np.ndarray) -> np.ndarray:
    """The obvious implementation of "most votes wins, ties broken by summed probability".

    Written out longhand, independently of the lexicographic encoding under test, so the test
    checks the encoding against the SPECIFICATION rather than against itself.
    """
    n_seeds, n_rows, n_classes = per_seed_probs.shape
    votes = per_seed_probs.argmax(axis=2)
    summed = per_seed_probs.sum(axis=0)
    decisions = np.empty(n_rows, dtype=int)
    for row in range(n_rows):
        counts = np.bincount(votes[:, row], minlength=n_classes)
        best = counts.max()
        contenders = [c for c in range(n_classes) if counts[c] == best]
        decisions[row] = max(contenders, key=lambda c: (summed[row, c], -c))
    return decisions


class CombinerScoreTests(unittest.TestCase):
    """The shared runner takes argmax of an (N, n_classes) score array and rejects non-finite
    values, so the ensemble combiners have to be expressed as finite scores that decide identically.
    """

    def _probs(self, seed: int = 0, n_seeds: int = 5, n: int = 32) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.random((n_seeds, n, 12))
        return raw / raw.sum(axis=2, keepdims=True)

    def test_soft_vote_scores_are_the_mean_probability(self) -> None:
        probs = self._probs()
        scores = combiner_scores(probs, "soft_vote")
        self.assertEqual(scores.shape, probs.shape[1:])
        self.assertTrue(np.all(np.isfinite(scores)))
        np.testing.assert_allclose(scores, probs.mean(axis=0))
        np.testing.assert_array_equal(
            scores.argmax(axis=1), probs.mean(axis=0).argmax(axis=1)
        )

    def test_hard_vote_scores_are_finite_and_match_the_specification(self) -> None:
        for seed in range(8):
            for n_seeds in (2, 3, 5):
                probs = self._probs(seed=seed, n_seeds=n_seeds)
                scores = combiner_scores(probs, "hard_vote")
                self.assertTrue(
                    np.all(np.isfinite(scores)), "hard_vote scores must not use -inf"
                )
                np.testing.assert_array_equal(
                    scores.argmax(axis=1), reference_hard_vote(probs)
                )

    def test_vote_count_outranks_confidence(self) -> None:
        """The load-bearing property of the encoding: no amount of confidence lets a class with
        fewer votes win. A naive count + summed_probability would break exactly here."""
        probs = np.zeros((3, 1, 12))
        # Class 5 wins 2 of 3 votes, but only barely each time.
        probs[0, 0, 5], probs[0, 0, 2] = 0.34, 0.33
        probs[1, 0, 5], probs[1, 0, 2] = 0.34, 0.33
        # Class 2 takes the third vote with near-total confidence.
        probs[2, 0, 2], probs[2, 0, 5] = 0.99, 0.01
        probs = probs / probs.sum(axis=2, keepdims=True)
        scores = combiner_scores(probs, "hard_vote")
        self.assertEqual(int(scores.argmax(axis=1)[0]), 5)
        np.testing.assert_array_equal(
            scores.argmax(axis=1), reference_hard_vote(probs)
        )

    def test_hard_vote_breaks_a_tie_by_summed_probability(self) -> None:
        # Two members, two different argmaxes: a 1-1 tie the summed probability must settle.
        probs = np.zeros((2, 1, 12))
        probs[0, 0, 3] = 0.9
        probs[0, 0, 7] = 0.1
        probs[1, 0, 7] = 0.6
        probs[1, 0, 3] = 0.4
        scores = combiner_scores(probs, "hard_vote")
        self.assertEqual(int(scores.argmax(axis=1)[0]), 3)

    def test_unknown_combiner_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            combiner_scores(self._probs(), "plurality")

    def test_bad_shapes_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            combiner_scores(np.zeros((4, 12)), "soft_vote")
        with self.assertRaises(ValueError):
            combiner_scores(np.zeros((0, 4, 12)), "soft_vote")

    @unittest.skipUnless(HAS_TORCH, "cross-check needs cnn_model, which imports torch")
    def test_agrees_with_cnn_model_combiners(self) -> None:
        from instrument_robustness.cnn_model import hard_vote, soft_vote

        probs = self._probs(seed=11)
        np.testing.assert_array_equal(
            combiner_scores(probs, "soft_vote").argmax(axis=1), soft_vote(probs)
        )
        np.testing.assert_array_equal(
            combiner_scores(probs, "hard_vote").argmax(axis=1), hard_vote(probs)
        )


class SnrPilotRecommendationTests(unittest.TestCase):
    def _curve(self, retentions: dict[int, float], noise_type: str = "white") -> list[dict]:
        return [
            {
                "noise_type": noise_type,
                "snr_db": snr,
                "macro_f1": retention,
                "accuracy": retention,
                "retention": retention,
                "macro_f1_drop": 1.0 - retention,
                "in_current_grid": False,
            }
            for snr, retention in retentions.items()
        ]

    def test_band_is_where_the_model_is_neither_intact_nor_floored(self) -> None:
        rows = self._curve({60: 0.99, 50: 0.90, 40: 0.55, 30: 0.25, 20: 0.05, 10: 0.01})
        result = recommend(rows, [60, 50, 40, 30, 20, 10])
        usable = result["per_noise_type"]["white"]["usable_snrs"]
        self.assertEqual(usable, [50, 40, 30])
        for snr in usable:
            retention = next(r["retention"] for r in rows if r["snr_db"] == snr)
            self.assertGreaterEqual(retention, USABLE_RETENTION_LOW)
            self.assertLessEqual(retention, USABLE_RETENTION_HIGH)
        # Shoulders: one candidate step out on each side, so the curve's knees are visible.
        self.assertEqual(
            result["per_noise_type"]["white"]["band_with_shoulders"], [60, 50, 40, 30, 20]
        )

    def test_default_candidates_cover_the_current_grid(self) -> None:
        from instrument_robustness.config import SNRS

        self.assertTrue(set(SNRS).issubset(CANDIDATE_SNRS))

    def test_a_custom_grid_does_not_raise(self) -> None:
        """Regression: `recommend` used to widen against the module default, so any level passed
        via --snrs that was not in CANDIDATE_SNRS raised ValueError from list.index."""
        rows = self._curve({70: 0.99, 55: 0.80, 45: 0.30, 33: 0.02})
        result = recommend(rows, [70, 55, 45, 33])
        self.assertEqual(result["per_noise_type"]["white"]["usable_snrs"], [55, 45])
        self.assertEqual(
            result["per_noise_type"]["white"]["band_with_shoulders"], [70, 55, 45, 33]
        )

    def test_an_all_floor_curve_explains_itself_instead_of_recommending_nothing(self) -> None:
        rows = self._curve({20: 0.05, 10: 0.02, 0: 0.01})
        result = recommend(rows, [20, 10, 0])
        detail = result["per_noise_type"]["white"]
        self.assertEqual(detail["usable_snrs"], [])
        self.assertIn("widen", detail["note"])
        self.assertEqual(result["suggested_shared_grid"], [])

    def test_shared_grid_unions_the_noise_types(self) -> None:
        rows = self._curve({60: 0.99, 50: 0.90, 40: 0.55, 30: 0.10}, "white")
        rows += self._curve({60: 0.99, 50: 0.99, 40: 0.85, 30: 0.40}, "natural")
        result = recommend(rows, [60, 50, 40, 30])
        self.assertEqual(result["suggested_shared_grid"], [60, 50, 40, 30])
        self.assertIn("white", result["per_noise_type"])
        self.assertIn("natural", result["per_noise_type"])


class SnrPilotSplitTests(unittest.TestCase):
    def test_subsampling_preserves_label_without_groupby_apply(self) -> None:
        """Regression for pandas 3, where GroupBy.apply drops the grouping column by default."""
        import pandas as pd

        from instrument_robustness.snr_pilot import validation_windows

        with tempfile.TemporaryDirectory() as temporary_dir:
            windows_csv = Path(temporary_dir) / "windows.csv"
            pd.DataFrame(
                [
                    {
                        "window_path": f"{label}_{index}.wav",
                        "label": label,
                        "split": "val",
                    }
                    for label in TARGET_LABELS
                    for index in range(2)
                ]
            ).to_csv(windows_csv, index=False)
            with (
                mock.patch(
                    "instrument_robustness.snr_pilot.WINDOWS_CSV",
                    windows_csv,
                ),
                mock.patch(
                    "instrument_robustness.snr_pilot.assert_artifact_fingerprint"
                ),
                mock.patch(
                    "pandas.core.groupby.generic.DataFrameGroupBy.apply",
                    side_effect=AssertionError("GroupBy.apply is pandas-version-sensitive"),
                ),
            ):
                frame = validation_windows(limit=len(TARGET_LABELS), seed=0)

        self.assertEqual(set(frame["label"]), set(TARGET_LABELS))
        self.assertEqual(len(frame), len(TARGET_LABELS))

    @unittest.skipUnless(REAL_BUILD, "needs a generated build (windows.csv)")
    def test_pilot_reads_validation_only(self) -> None:
        """The pilot must never touch test: choosing a grid is a design decision, and making it
        against test would turn test into a second validation set."""
        import pandas as pd

        from instrument_robustness.snr_pilot import validation_windows

        frame = validation_windows(limit=48, seed=0)
        self.assertGreater(len(frame), 0)
        self.assertEqual(set(frame["split"]), {"val"})
        everything = pd.read_csv(WINDOWS_CSV)
        test_paths = set(everything.loc[everything["split"] == "test", "window_path"])
        self.assertFalse(set(frame["window_path"]) & test_paths)


class MertPilotCheckpointTests(unittest.TestCase):
    SCHEMA = {
        "model_id": "m-a-p/MERT-v1-95M",
        "model_revision": "immutable-test-revision",
    }

    def test_uses_schema_stored_in_new_checkpoint(self) -> None:
        resolved = resolve_mert_embedding_schema(
            Path("best_probe.pt"),
            {"embedding_schema": self.SCHEMA},
        )
        self.assertEqual(resolved, self.SCHEMA)

    def test_supports_existing_checkpoint_via_hash_verified_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = Path(temporary_dir) / "best_probe.pt"
            model_path.write_bytes(b"existing SCC validation checkpoint")
            summary = {
                "config_fingerprint": config_fingerprint(),
                "label_order": TARGET_LABELS,
                "test_evaluated": False,
                "embedding_schema": self.SCHEMA,
                "output_files": {
                    "model": {
                        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()
                    }
                },
            }
            model_path.with_name("validation_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            resolved = resolve_mert_embedding_schema(model_path, {})

        self.assertEqual(resolved, self.SCHEMA)

    def test_refuses_a_checkpoint_not_named_by_the_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = Path(temporary_dir) / "best_probe.pt"
            model_path.write_bytes(b"different checkpoint")
            summary = {
                "config_fingerprint": config_fingerprint(),
                "label_order": TARGET_LABELS,
                "test_evaluated": False,
                "embedding_schema": self.SCHEMA,
                "output_files": {"model": {"sha256": "0" * 64}},
            }
            model_path.with_name("validation_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "does not match"):
                resolve_mert_embedding_schema(model_path, {})


class CleanSummaryContractTests(unittest.TestCase):
    """finalize_cnn and train_ast must emit what load_official_summary requires.

    Without label_order, test_examples and test_metrics.macro_f1 the clean-parity gate has nothing
    to compare against, and the CNN/CRNN/AST branches cannot enter the sweep at all.
    """

    def test_load_official_summary_accepts_a_contract_shaped_record(self) -> None:
        from instrument_robustness.config import TARGET_LABELS, config_fingerprint
        from instrument_robustness.noise_eval_common import load_official_summary

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "test_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "architecture": "MediumCNN",
                        "seeds": [42, 43],
                        "combiner": "soft_vote",
                        "label_order": list(TARGET_LABELS),
                        "config_fingerprint": config_fingerprint(),
                        "test_examples": 1255,
                        "test_metrics": {
                            "accuracy": 0.9,
                            "balanced_accuracy": 0.9,
                            "macro_f1": 0.89,
                            "mcc": 0.88,
                        },
                    }
                )
            )
            summary = load_official_summary(path)
        self.assertEqual(summary["test_examples"], 1255)
        self.assertAlmostEqual(summary["test_metrics"]["macro_f1"], 0.89)

    def test_a_summary_without_macro_f1_cannot_gate_the_sweep(self) -> None:
        """The pre-fix finalize_cnn shape: balanced accuracy and MCC but no macro-F1."""
        from instrument_robustness.config import TARGET_LABELS, config_fingerprint
        from instrument_robustness.noise_eval_common import load_official_summary

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "test_summary.json"
            path.write_text(
                json.dumps(
                    {
                        "architecture": "MediumCNN",
                        "label_order": list(TARGET_LABELS),
                        "config_fingerprint": config_fingerprint(),
                        "n_test": 1255,
                        "ensemble_balanced_accuracy": 0.9,
                        "ensemble_mcc": 0.88,
                    }
                )
            )
            summary = load_official_summary(path)
            with self.assertRaises(KeyError):
                _ = summary["test_metrics"]["macro_f1"]


if __name__ == "__main__":
    unittest.main()
