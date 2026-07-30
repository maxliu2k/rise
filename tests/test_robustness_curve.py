"""Tests for curve summarisation (audit item 18) and FDR control (audit item 19).

Both are written against the reporting error they prevent, not against golden numbers.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_robustness.config import NOISE_TYPES, SNRS
from instrument_robustness.robustness_curve import (
    CurvePoint,
    benjamini_hochberg,
    curve_from_summary,
    mean_retention,
    robustness_auc,
    snr_at_retention,
    summarise_sweep,
)

# The measured SVM white-noise curve from snr_pilot, used as a realistic shape.
SVM_WHITE = {60: 0.9515, 50: 0.8497, 40: 0.5376, 30: 0.2599, 20: 0.0931, 10: 0.0366, 0: 0.0132}
SVM_CLEAN = 0.9650


def svm_points(snrs=None) -> list[CurvePoint]:
    snrs = SNRS if snrs is None else snrs
    return [CurvePoint(float(s), SVM_WHITE[s]) for s in snrs]


class Item18SpacingTests(unittest.TestCase):
    def test_auc_is_insensitive_to_how_densely_the_curve_was_sampled(self) -> None:
        """The reporting error item 18 is about. Adding points where a model does well must not
        improve its summary score -- the model has not changed."""
        base = svm_points()
        denser = base + [CurvePoint(55.0, 0.9112), CurvePoint(45.0, 0.6779)]
        auc_before = robustness_auc(base, SVM_CLEAN)["auc"]
        auc_after = robustness_auc(denser, SVM_CLEAN)["auc"]
        mean_before = mean_retention(base, SVM_CLEAN)
        mean_after = mean_retention(denser, SVM_CLEAN)
        self.assertLess(abs(auc_after - auc_before), 0.01)
        # The unweighted mean moves by an order of magnitude more.
        self.assertGreater(abs(mean_after - mean_before), 0.05)
        self.assertGreater(
            abs(mean_after - mean_before), 10 * abs(auc_after - auc_before)
        )

    def test_auc_of_a_flat_curve_equals_its_retention(self) -> None:
        for retention in (1.0, 0.5, 0.25):
            points = [CurvePoint(float(s), SVM_CLEAN * retention) for s in SNRS]
            self.assertAlmostEqual(
                robustness_auc(points, SVM_CLEAN)["auc"], retention, places=10
            )

    def test_auc_is_normalised_by_span_not_point_count(self) -> None:
        coarse = [CurvePoint(0.0, SVM_CLEAN), CurvePoint(60.0, SVM_CLEAN)]
        record = robustness_auc(coarse, SVM_CLEAN)
        self.assertAlmostEqual(record["auc"], 1.0, places=10)
        self.assertEqual(record["span_db"], 60.0)
        self.assertEqual(record["n_points"], 2)

    def test_unsorted_input_is_handled(self) -> None:
        forward = robustness_auc(svm_points(), SVM_CLEAN)["auc"]
        backward = robustness_auc(list(reversed(svm_points())), SVM_CLEAN)["auc"]
        self.assertAlmostEqual(forward, backward, places=12)

    def test_snr_range_restricts_the_integration(self) -> None:
        full = robustness_auc(svm_points(), SVM_CLEAN)
        top = robustness_auc(svm_points(), SVM_CLEAN, snr_range=(40.0, 60.0))
        self.assertEqual(top["span_db"], 20.0)
        # The easy end of the curve must score higher than the whole range.
        self.assertGreater(top["auc"], full["auc"])

    def test_degenerate_inputs_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            robustness_auc([CurvePoint(0.0, 0.5)], SVM_CLEAN)
        with self.assertRaises(ValueError):
            robustness_auc(svm_points(), 0.0)
        with self.assertRaises(ValueError):
            robustness_auc([CurvePoint(10.0, 0.5), CurvePoint(10.0, 0.4)], SVM_CLEAN)
        with self.assertRaises(ValueError):
            robustness_auc(svm_points(), SVM_CLEAN, snr_range=(58.0, 60.0))

    def test_snr_at_retention_brackets_the_measured_points(self) -> None:
        half = snr_at_retention(svm_points(), SVM_CLEAN, target=0.5)
        self.assertIsNotNone(half)
        # Retention crosses 0.5 between 30 dB (0.269) and 40 dB (0.557).
        self.assertGreater(half, 30.0)
        self.assertLess(half, 40.0)

    def test_snr_at_retention_returns_none_when_never_crossed(self) -> None:
        flat = [CurvePoint(float(s), SVM_CLEAN) for s in SNRS]
        self.assertIsNone(snr_at_retention(flat, SVM_CLEAN, target=0.5))
        floored = [CurvePoint(float(s), 0.001) for s in SNRS]
        self.assertIsNone(snr_at_retention(floored, SVM_CLEAN, target=0.5))

    def test_an_impossible_target_is_refused(self) -> None:
        for target in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                snr_at_retention(svm_points(), SVM_CLEAN, target=target)


class Item19FdrTests(unittest.TestCase):
    def test_correction_reduces_rejections_on_a_mostly_null_family(self) -> None:
        """The error item 19 is about: at alpha=0.05, ~1 in 20 null comparisons looks significant."""
        rng = np.random.default_rng(0)
        p_values = {f"cond{i}": 1e-5 for i in range(3)}
        p_values.update(
            {f"null{i}": float(rng.uniform(0.01, 0.99)) for i in range(30)}
        )
        uncorrected = sum(1 for value in p_values.values() if value < 0.05)
        corrected = benjamini_hochberg(
            p_values, alpha=0.05, family="33 comparisons, synthetic"
        )
        self.assertLess(corrected["n_rejected"], uncorrected)
        self.assertGreaterEqual(corrected["n_rejected"], 3)

    def test_all_true_effects_survive(self) -> None:
        p_values = {f"c{i}": 1e-8 for i in range(10)}
        result = benjamini_hochberg(p_values, alpha=0.05, family="10 strong effects")
        self.assertEqual(result["n_rejected"], 10)
        self.assertTrue(all(record["rejected"] for record in result["results"]))

    def test_all_null_rejects_nothing(self) -> None:
        p_values = {f"c{i}": 0.5 + i * 0.01 for i in range(20)}
        result = benjamini_hochberg(p_values, alpha=0.05, family="20 nulls")
        self.assertEqual(result["n_rejected"], 0)

    def test_q_values_are_monotone_in_p_and_bounded(self) -> None:
        rng = np.random.default_rng(1)
        p_values = {f"c{i}": float(rng.uniform(0, 1)) for i in range(50)}
        result = benjamini_hochberg(p_values, alpha=0.05, family="50 random")
        q_values = [record["q_value"] for record in result["results"]]
        # results come back ordered by ascending p, so q must be non-decreasing
        self.assertEqual(q_values, sorted(q_values))
        for record in result["results"]:
            self.assertGreaterEqual(record["q_value"], record["p_value"] - 1e-12)
            self.assertLessEqual(record["q_value"], 1.0)

    def test_is_less_conservative_than_bonferroni(self) -> None:
        """BH controls FDR, not family-wise error; on correlated positives it must reject more."""
        p_values = {f"c{i}": 0.001 * (i + 1) for i in range(20)}
        result = benjamini_hochberg(p_values, alpha=0.05, family="20 graded")
        bonferroni = sum(1 for v in p_values.values() if v < 0.05 / len(p_values))
        self.assertGreaterEqual(result["n_rejected"], bonferroni)

    def test_family_must_be_named(self) -> None:
        """FDR without a stated family is uncheckable, so the label is mandatory."""
        with self.assertRaises(ValueError):
            benjamini_hochberg({"a": 0.01}, alpha=0.05, family="")

    def test_invalid_inputs_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            benjamini_hochberg({}, alpha=0.05, family="empty")
        with self.assertRaises(ValueError):
            benjamini_hochberg({"a": 1.5}, alpha=0.05, family="bad p")
        with self.assertRaises(ValueError):
            benjamini_hochberg({"a": 0.1}, alpha=0.0, family="bad alpha")


class SweepSummaryTests(unittest.TestCase):
    def _summary_csv(self, directory: Path) -> Path:
        rows = [
            {
                "noise_type": "clean",
                "snr_db": None,
                "replicate": None,
                "condition": "clean",
                "accuracy": 0.97,
                "macro_f1": SVM_CLEAN,
            }
        ]
        for noise_type in NOISE_TYPES:
            for snr in SNRS:
                rows.append(
                    {
                        "noise_type": noise_type,
                        "snr_db": snr,
                        "replicate": 0,
                        "condition": f"{noise_type}_{snr}",
                        "accuracy": SVM_WHITE[snr],
                        "macro_f1": SVM_WHITE[snr],
                    }
                )
        path = directory / "noise_sweep_summary.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_summarise_sweep_reads_the_real_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = self._summary_csv(Path(temporary_dir))
            result = summarise_sweep(path)
        self.assertAlmostEqual(result["clean_macro_f1"], SVM_CLEAN)
        self.assertEqual(set(result["curves"]), set(NOISE_TYPES))
        for record in result["curves"].values():
            self.assertEqual(record["n_points"], len(SNRS))
            self.assertGreater(record["auc"], 0.0)

    def test_the_clean_row_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "s.csv"
            pd.DataFrame(
                [{"noise_type": "white", "snr_db": 20, "macro_f1": 0.1}]
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                summarise_sweep(path)

    def test_curve_extraction_skips_clean_and_other_noise_types(self) -> None:
        rows = [
            {"noise_type": "clean", "snr_db": None, "macro_f1": 0.9},
            {"noise_type": "white", "snr_db": 20, "macro_f1": 0.5},
            {"noise_type": "white", "snr_db": 0, "macro_f1": 0.1},
            {"noise_type": "natural", "snr_db": 20, "macro_f1": 0.8},
        ]
        points = curve_from_summary(rows, noise_type="white")
        self.assertEqual([p.snr_db for p in points], [0.0, 20.0])

    def test_too_few_points_is_refused(self) -> None:
        rows = [{"noise_type": "white", "snr_db": 20, "macro_f1": 0.5}]
        with self.assertRaises(ValueError):
            curve_from_summary(rows, noise_type="white")

    def test_replicate_selection(self) -> None:
        rows = [
            {"noise_type": "white", "snr_db": 20, "replicate": 0, "macro_f1": 0.5},
            {"noise_type": "white", "snr_db": 0, "replicate": 0, "macro_f1": 0.1},
            {"noise_type": "white", "snr_db": 20, "replicate": 1, "macro_f1": 0.6},
            {"noise_type": "white", "snr_db": 0, "replicate": 1, "macro_f1": 0.2},
        ]
        first = curve_from_summary(rows, noise_type="white", replicate=0)
        second = curve_from_summary(rows, noise_type="white", replicate=1)
        self.assertEqual([p.macro_f1 for p in first], [0.1, 0.5])
        self.assertEqual([p.macro_f1 for p in second], [0.2, 0.6])


class NoiseSourceClusterTests(unittest.TestCase):
    """Audit item 8: predictions must carry the noise recording so it can be clustered on."""

    def test_noise_source_is_an_allowed_cluster_unit(self) -> None:
        from instrument_robustness.noise_stats import CLUSTER_COLUMNS

        self.assertIn("noise_source", CLUSTER_COLUMNS)
        self.assertEqual(CLUSTER_COLUMNS[0], "pitch_group", "default must stay conservative")

    def test_clean_condition_reports_no_source(self) -> None:
        from instrument_robustness.noise_eval_common import (
            NoiseCondition,
            noise_source_lookup,
        )

        self.assertEqual(
            noise_source_lookup(NoiseCondition("clean", "clean", None, None)), {}
        )

    def test_lookup_selects_the_right_condition_and_replicate(self) -> None:
        from instrument_robustness.noise_eval_common import (
            NoiseCondition,
            noise_source_lookup,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            noisy = Path(temporary_dir)
            pd.DataFrame(
                [
                    {"window_id": "w0", "noise_type": "white", "snr_db": 20,
                     "replicate": 0, "noise_source": "generated_gaussian"},
                    {"window_id": "w0", "noise_type": "natural", "snr_db": 20,
                     "replicate": 0, "noise_source": "audio/dog.wav"},
                    {"window_id": "w0", "noise_type": "natural", "snr_db": 20,
                     "replicate": 1, "noise_source": "audio/rain.wav"},
                ]
            ).to_csv(noisy / "noise_provenance.csv", index=False)
            first = noise_source_lookup(
                NoiseCondition("natural_20", "natural", 20, 0), noisy_dir=noisy
            )
            second = noise_source_lookup(
                NoiseCondition("natural_20_r1", "natural", 20, 1), noisy_dir=noisy
            )
        self.assertEqual(first, {"w0": "audio/dog.wav"})
        self.assertEqual(second, {"w0": "audio/rain.wav"})

    def test_missing_provenance_degrades_instead_of_raising(self) -> None:
        from instrument_robustness.noise_eval_common import (
            NoiseCondition,
            noise_source_lookup,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            self.assertEqual(
                noise_source_lookup(
                    NoiseCondition("white_20", "white", 20, 0),
                    noisy_dir=Path(temporary_dir),
                ),
                {},
            )


if __name__ == "__main__":
    unittest.main()


class Item16AudioInventoryTests(unittest.TestCase):
    """Clean fingerprints hash the CSV, not the audio. A window edited in place keeps its path, its
    row and the CSV hash, so every loader accepts it and every downstream number changes silently."""

    def _build(self, root: Path, *, contents: list[bytes]) -> Path:
        from instrument_robustness.config import write_artifact_fingerprint

        rows = []
        for index, blob in enumerate(contents):
            relative = f"work/windows/w{index}.wav"
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(blob)
            rows.append({"window_path": relative, "label": "flute", "split": "test"})
        pipeline = root / "pipeline"
        pipeline.mkdir(parents=True, exist_ok=True)
        windows_csv = pipeline / "windows.csv"
        pd.DataFrame(rows).to_csv(windows_csv, index=False)
        write_artifact_fingerprint(windows_csv, "step5_normalize")
        return windows_csv

    def test_records_then_verifies(self) -> None:
        from instrument_robustness.audio_inventory import (
            record_window_audio_inventory,
            verify_window_audio,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv = self._build(root, contents=[b"aaa", b"bbb"])
            recorded = record_window_audio_inventory(windows_csv=csv, data_root=root)
            self.assertEqual(recorded["file_count"], 2)
            result = verify_window_audio(windows_csv=csv, data_root=root)
            self.assertEqual(result["status"], "match")

    def test_an_edited_wav_is_detected_even_though_the_csv_is_untouched(self) -> None:
        """The exact gap item 16 describes."""
        from instrument_robustness.audio_inventory import (
            record_window_audio_inventory,
            verify_window_audio,
        )
        from instrument_robustness.config import assert_artifact_fingerprint

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv = self._build(root, contents=[b"aaa", b"bbb"])
            record_window_audio_inventory(windows_csv=csv, data_root=root)
            (root / "work/windows/w1.wav").write_bytes(b"XXX")  # same length, same path
            # The pre-existing CSV check still passes -- which is the whole problem.
            assert_artifact_fingerprint(csv, "step5_normalize")
            with self.assertRaises(ValueError) as caught:
                verify_window_audio(windows_csv=csv, data_root=root)
        self.assertIn("audio", str(caught.exception).lower())

    def test_a_renamed_file_changes_the_digest(self) -> None:
        """The path is part of the hashed record, so a rename is a change even at identical bytes."""
        from instrument_robustness.audio_inventory import window_audio_inventory

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = self._build(root, contents=[b"aaa", b"bbb"])
            before = window_audio_inventory(windows_csv=first, data_root=root)["sha256"]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            second = self._build(root, contents=[b"bbb", b"aaa"])  # bytes swapped between paths
            after = window_audio_inventory(windows_csv=second, data_root=root)["sha256"]
        self.assertNotEqual(before, after)

    def test_a_missing_file_is_reported_not_silently_hashed_around(self) -> None:
        from instrument_robustness.audio_inventory import (
            record_window_audio_inventory,
            window_audio_inventory,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv = self._build(root, contents=[b"aaa", b"bbb"])
            (root / "work/windows/w0.wav").unlink()
            result = window_audio_inventory(windows_csv=csv, data_root=root)
            self.assertEqual(result["missing"], ["work/windows/w0.wav"])
            self.assertEqual(result["file_count"], 1)
            with self.assertRaises(ValueError):
                record_window_audio_inventory(windows_csv=csv, data_root=root)

    def test_an_unrecorded_build_reports_rather_than_failing_by_default(self) -> None:
        """Refusing to load every artifact built before this check existed would be worse than the
        gap it closes, so absence is a status and only an error on request."""
        from instrument_robustness.audio_inventory import verify_window_audio

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            csv = self._build(root, contents=[b"aaa"])
            self.assertEqual(
                verify_window_audio(windows_csv=csv, data_root=root)["status"],
                "not_recorded",
            )
            with self.assertRaises(ValueError):
                verify_window_audio(windows_csv=csv, data_root=root, required=True)
