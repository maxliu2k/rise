from __future__ import annotations

import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf

from instrument_robustness import prep_data
from instrument_robustness.config import (
    SR,
    StaleArtifactError,
    TARGET_LABELS,
    assert_artifact_fingerprint,
    config_fingerprint,
    write_artifact_fingerprint,
)
from instrument_robustness.step3_split import assign_groups, verify_no_group_leak
from instrument_robustness.step4_window import (
    MIN_CONTENT,
    WIN,
    tile_to_length,
    window_one,
)


class SplitRegressionTests(unittest.TestCase):
    def test_group_assignment_is_deterministic_and_leak_free(self) -> None:
        sizes = {"violin_A4": 5, "violin_B4": 3, "violin_C5": 2}
        fracs = {"train": 0.7, "val": 0.15, "test": 0.15}

        first = assign_groups(sizes, fracs, random.Random(17))
        second = assign_groups(sizes, fracs, random.Random(17))

        self.assertEqual(first, second)
        frame = pd.DataFrame(
            [
                {"grp": group, "split": split}
                for group, split in first.items()
                for _ in range(sizes[group])
            ]
        )
        self.assertEqual(verify_no_group_leak(frame), len(sizes))

    def test_leak_verifier_rejects_a_group_in_two_splits(self) -> None:
        frame = pd.DataFrame(
            [
                {"grp": "violin_A4", "split": "train"},
                {"grp": "violin_A4", "split": "test"},
            ]
        )

        with self.assertRaises(AssertionError):
            verify_no_group_leak(frame)


class WindowRegressionTests(unittest.TestCase):
    def test_short_signal_is_tiled_to_exact_length(self) -> None:
        segment = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

        result = tile_to_length(segment, 8)

        np.testing.assert_array_equal(
            result,
            np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]),
        )

    def test_window_writer_tiles_short_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            trimmed_root = root / "work" / "trimmed"
            trimmed = trimmed_root / "violin" / "A4.wav"
            windows = root / "work" / "windows"
            trimmed.parent.mkdir(parents=True)
            waveform = np.full(MIN_CONTENT + 10, 0.25, dtype=np.float32)
            sf.write(trimmed, waveform, SR, subtype="PCM_16")

            with (
                patch("instrument_robustness.step4_window.ROOT", root),
                patch("instrument_robustness.step4_window.TRIMMED", trimmed_root),
                patch("instrument_robustness.step4_window.WINDOWS", windows),
            ):
                rows = window_one(
                    (
                        "work/trimmed/violin/A4.wav",
                        "violin",
                        "A4",
                        "train",
                        "violin/A4/source.mp3",
                    )
                )

            self.assertEqual(len(rows), 1)
            output, output_sr = sf.read(root / rows[0][0], dtype="float32")
            self.assertEqual(output_sr, SR)
            self.assertEqual(output.shape, (WIN,))
            self.assertTrue(np.all(np.abs(output) > 0))

    def test_window_writer_drops_tiny_trailing_remainder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            trimmed_root = root / "work" / "trimmed"
            trimmed = trimmed_root / "violin" / "A4.wav"
            windows = root / "work" / "windows"
            trimmed.parent.mkdir(parents=True)
            waveform = np.full(WIN + MIN_CONTENT - 1, 0.25, dtype=np.float32)
            sf.write(trimmed, waveform, SR, subtype="PCM_16")

            with (
                patch("instrument_robustness.step4_window.ROOT", root),
                patch("instrument_robustness.step4_window.TRIMMED", trimmed_root),
                patch("instrument_robustness.step4_window.WINDOWS", windows),
            ):
                rows = window_one(
                    (
                        "work/trimmed/violin/A4.wav",
                        "violin",
                        "A4",
                        "train",
                        "violin/A4/source.mp3",
                    )
                )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][-1], 3.0)


class FingerprintRegressionTests(unittest.TestCase):
    def test_fingerprint_covers_load_bearing_pipeline_settings(self) -> None:
        fingerprint = config_fingerprint()

        self.assertEqual(fingerprint["split_group_fields"], ["label", "note"])
        self.assertEqual(fingerprint["split_seed"], 0)
        self.assertEqual(fingerprint["short_window_policy"], "tile")
        self.assertEqual(fingerprint["target_rms"], 0.1)
        self.assertEqual(fingerprint["articulations"]["violin"], ["arco-normal"])
        self.assertEqual(fingerprint["articulations"]["flute"], ["normal"])

    def test_sidecar_rejects_wrong_pipeline_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact = Path(temporary_dir) / "windows.csv"
            artifact.touch()
            write_artifact_fingerprint(artifact, "step4_window")

            with self.assertRaises(StaleArtifactError):
                assert_artifact_fingerprint(artifact, "step5_normalize")

    def test_sidecar_rejects_changed_artifact_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact = Path(temporary_dir) / "splits.csv"
            artifact.write_text("split\ntrain\n", encoding="utf-8")
            write_artifact_fingerprint(artifact, "step3_split")
            artifact.write_text("split\ntest\n", encoding="utf-8")

            with self.assertRaises(StaleArtifactError):
                assert_artifact_fingerprint(artifact, "step3_split")


class PrepDataRegressionTests(unittest.TestCase):
    def test_main_writes_manifest_and_its_fingerprint(self) -> None:
        rows = [
            {
                "path": f"{label}/A4/{label}_A4_1_forte_normal.mp3",
                "label": label,
                "family": prep_data.FAMILY[label],
                "duration_s": 1.0,
                "sample_rate": SR,
                "note": "A4",
                "midi": 69,
                "dynamic": "forte",
                "technique": "normal",
                "is_plain": 1,
                "is_phrase": 0,
            }
            for label in TARGET_LABELS
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manifest_path = root / "manifest.csv"
            fingerprint_path = root / "manifest_fingerprint.json"
            with (
                patch.multiple(
                    prep_data,
                    DATA_ROOT=root,
                    PIPE=root / "pipeline",
                    WORK=root / "work",
                    FEATURES=root / "features",
                    DATA_RAW=root / "raw",
                    MANIFEST_IN=manifest_path,
                    MANIFEST_FINGERPRINT=fingerprint_path,
                ),
                patch.object(prep_data, "download_and_extract"),
                patch.object(prep_data, "build_rows", return_value=(rows, Counter())),
            ):
                prep_data.main()

            self.assertTrue(manifest_path.is_file())
            self.assertTrue(fingerprint_path.is_file())
            assert_artifact_fingerprint(
                manifest_path,
                "prep_data",
                fingerprint_path=fingerprint_path,
            )


if __name__ == "__main__":
    unittest.main()
