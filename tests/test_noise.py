from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from instrument_robustness.config import (
    SR,
    TARGET_LABELS,
    config_fingerprint_json,
    write_artifact_fingerprint,
)
from instrument_robustness.featurelib import SVM_FEATURE_NAMES
from instrument_robustness.noise_eval_common import (
    assert_clean_parity,
    load_test_frame,
    run_noise_evaluation,
)
from instrument_robustness.noise_stats import (
    cluster_bootstrap,
    cluster_sign_test,
    macro_f1,
    paired_frames,
)
from instrument_robustness.noise_eval_svm import load_training_statistics
from instrument_robustness.config import N_REPLICATES
from instrument_robustness.noise_metrics import DIAGNOSTIC_COLUMNS
from instrument_robustness.noise_sweep import (
    CLIP_LEN,
    SEED_SCHEME,
    out_path,
    NOISE_MANIFEST_VERSION,
    NOISE_TYPES,
    SNRS,
    Esc50Clip,
    dataset_build_identity,
    diagnostic_protocol,
    draw_noise,
    generate,
    DEMAND_CHANNEL,
    DEMAND_ENVIRONMENTS,
    DemandRecording,
    load_demand_index,
    load_esc50_index,
    measured_snr,
    mix_at_snr,
    MAX_REALIZATION_COSINE_DEVIATION,
    noise_preprocessing_protocol,
    read_audio_window,
    sha256_file,
    validate_noise_manifest,
    window_seed,
)


def write_dataset_files(root: Path, *, all_labels: bool = False) -> dict[str, Path]:
    pipeline = root / "pipeline"
    pipeline.mkdir(parents=True)
    manifest_csv = root / "manifest.csv"
    manifest_fingerprint = root / "manifest_fingerprint.json"
    manifest_labeled = pipeline / "manifest_labeled.csv"
    windows_csv = pipeline / "windows.csv"
    labels = TARGET_LABELS if all_labels else TARGET_LABELS[:1]
    manifest_rows = []
    window_rows = []
    for index, label in enumerate(labels):
        source_path = f"raw/{label}_A{index}.mp3"
        window_path = f"work/windows/{label}_A{index}_w000.wav"
        manifest_rows.append(
            {"path": source_path, "label": label, "note": f"A{index}"}
        )
        window_rows.append(
            {
                "window_path": window_path,
                "source_path": source_path,
                "label": label,
                "split": "test",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    write_artifact_fingerprint(
        manifest_csv,
        "prep_data",
        fingerprint_path=manifest_fingerprint,
    )
    pd.DataFrame(manifest_rows).to_csv(manifest_labeled, index=False)
    write_artifact_fingerprint(manifest_labeled, "step0_filter")
    pd.DataFrame(window_rows).to_csv(windows_csv, index=False)
    write_artifact_fingerprint(windows_csv, "step5_normalize")
    return {
        "manifest_csv": manifest_csv,
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_labeled": manifest_labeled,
        "windows_csv": windows_csv,
    }


def prediction_frame(
    predicted: list[str],
    *,
    true: list[str] | None = None,
) -> pd.DataFrame:
    true = true or [TARGET_LABELS[0]] * len(predicted)
    return pd.DataFrame(
        {
            "window_id": [f"window_{index}" for index in range(len(predicted))],
            "source_path": [f"source_{index // 2}" for index in range(len(predicted))],
            "pitch_group": [f"group_{index // 2}" for index in range(len(predicted))],
            "true_label": true,
            "predicted_label": predicted,
            "correct": np.asarray(true) == np.asarray(predicted),
        }
    )


def write_completed_noise_sweep(root: Path, paths: dict[str, Path]) -> Path:
    noisy_dir = root / "work" / "windows_noisy"
    noisy_dir.mkdir(parents=True)
    identity = dataset_build_identity(
        manifest_csv=paths["manifest_csv"],
        manifest_fingerprint=paths["manifest_fingerprint"],
        windows_csv=paths["windows_csv"],
    )
    rows = []
    windows = pd.read_csv(paths["windows_csv"])
    for window in windows.itertuples(index=False):
        clean_path = root / window.window_path
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.touch()
        window_id = Path(window.window_path).stem
        for noise_type in NOISE_TYPES:
            for snr in SNRS:
              for replicate in range(N_REPLICATES):
                output = out_path(
                    noise_type, snr, window_id, replicate=replicate, noisy_dir=noisy_dir
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                rows.append(
                    {
                        "window_id": window_id,
                        "noise_type": noise_type,
                        "snr_db": snr,
                        "replicate": replicate,
                        "seed": 1,
                        "noise_source": noise_type,
                        "noise_source_sha256": "source-hash",
                        "crop_start_resampled_sample": 0,
                        "noise_target": None if noise_type == "white" else 3,
                        "noise_category": None if noise_type == "white" else "dog",
                        "noise_fold": None if noise_type == "white" else 1,
                        "unscaled_noise_power": 1.0,
                        "realized_snr_db": snr,
                        **{column: 0.0 for column in DIAGNOSTIC_COLUMNS},
                        "output_path": str(output.relative_to(root)),
                        "output_sha256": sha256_file(output),
                    }
                )
    provenance = noisy_dir / "noise_provenance.csv"
    pd.DataFrame(rows).to_csv(provenance, index=False)
    manifest = {
        "manifest_version": NOISE_MANIFEST_VERSION,
        "state": "complete",
        "dataset": identity,
        "snrs": SNRS,
        "noise_types": NOISE_TYPES,
        "n_replicates": N_REPLICATES,
        "n_test_windows": len(windows),
        "n_files": len(rows),
        "waveform_format": {
            "sample_rate": SR,
            "samples": CLIP_LEN,
            "channels": 1,
            "subtype": "FLOAT",
            "post_mix_normalization": False,
        },
        "seed_scheme": SEED_SCHEME,
        "one_realization_scaled_to_all_snrs": True,
        "noise_preprocessing": noise_preprocessing_protocol(),
        "diagnostics": diagnostic_protocol(),
        "provenance_file": provenance.name,
        "provenance_sha256": sha256_file(provenance),
        "provenance_rows": len(rows),
    }
    (noisy_dir / "noise_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return noisy_dir


class Esc50ProvenanceTests(unittest.TestCase):
    """The ESC-50 category and fold must reach per-mixture provenance.

    Collapsing 20 ESC-50 classes into "natural" is a defensible grouping, but only if the original
    label survives into the output -- it cannot be recovered afterwards without regenerating the
    whole sweep, so a missing column here is an unrecoverable loss rather than an inconvenience.
    """

    def test_white_noise_carries_null_corpus_fields(self) -> None:
        noise, provenance = draw_noise("white", np.random.default_rng(0), {})
        self.assertEqual(noise.shape, (CLIP_LEN,))
        self.assertAlmostEqual(float(np.mean(noise, dtype=np.float64)), 0.0, places=8)
        # Present-but-None rather than absent, so the provenance CSV has one schema across every
        # noise type instead of ragged columns.
        for field in ("noise_target", "noise_category", "noise_fold"):
            self.assertIn(field, provenance)
            self.assertIsNone(provenance[field])

    def test_esc50_index_requires_category_and_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio, meta = root / "audio", root / "meta"
            audio.mkdir()
            meta.mkdir()
            (audio / "1-137-A-32.wav").touch()
            pd.DataFrame([{"filename": "1-137-A-32.wav", "target": 32}]).to_csv(
                meta / "esc50.csv", index=False
            )
            with unittest.mock.patch.multiple(
                "instrument_robustness.noise_sweep",
                ESC50_ROOT=root,
                ESC50_DIR=audio,
                ESC50_META=meta / "esc50.csv",
            ):
                with self.assertRaises(ValueError) as caught:
                    load_esc50_index()
        message = str(caught.exception)
        self.assertIn("category", message)
        self.assertIn("fold", message)

    def test_drawn_segment_reports_its_original_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clip_path = root / "audio" / "1-137-A-32.wav"
            clip_path.parent.mkdir(parents=True)
            sf.write(
                str(clip_path),
                np.random.default_rng(1).standard_normal(CLIP_LEN * 2).astype("float32"),
                SR,
            )
            clip = Esc50Clip(path=clip_path, target=32, category="pouring_water", fold=4)
            with unittest.mock.patch(
                "instrument_robustness.noise_sweep.ESC50_ROOT", root
            ):
                _, provenance = draw_noise(
                    "natural",
                    np.random.default_rng(0),
                    {"natural": [clip]},
                )
        self.assertEqual(provenance["noise_target"], 32)
        self.assertEqual(provenance["noise_category"], "pouring_water")
        self.assertEqual(provenance["noise_fold"], 4)

    def test_esc50_crop_is_centered_before_its_power_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clip_path = root / "audio" / "offset.wav"
            clip_path.parent.mkdir(parents=True)
            time = np.arange(CLIP_LEN, dtype=np.float64) / SR
            waveform = (0.25 + 0.01 * np.sin(2 * np.pi * 440 * time)).astype(
                np.float32
            )
            sf.write(str(clip_path), waveform, SR, subtype="FLOAT")
            clip = Esc50Clip(path=clip_path, target=17, category="thunderstorm", fold=2)
            with unittest.mock.patch(
                "instrument_robustness.noise_sweep.ESC50_ROOT", root
            ):
                noise, _ = draw_noise(
                    "natural",
                    np.random.default_rng(0),
                    {"natural": [clip]},
                )

        self.assertAlmostEqual(float(np.mean(noise, dtype=np.float64)), 0.0, places=8)
        self.assertGreater(float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))), 1e-3)

    def test_constant_esc50_crop_is_silent_after_centering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            clip_path = root / "audio" / "constant.wav"
            clip_path.parent.mkdir(parents=True)
            sf.write(
                str(clip_path),
                np.full(CLIP_LEN, 0.25, dtype=np.float32),
                SR,
                subtype="FLOAT",
            )
            clip = Esc50Clip(path=clip_path, target=17, category="thunderstorm", fold=2)
            with unittest.mock.patch(
                "instrument_robustness.noise_sweep.ESC50_ROOT", root
            ):
                with self.assertRaisesRegex(ValueError, "non-silent centered"):
                    draw_noise(
                        "natural",
                        np.random.default_rng(0),
                        {"natural": [clip]},
                    )

    def test_manifest_validation_rejects_provenance_without_the_new_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root, all_labels=True)
            noisy_dir = write_completed_noise_sweep(root, paths)
            provenance_path = noisy_dir / "noise_provenance.csv"
            frame = pd.read_csv(provenance_path)
            frame = frame.drop(columns=["noise_category", "noise_fold"])
            frame.to_csv(provenance_path, index=False)
            manifest_path = noisy_dir / "noise_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["provenance_sha256"] = sha256_file(provenance_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                validate_noise_manifest(
                    noisy_dir=noisy_dir,
                    data_root=root,
                    windows_csv=paths["windows_csv"],
                    manifest_csv=paths["manifest_csv"],
                    manifest_fingerprint=paths["manifest_fingerprint"],
                )
        self.assertIn("missing columns", str(caught.exception))

    def test_manifest_validation_rejects_excessive_residual_dc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root, all_labels=True)
            noisy_dir = write_completed_noise_sweep(root, paths)
            provenance_path = noisy_dir / "noise_provenance.csv"
            frame = pd.read_csv(provenance_path)
            frame.loc[0, "noise_dc_power_share"] = 0.5
            frame.to_csv(provenance_path, index=False)
            manifest_path = noisy_dir / "noise_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["provenance_sha256"] = sha256_file(provenance_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                validate_noise_manifest(
                    noisy_dir=noisy_dir,
                    data_root=root,
                    windows_csv=paths["windows_csv"],
                    manifest_csv=paths["manifest_csv"],
                    manifest_fingerprint=paths["manifest_fingerprint"],
                )

        self.assertIn("residual DC", str(caught.exception))



class DemandCorpusTests(unittest.TestCase):
    """DEMAND indexing and drawing. `studio` is wired but NOT in config.NOISE_TYPES yet."""

    def build(self, root: Path, *, seconds: float = 8.0, channels: int = 16) -> None:
        rng = np.random.default_rng(0)
        for environment in DEMAND_ENVIRONMENTS:
            directory = root / environment
            directory.mkdir(parents=True)
            for channel in range(1, channels + 1):
                tone = rng.standard_normal(int(48000 * seconds)).astype(np.float32) * 0.05
                sf.write(directory / f"ch{channel:02d}.wav", tone, 48000, subtype="PCM_16")

    def test_indexes_one_channel_per_environment_not_sixteen(self) -> None:
        """The 16 channels are one array on one scene; indexing all of them fakes 16x diversity.

        If this fires, someone has started treating ch01..ch16 as independent samples. That
        multiplies the apparent studio corpus by 16 while adding almost no acoustic diversity,
        and nothing downstream would reveal it.
        """
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.build(root)
            index = load_demand_index(root)

            self.assertEqual(set(index), {"studio"})
            self.assertEqual(len(index["studio"]), len(DEMAND_ENVIRONMENTS))
            self.assertEqual(len(index["studio"]), 18)
            self.assertEqual({r.channel for r in index["studio"]}, {DEMAND_CHANNEL})
            # ordered by environment name, because a seeded draw indexes into this list
            self.assertEqual(
                [r.environment for r in index["studio"]], sorted(DEMAND_ENVIRONMENTS)
            )

    def test_draw_is_seed_reproducible_and_carries_environment_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.build(root, channels=1)
            index = load_demand_index(root)

            first, provenance = draw_noise(
                "studio", np.random.default_rng(7), {}, demand_index=index
            )
            again, _ = draw_noise("studio", np.random.default_rng(7), {}, demand_index=index)

            self.assertEqual(first.shape, (CLIP_LEN,))
            self.assertEqual(first.dtype, np.float32)
            np.testing.assert_array_equal(first, again)
            self.assertAlmostEqual(float(np.mean(first, dtype=np.float64)), 0.0, places=6)

            self.assertIn(provenance["noise_environment"], DEMAND_ENVIRONMENTS)
            self.assertEqual(provenance["noise_source_sr"], 48000)
            # ESC-50 fields present-but-None keeps the provenance CSV one schema
            self.assertIsNone(provenance["noise_target"])
            self.assertIsNone(provenance["noise_fold"])

    def test_studio_without_a_demand_index_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs a DEMAND index"):
            draw_noise("studio", np.random.default_rng(0), {})

    def test_a_recording_too_short_for_one_window_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.build(root, seconds=1.0, channels=1)
            with self.assertRaisesRegex(ValueError, "need at least"):
                load_demand_index(root)

    def test_seed_is_build_scoped_and_snr_independent(self) -> None:
        first = window_seed("window", "white", "build-a")
        self.assertEqual(first, window_seed("window", "white", "build-a"))
        self.assertNotEqual(first, window_seed("window", "white", "build-b"))

    def test_power_snr_is_recovered(self) -> None:
        clean = np.full(CLIP_LEN, 0.1, dtype=np.float32)
        noise = np.random.default_rng(0).standard_normal(CLIP_LEN).astype(np.float32)
        for snr in SNRS:
            mixture, _, _ = mix_at_snr(clean, noise, snr)
            self.assertAlmostEqual(measured_snr(clean, mixture), snr, places=4)

    def test_realization_tolerance_separates_rounding_from_a_redrawn_realization(self) -> None:
        """The reuse bound must sit between float32 rounding and an actually different draw.

        validate() checks that one realization is scaled to every SNR by comparing the added
        components `noisy - clean`. `noisy` is float32, so that subtraction is never exact and
        the cosine deviation has a nonzero floor. The bound was originally 1e-6, which is BELOW
        that floor: on 2026-08-03 it failed the entire sweep on a mechanical draw measuring
        6.62e-6 of correct arithmetic.

        This pins both sides. If it fires, the bound has drifted into the rounding noise again
        (lower assert) or grown so loose it would accept a genuinely re-drawn realization
        (upper assert).
        """
        rng = np.random.default_rng(0)
        clean = rng.standard_normal(CLIP_LEN).astype(np.float32)
        clean *= 0.1 / float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))

        def added_components(noise: np.ndarray) -> list[np.ndarray]:
            return [mix_at_snr(clean, noise, snr)[0] - clean for snr in SNRS]

        def cosine(first: np.ndarray, second: np.ndarray) -> float:
            unit = lambda v: v / (np.linalg.norm(v) + 1e-12)  # noqa: E731
            return float(abs(np.dot(unit(first), unit(second))))

        one = rng.standard_normal(CLIP_LEN).astype(np.float32)
        components = added_components(one)
        rounding = 1 - min(cosine(c, components[0]) for c in components)

        other = rng.standard_normal(CLIP_LEN).astype(np.float32)
        redrawn = 1 - cosine(added_components(other)[-1], components[0])

        self.assertLess(rounding, MAX_REALIZATION_COSINE_DEVIATION)
        self.assertGreater(redrawn, MAX_REALIZATION_COSINE_DEVIATION)
        # and the two must not be anywhere near each other
        self.assertGreater(redrawn / max(rounding, 1e-12), 1e3)

        # Synthetic Gaussian input rounds far more kindly than real audio -- it lands near 1e-7,
        # which would have slipped under the old 1e-6 bound too. So this assertion, not the one
        # above, is what actually pins the 2026-08-03 regression: the worst rounding MEASURED on
        # the sealed 97b1cdd2 build (mechanical, 6.62e-6) must sit well inside the bound.
        WORST_MEASURED_ROUNDING = 6.62e-6
        self.assertGreater(
            MAX_REALIZATION_COSINE_DEVIATION,
            WORST_MEASURED_ROUNDING * 10,
            "bound is within 10x of rounding measured on real audio; it will false-alarm",
        )

    def test_float_window_preserves_headroom_and_rejects_wrong_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            valid = root / "valid.wav"
            waveform = np.full(CLIP_LEN, 2.5, dtype=np.float32)
            sf.write(valid, waveform, SR, subtype="FLOAT")
            loaded = read_audio_window(valid)
            self.assertGreater(float(loaded.max()), 1.0)

            short = root / "short.wav"
            sf.write(short, waveform[:-1], SR, subtype="FLOAT")
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                read_audio_window(short)

    def test_dataset_identity_includes_actual_windows_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root)
            first = dataset_build_identity(
                manifest_csv=paths["manifest_csv"],
                manifest_fingerprint=paths["manifest_fingerprint"],
                windows_csv=paths["windows_csv"],
            )
            frame = pd.read_csv(paths["windows_csv"])
            frame.loc[0, "window_path"] = "work/windows/changed.wav"
            frame.to_csv(paths["windows_csv"], index=False)
            write_artifact_fingerprint(paths["windows_csv"], "step5_normalize")
            second = dataset_build_identity(
                manifest_csv=paths["manifest_csv"],
                manifest_fingerprint=paths["manifest_fingerprint"],
                windows_csv=paths["windows_csv"],
            )
            self.assertNotEqual(
                first["dataset_fingerprint"],
                second["dataset_fingerprint"],
            )

    def test_manifest_validation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root)
            noisy_dir = write_completed_noise_sweep(root, paths)
            manifest_path = noisy_dir / "noise_manifest.json"

            validate_noise_manifest(
                noisy_dir=noisy_dir,
                data_root=root,
                windows_csv=paths["windows_csv"],
                manifest_csv=paths["manifest_csv"],
                manifest_fingerprint=paths["manifest_fingerprint"],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset"]["windows_csv_sha256"] = "stale"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol/build"):
                validate_noise_manifest(
                    noisy_dir=noisy_dir,
                    data_root=root,
                    windows_csv=paths["windows_csv"],
                    manifest_csv=paths["manifest_csv"],
                    manifest_fingerprint=paths["manifest_fingerprint"],
                )

    def test_manifest_rejects_stale_diagnostic_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root)
            noisy_dir = write_completed_noise_sweep(root, paths)
            manifest_path = noisy_dir / "noise_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["diagnostics"]["instrument_band_hz"] = [50.0, 8000.0]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "diagnostics"):
                validate_noise_manifest(
                    noisy_dir=noisy_dir,
                    data_root=root,
                    windows_csv=paths["windows_csv"],
                    manifest_csv=paths["manifest_csv"],
                    manifest_fingerprint=paths["manifest_fingerprint"],
                )

    def test_shared_runner_writes_all_conditions_after_clean_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root, all_labels=True)
            noisy_dir = write_completed_noise_sweep(root, paths)
            output_dir = root / "artifacts" / "model" / "noise"

            def predict_scores(input_paths: list[Path]) -> np.ndarray:
                scores = np.zeros(
                    (len(input_paths), len(TARGET_LABELS)),
                    dtype=np.float32,
                )
                for index in range(len(input_paths)):
                    scores[index, index % len(TARGET_LABELS)] = 1.0
                return scores

            summary = run_noise_evaluation(
                model_name="model",
                file_prefix="model_test_",
                predict_scores=predict_scores,
                official_macro_f1=1.0,
                official_examples=len(TARGET_LABELS),
                model_sha256="model-hash",
                output_dir=output_dir,
                windows_csv=paths["windows_csv"],
                manifest_labeled=paths["manifest_labeled"],
                data_root=root,
                noisy_dir=noisy_dir,
                manifest_csv=paths["manifest_csv"],
                manifest_fingerprint=paths["manifest_fingerprint"],
            )

            # Derived from the configured grid, not a literal: this test is about the runner
            # covering EVERY condition, so it must not break when the grid is retuned.
            self.assertEqual(
                len(summary),
                1 + len(NOISE_TYPES) * len(SNRS) * N_REPLICATES,
            )
            self.assertTrue((output_dir / "metrics_clean.json").is_file())
            for noise_type in NOISE_TYPES:
                for snr in SNRS:
                    for replicate in range(N_REPLICATES):
                        suffix = (
                            f"_r{replicate}" if N_REPLICATES > 1 else ""
                        )
                        self.assertTrue(
                            (
                                output_dir
                                / f"model_test_{noise_type}_{snr}{suffix}.csv"
                            ).is_file(),
                            f"missing predictions for {noise_type} at {snr} dB, "
                            f"replicate {replicate}",
                        )

    def test_pitch_groups_come_from_authoritative_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root, all_labels=True)
            frame = load_test_frame(
                windows_csv=paths["windows_csv"],
                manifest_labeled=paths["manifest_labeled"],
            )
            self.assertEqual(len(frame), len(TARGET_LABELS))
            self.assertEqual(
                frame.iloc[0]["pitch_group"],
                f"{TARGET_LABELS[0]}_A0",
            )

    def test_test_frame_accepts_matching_note_already_in_windows_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root, all_labels=True)
            windows = pd.read_csv(paths["windows_csv"])
            manifest = pd.read_csv(paths["manifest_labeled"]).rename(
                columns={"path": "source_path"}
            )
            windows = windows.merge(
                manifest[["source_path", "note"]],
                on="source_path",
                validate="one_to_one",
            )
            windows.to_csv(paths["windows_csv"], index=False)
            write_artifact_fingerprint(paths["windows_csv"], "step5_normalize")

            frame = load_test_frame(
                windows_csv=paths["windows_csv"],
                manifest_labeled=paths["manifest_labeled"],
            )

            self.assertIn("note", frame)
            self.assertNotIn("manifest_note", frame)
            self.assertTrue(
                frame["pitch_group"].equals(
                    frame["label"].astype(str) + "_" + frame["note"].astype(str)
                )
            )

    def test_clean_parity_requires_an_official_result_and_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "Official clean macro-F1"):
            assert_clean_parity(
                0.9,
                official_macro_f1=None,
                measured_examples=10,
                official_examples=10,
            )
        with self.assertRaisesRegex(ValueError, "evaluated 10"):
            assert_clean_parity(
                0.9,
                official_macro_f1=0.9,
                measured_examples=10,
                official_examples=11,
            )

    def test_clean_parity_allows_only_small_cross_platform_drift(self) -> None:
        assert_clean_parity(
            0.9015,
            official_macro_f1=0.9,
            measured_examples=10,
            official_examples=10,
        )
        with self.assertRaisesRegex(ValueError, "Clean parity failed"):
            assert_clean_parity(
                0.9025,
                official_macro_f1=0.9,
                measured_examples=10,
                official_examples=10,
            )

    def test_svm_noise_adapter_reuses_saved_train_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "stats.npz"
            mean = np.arange(len(SVM_FEATURE_NAMES), dtype=np.float32)
            std = np.full(len(SVM_FEATURE_NAMES), 2.0, dtype=np.float32)
            np.savez(
                path,
                svm_mean=mean,
                svm_std=std,
                svm_feature_names=np.asarray(SVM_FEATURE_NAMES),
                computed_on=np.asarray("train"),
                config_fingerprint=np.asarray(config_fingerprint_json()),
            )
            loaded_mean, loaded_std = load_training_statistics(path)
            np.testing.assert_array_equal(loaded_mean, mean)
            np.testing.assert_array_equal(loaded_std, std)

    def test_macro_f1_keeps_all_twelve_labels(self) -> None:
        frame = prediction_frame([TARGET_LABELS[0]])
        self.assertAlmostEqual(macro_f1(frame), 1 / len(TARGET_LABELS))

    def test_pairing_checks_cluster_and_truth_columns(self) -> None:
        first = prediction_frame([TARGET_LABELS[0], TARGET_LABELS[0]])
        second = first.copy()
        second.loc[0, "pitch_group"] = "different"
        with self.assertRaisesRegex(ValueError, "pitch_group"):
            paired_frames(first, second)

    def test_cluster_statistics_are_explicit_and_reproducible(self) -> None:
        true = [TARGET_LABELS[0]] * 4
        first = prediction_frame(
            [TARGET_LABELS[1]] * 4,
            true=true,
        )
        second = prediction_frame(
            [TARGET_LABELS[0]] * 4,
            true=true,
        )
        bootstrap = cluster_bootstrap(first, second, n_boot=20, seed=3)
        sign = cluster_sign_test(first, second)
        self.assertEqual(bootstrap["fixed_label_order"], TARGET_LABELS)
        self.assertGreater(bootstrap["delta_macro_f1"], 0)
        self.assertEqual(sign["test"], "exact_cluster_sign_test")
        self.assertEqual(sign["b_better"], 2)


if __name__ == "__main__":
    unittest.main()


class ChunkedGenerationMatchesMonolithic(unittest.TestCase):
    """A sweep generated in (noise_type, replicate) chunks must be byte-identical to one pass.

    This is the property the streaming evaluator rests on: the noise corpus is ~16 GB and cannot
    be materialized whole under the project quota, so it is generated a chunk at a time, scored,
    and deleted. If a chunked file differed from a monolithic one by even a rounding step, the
    fine-tuned MERT would be scored on audio the other six models never saw -- the same class of
    failure REPOSITORY_AUDIT.md records for PANNs, arriving as a plausible number rather than a
    crash.

    White noise is used because it needs no external corpus, so this runs anywhere the suite
    runs. The mechanism under test -- one draw per (window, type, replicate), rescaled across
    every SNR -- is identical for the ESC-50 and DEMAND paths.
    """

    def _write_clean_windows(self, root: Path, paths: dict[str, Path]) -> None:
        rng = np.random.default_rng(0)
        for window in pd.read_csv(paths["windows_csv"]).itertuples(index=False):
            clean_path = root / window.window_path
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            audio = rng.standard_normal(CLIP_LEN).astype(np.float32) * 0.1
            sf.write(clean_path, audio, SR, subtype="FLOAT")

    def _hash_wavs(self, noisy_dir: Path) -> dict[str, str]:
        """Hash the decoded SAMPLES, never the file.

        A float WAV written by libsndfile carries a PEAK chunk whose timeStamp field holds the
        Unix time of the write, so two files with identical audio written a second apart differ
        at exactly one byte (offset 60). Hashing the container would make this test fail for a
        reason that has nothing to do with the audio -- and, more importantly, it is why the
        `output_sha256` column in the noise provenance cannot be used to verify a REGENERATED
        corpus against an older run. Compare predictions for that, not hashes.
        """
        import hashlib

        digests = {}
        for path in sorted(noisy_dir.rglob("*.wav")):
            samples, sample_rate = sf.read(path, dtype="float32")
            self.assertEqual(sample_rate, SR)
            digests[str(path.relative_to(noisy_dir))] = hashlib.sha256(
                samples.tobytes()
            ).hexdigest()
        return digests

    def _generate(self, root, paths, noisy_dir, replicates):
        return generate(
            data_root=root,
            windows_csv=paths["windows_csv"],
            manifest_csv=paths["manifest_csv"],
            manifest_fingerprint=paths["manifest_fingerprint"],
            noisy_dir=noisy_dir,
            only_noise_types=("white",),
            only_replicates=replicates,
            write_completion=False,
        )

    def test_chunked_output_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root)
            self._write_clean_windows(root, paths)

            whole_dir = root / "whole"
            self._generate(root, paths, whole_dir, tuple(range(N_REPLICATES)))
            whole = self._hash_wavs(whole_dir)
            self.assertEqual(len(whole), len(SNRS) * N_REPLICATES)

            chunked_dir = root / "chunked"
            chunked: dict[str, str] = {}
            for replicate in range(N_REPLICATES):
                self._generate(root, paths, chunked_dir, (replicate,))
                chunked.update(self._hash_wavs(chunked_dir))
                # Delete between chunks, exactly as the streaming evaluator does -- and prove
                # the next chunk is not silently reusing what the previous one left behind.
                for path in chunked_dir.rglob("*.wav"):
                    path.unlink()

            self.assertEqual(whole, chunked)

    def test_partial_sweep_cannot_claim_completion(self) -> None:
        with self.assertRaisesRegex(ValueError, "completion manifest for a partial sweep"):
            generate(only_noise_types=("white",), write_completion=True)

    def test_chunk_provenance_does_not_overwrite_the_canonical_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = write_dataset_files(root)
            self._write_clean_windows(root, paths)
            noisy_dir = root / "chunk"
            written = self._generate(root, paths, noisy_dir, (0,))
            self.assertNotEqual(written.name, "noise_provenance.csv")
            self.assertFalse((noisy_dir / "noise_provenance.csv").exists())
            self.assertFalse((noisy_dir / "noise_manifest.json").exists())

    def test_wav_file_hash_is_not_reproducible_but_audio_is(self) -> None:
        """Pin the reason chunk equality is checked on samples rather than on file bytes.

        libsndfile stamps the PEAK chunk of a float WAV with the write time. If a future
        libsndfile stops doing that, this test fails and the _hash_wavs docstring -- and the
        claim that provenance output_sha256 cannot verify a regenerated corpus -- should be
        revisited rather than left as folklore.
        """
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            audio = np.linspace(-0.5, 0.5, CLIP_LEN, dtype=np.float32)
            first, second = root / "a.wav", root / "b.wav"
            sf.write(first, audio, SR, subtype="FLOAT")
            sf.write(second, audio, SR, subtype="FLOAT")

            same_samples = np.array_equal(
                sf.read(first, dtype="float32")[0],
                sf.read(second, dtype="float32")[0],
            )
            self.assertTrue(same_samples, "identical input produced different samples")

            differing = [
                offset
                for offset, (x, y) in enumerate(
                    zip(first.read_bytes(), second.read_bytes())
                )
                if x != y
            ]
            # Zero when both writes land in the same second; never more than the 4-byte stamp.
            self.assertLessEqual(len(differing), 4)
            if differing:
                self.assertTrue(
                    all(56 <= offset < 64 for offset in differing),
                    f"container differs outside the PEAK timestamp: {differing}",
                )
