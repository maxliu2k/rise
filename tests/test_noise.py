from __future__ import annotations

import json
import tempfile
import unittest
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
from instrument_robustness.noise_sweep import (
    CLIP_LEN,
    NOISE_MANIFEST_VERSION,
    NOISE_TYPES,
    SNRS,
    dataset_build_identity,
    measured_snr,
    mix_at_snr,
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
                output = (
                    noisy_dir / noise_type / f"snr{snr}" / f"{window_id}.wav"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
                rows.append(
                    {
                        "window_id": window_id,
                        "noise_type": noise_type,
                        "snr_db": snr,
                        "seed": 1,
                        "noise_source": noise_type,
                        "noise_source_sha256": "source-hash",
                        "crop_start_resampled_sample": 0,
                        "unscaled_noise_power": 1.0,
                        "realized_snr_db": snr,
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
        "n_test_windows": len(windows),
        "n_files": len(rows),
        "waveform_format": {
            "sample_rate": SR,
            "samples": CLIP_LEN,
            "channels": 1,
            "subtype": "FLOAT",
            "post_mix_normalization": False,
        },
        "seed_scheme": (
            "sha256(dataset_fingerprint|window_id|noise_type)[:4]; SNR excluded"
        ),
        "one_realization_scaled_to_all_snrs": True,
        "provenance_file": provenance.name,
        "provenance_sha256": sha256_file(provenance),
        "provenance_rows": len(rows),
    }
    (noisy_dir / "noise_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return noisy_dir


class NoiseTests(unittest.TestCase):
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

            self.assertEqual(len(summary), 16)
            self.assertTrue((output_dir / "metrics_clean.json").is_file())
            self.assertTrue(
                (output_dir / "model_test_mechanical_-5.csv").is_file()
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
