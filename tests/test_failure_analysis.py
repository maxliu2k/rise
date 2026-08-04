from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_robustness.config import NOISE_TYPES, TARGET_LABELS, config_fingerprint
from instrument_robustness.failure_analysis import (
    ANALYSIS_FAMILY,
    CANONICAL_DATASET_FINGERPRINT,
    IDENTITY_COLUMNS,
    MODEL_PREFIXES,
    N_PERMUTATIONS,
    ModelSweep,
    distance_confusion_tests,
    load_model_sweep,
    load_prediction_frame,
    matrix_permutation_spearman,
    normalized_curve_area,
    pair_values,
    row_normalized_confusion,
    scalar_permutation_spearman,
    training_feature_tables,
    validate_analysis_dataset,
    validate_shared_sweeps,
    validate_svm_training_features,
)
from instrument_robustness.featurelib import SVM_FEATURE_NAMES
from instrument_robustness.noise_eval_common import NoiseCondition


def prediction_frame(*, noise_source: str = "clean") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [f"w{index}" for index in range(len(TARGET_LABELS))],
            "source_path": [f"source{index}" for index in range(len(TARGET_LABELS))],
            "pitch_group": [f"{label}_A{index}" for index, label in enumerate(TARGET_LABELS)],
            "noise_source": [noise_source] * len(TARGET_LABELS),
            "true_label": TARGET_LABELS,
            "predicted_label": TARGET_LABELS,
        }
    )


def model_sweep(model: str, *, manifest: str = "manifest") -> ModelSweep:
    return ModelSweep(
        model=model,
        frames={"clean": prediction_frame()},
        dataset={"fingerprint": "dataset"},
        noise_manifest_sha256=manifest,
        model_sha256=f"{model}-hash",
        input_hashes={},
    )


class PureAnalysisTests(unittest.TestCase):
    def test_permutation_count_is_frozen_at_one_hundred_thousand(self) -> None:
        self.assertEqual(N_PERMUTATIONS, 100_000)

    def test_curve_area_uses_db_spacing(self) -> None:
        snrs = np.asarray([0.0, 10.0, 30.0])
        values = np.asarray([[0.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        area = normalized_curve_area(snrs, values)
        self.assertAlmostEqual(area[0], 25.0 / 30.0)
        self.assertAlmostEqual(area[1], 2.0)

    def test_symmetric_confusion_uses_directional_rates_not_counts(self) -> None:
        rows = []
        for index, label in enumerate(TARGET_LABELS):
            support = 4 if index == 0 else 2
            predictions = [label] * support
            if index == 0:
                predictions[:2] = [TARGET_LABELS[1]] * 2
            if index == 1:
                predictions[:1] = [TARGET_LABELS[0]]
            for prediction in predictions:
                rows.append({"true_label": label, "predicted_label": prediction})
        frame = pd.DataFrame(rows)
        rates = row_normalized_confusion(frame)
        pairs = pair_values(rates)
        # 0->1 is 2/4 and 1->0 is 1/2, so their symmetric rate is 0.5.
        self.assertAlmostEqual(pairs[0], 0.5)

    def test_training_centroids_use_standardized_88d_features(self) -> None:
        count = len(TARGET_LABELS)
        X = np.zeros((count, 88), dtype=np.float64)
        X[:, 0] = np.arange(count)
        y = np.arange(count)
        raw_mean = np.zeros(88)
        raw_std = np.ones(88)
        raw_std[SVM_FEATURE_NAMES.index("centroid_mean")] = 2.0
        X[:, SVM_FEATURE_NAMES.index("centroid_mean")] = 3.0
        centroids, profiles, distances = training_feature_tables(
            X, y, list(SVM_FEATURE_NAMES), raw_mean, raw_std
        )
        self.assertEqual(centroids.shape, (count, 90))
        self.assertEqual(profiles.shape, (count, 7))
        self.assertAlmostEqual(profiles.iloc[0]["spectral_centroid_hz"], 6.0)
        self.assertAlmostEqual(distances[0, count - 1], count - 1)

    def test_scalar_identity_permutation_finds_a_perfect_monotone_relation(self) -> None:
        rng = np.random.default_rng(4)
        permutations = np.vstack([rng.permutation(12) for _ in range(1000)])
        values = np.arange(12, dtype=float)
        result = scalar_permutation_spearman(values, values, permutations)
        self.assertAlmostEqual(result["spearman_rho"], 1.0)
        self.assertLess(result["permutation_p"], 0.01)

    def test_matrix_test_permutes_complete_instrument_identities(self) -> None:
        count = len(TARGET_LABELS)
        coordinates = np.arange(count, dtype=float)
        distances = np.abs(coordinates[:, None] - coordinates[None, :])
        outcome = -distances
        rng = np.random.default_rng(5)
        permutations = np.vstack([rng.permutation(count) for _ in range(1000)])
        result = matrix_permutation_spearman(distances, outcome, permutations)
        self.assertAlmostEqual(result["spearman_rho"], -1.0)
        self.assertLess(result["permutation_p"], 0.01)

    def test_primary_family_is_exactly_eighteen_tests(self) -> None:
        pairs = list(
            (TARGET_LABELS[first], TARGET_LABELS[second])
            for first in range(len(TARGET_LABELS))
            for second in range(first + 1, len(TARGET_LABELS))
        )
        rows = []
        for model in MODEL_PREFIXES:
            for noise_type in NOISE_TYPES:
                for first, second in pairs:
                    rows.append(
                        {
                            "model": model,
                            "instrument_a": first,
                            "instrument_b": second,
                            "noise_type": noise_type,
                            "mean_confusion_increase_auc": 0.0,
                        }
                    )
        count = len(TARGET_LABELS)
        coordinates = np.arange(count, dtype=float)
        distances = np.abs(coordinates[:, None] - coordinates[None, :])
        rng = np.random.default_rng(6)
        permutations = np.vstack([rng.permutation(count) for _ in range(20)])
        tests, correction = distance_confusion_tests(
            pd.DataFrame(rows), distances, permutations
        )
        expected = len(MODEL_PREFIXES) * len(NOISE_TYPES)
        self.assertEqual(expected, 18)
        self.assertEqual(len(tests), expected)
        self.assertEqual(correction["n_comparisons"], expected)
        self.assertEqual(correction["family"], ANALYSIS_FAMILY)
        self.assertTrue(tests["degenerate"].all())


class InputGateTests(unittest.TestCase):
    def test_analysis_gate_rejects_results_from_a_retired_build(self) -> None:
        sweeps = {model: model_sweep(model) for model in MODEL_PREFIXES}
        current = {"dataset_fingerprint": CANONICAL_DATASET_FINGERPRINT}
        with self.assertRaisesRegex(ValueError, "stale or from a different build"):
            validate_analysis_dataset(sweeps, current)

    def test_analysis_gate_rejects_a_noncanonical_active_data_root(self) -> None:
        dataset = {
            "dataset_fingerprint": CANONICAL_DATASET_FINGERPRINT,
            "windows_csv_sha256": "canonical-windows",
        }
        sweeps = {model: model_sweep(model) for model in MODEL_PREFIXES}
        sweeps = {
            model: ModelSweep(
                model=sweep.model,
                frames=sweep.frames,
                dataset=dataset,
                noise_manifest_sha256=sweep.noise_manifest_sha256,
                model_sha256=sweep.model_sha256,
                input_hashes=sweep.input_hashes,
            )
            for model, sweep in sweeps.items()
        }
        with self.assertRaisesRegex(ValueError, "active data root is not"):
            validate_analysis_dataset(
                sweeps,
                {"dataset_fingerprint": "retired-local-build"},
            )

    def test_analysis_gate_accepts_only_an_exact_canonical_identity(self) -> None:
        dataset = {
            "dataset_fingerprint": CANONICAL_DATASET_FINGERPRINT,
            "windows_csv_sha256": "canonical-windows",
        }
        sweeps = {model: model_sweep(model) for model in MODEL_PREFIXES}
        sweeps = {
            model: ModelSweep(
                model=sweep.model,
                frames=sweep.frames,
                dataset=dataset,
                noise_manifest_sha256=sweep.noise_manifest_sha256,
                model_sha256=sweep.model_sha256,
                input_hashes=sweep.input_hashes,
            )
            for model, sweep in sweeps.items()
        }
        validate_analysis_dataset(sweeps, dict(dataset))
        with self.assertRaisesRegex(ValueError, "full dataset identities differ"):
            validate_analysis_dataset(
                sweeps,
                {
                    "dataset_fingerprint": CANONICAL_DATASET_FINGERPRINT,
                    "windows_csv_sha256": "different-windows",
                },
            )

    def test_training_features_must_match_the_final_svm_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "train.npz"
            features.write_bytes(b"canonical features")
            digest = hashlib.sha256(features.read_bytes()).hexdigest()
            summary = root / "test_summary.json"
            summary.write_text(
                json.dumps({"input_files": {"train": {"sha256": digest}}}),
                encoding="utf-8",
            )
            validate_svm_training_features(features, summary)
            features.write_bytes(b"different features")
            with self.assertRaisesRegex(ValueError, "exact array"):
                validate_svm_training_features(features, summary)

    def test_prediction_loader_rejects_missing_true_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            prediction_frame().iloc[:-1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing true classes"):
                load_prediction_frame(path)

    def test_model_loader_rejects_mixed_model_hashes(self) -> None:
        conditions = [
            NoiseCondition("clean", "clean", None, None),
            NoiseCondition("white_0_r0", "white", 0, 0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, condition in enumerate(conditions):
                frame = prediction_frame(
                    noise_source="clean" if condition.tag == "clean" else "white-draw"
                )
                frame.to_csv(directory / f"test_{condition.tag}.csv", index=False)
                metrics = {
                    "condition": condition.tag,
                    "noise_type": condition.noise_type,
                    "snr_db": condition.snr_db,
                    "replicate": condition.replicate,
                    "n": len(frame),
                    "config_fingerprint": config_fingerprint(),
                    "dataset": {"fingerprint": "dataset"},
                    "noise_manifest": {"sha256": "manifest"},
                    "model_sha256": f"model-{index}",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "confusion_matrix": np.eye(len(TARGET_LABELS), dtype=int).tolist(),
                    "classification_report": {
                        label: {"recall": 1.0} for label in TARGET_LABELS
                    },
                }
                (directory / f"metrics_{condition.tag}.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
            pd.DataFrame(
                [
                    {"condition": condition.tag, "accuracy": 1.0, "macro_f1": 1.0}
                    for condition in conditions
                ]
            ).to_csv(
                directory / "noise_sweep_summary.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "mixes dataset"):
                load_model_sweep(
                    "test",
                    directory=directory,
                    prefix="test_",
                    conditions=conditions,
                )
            second_metrics = directory / "metrics_white_0_r0.json"
            repaired = json.loads(second_metrics.read_text(encoding="utf-8"))
            repaired["model_sha256"] = "model-0"
            second_metrics.write_text(json.dumps(repaired), encoding="utf-8")
            sweep = load_model_sweep(
                "test",
                directory=directory,
                prefix="test_",
                conditions=conditions,
            )
            self.assertEqual(set(sweep.frames), {"clean", "white_0_r0"})
            self.assertEqual(sweep.noise_manifest_sha256, "manifest")

    def test_shared_gate_rejects_a_different_noise_manifest(self) -> None:
        sweeps = {model: model_sweep(model) for model in MODEL_PREFIXES}
        sweeps["ast"] = model_sweep("ast", manifest="different")
        with self.assertRaisesRegex(ValueError, "different noise manifest"):
            validate_shared_sweeps(sweeps)

    def test_shared_gate_rejects_different_noise_draw_identity(self) -> None:
        sweeps = {model: model_sweep(model) for model in MODEL_PREFIXES}
        changed = prediction_frame()
        changed.loc[0, "noise_source"] = "different-draw"
        sweeps["ast"] = ModelSweep(
            model="ast",
            frames={"clean": changed},
            dataset={"fingerprint": "dataset"},
            noise_manifest_sha256="manifest",
            model_sha256="ast-hash",
            input_hashes={},
        )
        with self.assertRaisesRegex(ValueError, "noise_source"):
            validate_shared_sweeps(sweeps)

    def test_identity_columns_include_noise_source(self) -> None:
        self.assertIn("noise_source", IDENTITY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
