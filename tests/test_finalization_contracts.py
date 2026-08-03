from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instrument_robustness.config import TARGET_LABELS, config_fingerprint
from instrument_robustness import finalize_ast, finalize_panns, summarize_results
from instrument_robustness.bundle_weights import EXTERNAL_WEIGHTS, HISTORICAL_EXTERNAL_WEIGHTS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalizationContractTests(unittest.TestCase):
    def test_summarize_gates_on_selection_metric_even_when_macro_f1_is_present(self) -> None:
        """A balanced-accuracy-selected result must not print `canonical`.

        The gate used to run only when a summary had NO explicit macro_f1, so any result that
        recorded one skipped it entirely -- an AST checkpoint selected on balanced accuracy
        printed `canonical` for weeks purely because its summary also carried a macro-F1 number.
        If this test fires, that hole has reopened.
        """
        spec = dict(source="m/test_summary.json", fp="config_fingerprint",
                    test="test_metrics", split="test")
        fingerprint = config_fingerprint()
        base = {"config_fingerprint": fingerprint, "test_examples": 1255,
                "test_metrics": {"macro_f1": 0.99, "accuracy": 0.99}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "m").mkdir()
            summary = root / "m" / "test_summary.json"

            with patch.object(summarize_results, "ARTIFACTS", root):
                # selected on balanced accuracy -> refused, despite a valid macro_f1 being present
                summary.write_text(json.dumps(
                    {**base, "selection_metric": "validation_balanced_accuracy"}))
                row = summarize_results.clean_row("M", spec, fingerprint)
                self.assertIn("STALE", row["status"])
                self.assertIsNone(row["macro_f1"])

                # no metric anywhere -> also refused, rather than guessed
                summary.write_text(json.dumps(base))
                self.assertIn("STALE", summarize_results.clean_row("M", spec, fingerprint)["status"])

                # absent here but recorded in the sibling validation summary -> accepted.
                # finalize_svm/finalize_mert spent their one test evaluation before they began
                # propagating the field, so this fallback is what keeps them out of a false STALE.
                (root / "m" / "validation_summary.json").write_text(
                    json.dumps({"selection_metric": "validation_macro_f1"}))
                row = summarize_results.clean_row("M", spec, fingerprint)
                self.assertEqual(row["status"], "canonical")
                self.assertAlmostEqual(row["macro_f1"], 0.99)

    def test_external_panns_pointer_names_the_reported_finetune(self) -> None:
        """The external pointers name the FINE-TUNE that produced the reported number.

        This guarded against substituting the much smaller PANNs linear probe for the full
        fine-tune the reported result actually used. The 8,378-build entry moved to
        HISTORICAL_EXTERNAL_WEIGHTS when the corrected 97b1cdd2 build was trained; the guarantee
        is unchanged, so it is asserted in both places rather than dropped.
        """
        historical = HISTORICAL_EXTERNAL_WEIGHTS["panns_finetune_philharmonia_8378.pt"]
        self.assertEqual(
            historical["sha256"],
            "00cc195e1cbea756fc0afcb1ab823d639e31668c1a859f67941c29fda40741e3",
        )
        self.assertIn("v1.0-panns-12class", historical["download_url"])
        self.assertEqual(
            HISTORICAL_EXTERNAL_WEIGHTS["ast_finetuned_philharmonia_8378.safetensors"]["sha256"],
            "25789685e1cb0a4df0d64e5c84df84f49eff72c2831cc8e89fb02bd7676763e7",
        )

        # The CURRENT pointers must describe the corrected build, and must not silently reuse a
        # historical checkpoint's hash -- that substitution is exactly what build() refuses.
        frozen = "97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf"
        historical_hashes = {r["sha256"] for r in HISTORICAL_EXTERNAL_WEIGHTS.values()}
        self.assertEqual(set(EXTERNAL_WEIGHTS), {"ast_finetuned.safetensors", "panns_finetune.pt"})
        for name, record in EXTERNAL_WEIGHTS.items():
            self.assertEqual(record["dataset_fingerprint"], frozen, name)
            self.assertNotIn(record["sha256"], historical_hashes, name)
            self.assertTrue(record["scc_path"].endswith(name), name)

    def test_panns_selection_contract_and_single_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            windows = root / "windows.csv"
            windows.write_text("split\ntrain\n", encoding="utf-8")
            model = root / "selected_model.pt"
            model.write_bytes(b"selected PANNs")
            (root / "validation_summary.json").write_text(json.dumps({
                "test_evaluated": False,
                "selection_metric": "validation_macro_f1",
                "label_order": list(TARGET_LABELS),
                "config_fingerprint": config_fingerprint(),
                "windows_manifest": {"sha256": digest(windows)},
                "output_files": {"model": {"sha256": digest(model)}},
            }), encoding="utf-8")
            with patch.object(finalize_panns, "WINDOWS_CSV", windows):
                finalize_panns.validate_selection(root)
                finalize_panns.claim_finalization(root)
                with self.assertRaises(FileExistsError):
                    finalize_panns.claim_finalization(root)

    def test_ast_rejects_non_macro_f1_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            windows = root / "windows.csv"
            windows.write_text("split\ntrain\n", encoding="utf-8")
            model = root / "model.safetensors"
            model.write_bytes(b"selected AST")
            (root / "validation_summary.json").write_text(json.dumps({
                "test_evaluated": False,
                "selection_metric": "validation_balanced_accuracy",
                "label_order": list(TARGET_LABELS),
                "config_fingerprint": config_fingerprint(),
                "windows_manifest": {"sha256": digest(windows)},
                "output_files": {"model": {"sha256": digest(model)}},
            }), encoding="utf-8")
            with patch.object(finalize_ast, "WINDOWS_CSV", windows):
                with self.assertRaisesRegex(ValueError, "macro-F1"):
                    finalize_ast.validate_selection(root)

    def test_training_entry_points_do_not_reference_test_split(self) -> None:
        panns_source = Path("src/instrument_robustness/train_panns.py").read_text()
        ast_source = Path("src/instrument_robustness/train_ast.py").read_text()
        self.assertNotIn('load_split("test"', panns_source)
        self.assertNotIn('make_ast_dataloader("test"', ast_source)
        self.assertNotIn('dfs["test"]', panns_source)


if __name__ == "__main__":
    unittest.main()
