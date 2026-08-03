from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instrument_robustness.config import TARGET_LABELS, config_fingerprint
from instrument_robustness import finalize_ast, finalize_panns
from instrument_robustness.bundle_weights import EXTERNAL_WEIGHTS, HISTORICAL_EXTERNAL_WEIGHTS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalizationContractTests(unittest.TestCase):
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
