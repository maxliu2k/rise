"""Perform the single permitted AST test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from instrument_robustness.config import ARTIFACTS, TARGET_LABELS, WINDOWS_CSV, assert_fingerprint

FINAL_FILES = (
    "final_evaluation_status.json",
    "test_summary.json",
    "test_summary.csv",
    "test_confusion_matrix.csv",
    "test_by_instrument.csv",
    "test_by_family.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_weights(output_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = output_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No AST weights found under {output_dir}")


def validate_selection(output_dir: Path) -> tuple[dict, Path]:
    summary_path = output_dir / "validation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_evaluated") is not False:
        raise ValueError("The AST validation summary does not describe a sealed test set")
    if summary.get("selection_metric") != "validation_macro_f1":
        raise ValueError("AST must be selected by validation macro-F1")
    if summary.get("label_order") != list(TARGET_LABELS):
        raise ValueError("AST label order does not match the fixed project order")
    assert_fingerprint(summary.get("config_fingerprint"), str(summary_path))
    if summary.get("windows_manifest", {}).get("sha256") != sha256(WINDOWS_CSV):
        raise ValueError("AST validation used a different windows.csv")
    weights = selected_weights(output_dir)
    if summary.get("output_files", {}).get("model", {}).get("sha256") != sha256(weights):
        raise ValueError("AST weights do not match validation_summary.json")
    return summary, weights


def claim_finalization(output_dir: Path) -> Path:
    existing = [name for name in FINAL_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing another AST test access; final artifacts already exist: "
            + ", ".join(existing)
        )
    path = output_dir / "final_evaluation_status.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"state": "claimed", "test_access_count": 0, "test_evaluation_count": 0}, handle, indent=2)
        handle.write("\n")
    return path


def _write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def finalize(output_dir: Path, *, batch_size: int, device_name: str | None) -> dict:
    validation, weights = validate_selection(output_dir)
    status_path = claim_finalization(output_dir)
    test_access_count = 0
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        from instrument_robustness.ast_data import make_ast_dataloader
        from instrument_robustness.train_ast import _run_epoch, _write_test_reports

        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        extractor = ASTFeatureExtractor.from_pretrained(output_dir)
        model = ASTForAudioClassification.from_pretrained(output_dir).to(device)
        assert_fingerprint(
            getattr(model.config, "instrument_robustness_fingerprint", None),
            str(output_dir),
        )
        if getattr(model.config, "instrument_robustness_windows_sha256", None) != sha256(WINDOWS_CSV):
            raise ValueError("AST checkpoint records a different windows.csv")

        test_access_count = 1
        _write_status(status_path, {"state": "evaluating_test", "test_access_count": test_access_count, "test_evaluation_count": 0})
        loader = make_ast_dataloader(
            "test",
            batch_size=batch_size,
            extractor=extractor,
            pin_memory=device.type == "cuda",
            label_names=TARGET_LABELS,
            manifest_path=WINDOWS_CSV,
            shuffle=False,
        )
        metrics, true, predicted = _run_epoch(
            model,
            loader,
            device,
            len(TARGET_LABELS),
            collect_predictions=True,
            phase="test",
        )
        reports = _write_test_reports(output_dir, true, predicted, TARGET_LABELS)
        summary = {
            "protocol": "validation-selected AST checkpoint evaluated on test exactly once",
            "model": validation["model"],
            "selection_metric": "validation_macro_f1",
            "best_epoch": validation["best_epoch"],
            "label_order": list(TARGET_LABELS),
            "config_fingerprint": validation["config_fingerprint"],
            "test_examples": int(len(loader.dataset)),
            "test_metrics": metrics,
            "per_instrument": reports["per_instrument"],
            "per_family": reports["per_family"],
            "test_access_count": 1,
            "test_evaluation_count": 1,
            "output_files": {"model": {"path": str(weights), "sha256": sha256(weights)}},
        }
        summary_path = output_dir / "test_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _write_status(status_path, {
            "state": "complete",
            "test_access_count": 1,
            "test_evaluation_count": 1,
            "test_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        })
        return summary
    except Exception as error:
        _write_status(status_path, {"state": "failed", "test_access_count": test_access_count, "test_evaluation_count": 0, "error": repr(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "ast")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = finalize(args.output_dir, batch_size=args.batch_size, device_name=args.device)
    print("AST test macro-F1:", result["test_metrics"]["macro_f1"])


if __name__ == "__main__":
    main()
