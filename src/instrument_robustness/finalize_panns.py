"""Perform the single permitted PANNs test evaluation.

Training writes a validation-selected checkpoint without reading test. This command verifies that
sealed record, claims the finalization slot before loading test, and evaluates the selected model
once. A failed attempt leaves a status record and must be investigated rather than silently rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from instrument_robustness.config import ARTIFACTS, TARGET_LABELS, WINDOWS_CSV, assert_fingerprint

FINAL_FILES = ("final_evaluation_status.json", "test_summary.json", "test_confusion_matrix.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selection(output_dir: Path) -> tuple[dict, Path]:
    summary_path = output_dir / "validation_summary.json"
    checkpoint_path = output_dir / "selected_model.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("PANNs validation_summary.json and selected_model.pt are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_evaluated") is not False:
        raise ValueError("The PANNs validation summary does not describe a sealed test set")
    if summary.get("selection_metric") != "validation_macro_f1":
        raise ValueError("PANNs must be selected by validation macro-F1")
    if summary.get("label_order") != list(TARGET_LABELS):
        raise ValueError("PANNs label order does not match the fixed project label order")
    assert_fingerprint(summary.get("config_fingerprint"), str(summary_path))
    if summary.get("windows_manifest", {}).get("sha256") != sha256(WINDOWS_CSV):
        raise ValueError("PANNs validation used a different windows.csv")
    expected = summary.get("output_files", {}).get("model", {}).get("sha256")
    if expected != sha256(checkpoint_path):
        raise ValueError("selected_model.pt does not match validation_summary.json")
    return summary, checkpoint_path


def claim_finalization(output_dir: Path) -> Path:
    existing = [name for name in FINAL_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing another PANNs test access; final artifacts already exist: "
            + ", ".join(existing)
        )
    status_path = output_dir / "final_evaluation_status.json"
    with status_path.open("x", encoding="utf-8") as handle:
        json.dump({"state": "claimed", "test_access_count": 0, "test_evaluation_count": 0}, handle, indent=2)
        handle.write("\n")
    return status_path


def _write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def finalize(output_dir: Path, *, batch_size: int, num_workers: int, device_name: str | None) -> dict:
    validation, checkpoint_path = validate_selection(output_dir)
    status_path = claim_finalization(output_dir)
    test_access_count = 0
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader

        from instrument_robustness.train_panns import (
            PannsClassifier,
            CKPT,
            WindowWaveformDataset,
            build_backbone,
            get_device,
            load_split,
            predict_full,
            report,
        )

        device = device_name or get_device()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("label_order") != list(TARGET_LABELS):
            raise ValueError("PANNs checkpoint label order is invalid")
        assert_fingerprint(checkpoint.get("config_fingerprint"), str(checkpoint_path))
        base_hash = validation.get("base_checkpoint", {}).get("sha256")
        if checkpoint.get("base_checkpoint_sha256") != base_hash:
            raise ValueError("PANNs base-checkpoint identity disagrees with validation summary")
        if sha256(CKPT) != base_hash:
            raise ValueError("The installed PANNs CNN14 base checkpoint differs from validation")

        model = PannsClassifier(build_backbone(), freeze=checkpoint["mode"] == "probe")
        if checkpoint["mode"] == "probe":
            model.head.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)

        test_access_count = 1
        _write_status(status_path, {"state": "evaluating_test", "test_access_count": test_access_count, "test_evaluation_count": 0})
        test_frame = load_split("test")
        loader = DataLoader(WindowWaveformDataset(test_frame), batch_size=batch_size, num_workers=num_workers)
        true, predicted = predict_full(model, loader, device)
        metrics = report(true, predicted)

        confusion = pd.DataFrame(
            np.asarray(metrics["confusion_matrix"]),
            index=TARGET_LABELS,
            columns=TARGET_LABELS,
        )
        confusion.index.name = "actual"
        confusion_path = output_dir / "test_confusion_matrix.csv"
        confusion.to_csv(confusion_path)
        summary = {
            "protocol": "validation-selected PANNs checkpoint evaluated on test exactly once",
            "model": validation["model"],
            "mode": validation["mode"],
            "selection_metric": "validation_macro_f1",
            "label_order": list(TARGET_LABELS),
            "config_fingerprint": validation["config_fingerprint"],
            "test_examples": int(len(true)),
            "test_metrics": metrics,
            "test_access_count": 1,
            "test_evaluation_count": 1,
            "output_files": {
                "model": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
                "confusion_matrix": {"path": str(confusion_path), "sha256": sha256(confusion_path)},
            },
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
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "panns")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = finalize(args.output_dir, batch_size=args.batch_size, num_workers=args.num_workers, device_name=args.device)
    print("PANNs test macro-F1:", result["test_metrics"]["macro_f1"])


if __name__ == "__main__":
    main()
