from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from instrument_robustness.config import (
    PIPE,
    ROOT,
    TARGET_LABELS,
    assert_fingerprint,
    config_fingerprint,
)
from instrument_robustness.extract_mert import choose_device, extract_mert_splits
from instrument_robustness.mert_data import (
    MERT_FEATURE_DIR,
    load_mert_embedding_metadata,
    load_mert_embeddings,
)
from instrument_robustness.train_mert import (
    DEFAULT_OUTPUT_DIR,
    class_weight_vector,
    predict,
    score,
    sha256,
    train_fixed_epochs,
)


FINAL_OUTPUT_NAMES = (
    "final_evaluation_status.json",
    "final_probe.pt",
    "test_confusion_matrix.csv",
    "test_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the validation-selected frozen MERT probe on train, "
            "extract the sealed test embeddings, and evaluate test exactly once."
        )
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--windows-csv", type=Path, default=PIPE / "windows.csv")
    parser.add_argument("--feature-dir", type=Path, default=MERT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.extraction_batch_size <= 0:
        parser.error("--extraction-batch-size must be greater than zero")
    return args


def write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def _validate_validation_run(
    summary: dict[str, object],
    *,
    summary_path: Path,
    feature_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    assert_fingerprint(summary.get("config_fingerprint"), str(summary_path))
    if summary.get("label_order") != TARGET_LABELS:
        raise ValueError("The MERT validation summary uses an unexpected label order")
    if summary.get("test_evaluated") is not False:
        raise ValueError("The MERT validation summary does not describe a sealed test set")
    if summary.get("selection_metric") != "validation_macro_f1":
        raise ValueError("MERT must be selected by validation macro-F1")

    for split in ("train", "val"):
        path = feature_dir / f"{split}.npz"
        recorded = summary.get("input_files", {}).get(split, {}).get("sha256")
        if recorded != sha256(path):
            raise ValueError(f"{split}.npz has changed since MERT model selection")

    expected_outputs = {
        "validation_search": output_dir / "validation_search.csv",
        "validation_confusion_matrix": output_dir / "validation_confusion_matrix.csv",
        "model": output_dir / "best_probe.pt",
    }
    for key, path in expected_outputs.items():
        recorded = summary.get("output_files", {}).get(key, {}).get("sha256")
        if recorded != sha256(path):
            raise ValueError(f"{path.name} has changed since MERT model selection")

    train_metadata = load_mert_embedding_metadata("train", feature_dir=feature_dir)
    val_metadata = load_mert_embedding_metadata("val", feature_dir=feature_dir)
    if train_metadata != val_metadata:
        raise ValueError("Train and validation MERT embeddings use different extractors")
    if summary.get("embedding_schema") != train_metadata:
        raise ValueError("MERT embedding metadata changed after validation selection")
    return train_metadata


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "validation_summary.json"
    status_path = output_dir / "final_evaluation_status.json"
    final_model_path = output_dir / "final_probe.pt"
    confusion_path = output_dir / "test_confusion_matrix.csv"
    test_summary_path = output_dir / "test_summary.json"
    test_feature_path = feature_dir / "test.npz"

    existing = [
        path
        for path in (
            test_feature_path,
            *(output_dir / name for name in FINAL_OUTPUT_NAMES),
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Final MERT evaluation has already started or completed; refusing "
            "another test access. Existing: "
            + ", ".join(str(path) for path in existing)
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run train_mert and freeze validation selection first."
        )

    with summary_path.open(encoding="utf-8") as file:
        validation_summary = json.load(file)
    embedding_metadata = _validate_validation_run(
        validation_summary,
        summary_path=summary_path,
        feature_dir=feature_dir,
        output_dir=output_dir,
    )

    X_train, y_train = load_mert_embeddings("train", feature_dir=feature_dir)
    X_val, y_val = load_mert_embeddings("val", feature_dir=feature_dir)
    # Fit on train only, NOT train+val. Validation is loaded to verify the embedding schema and
    # input hashes recorded during selection; it must not enter the fit. The CNN/CRNN/AST/PANNs
    # families cannot refit on train+val because their stopping rule is chosen on validation, so a
    # MERT probe trained on 7,119 windows would be compared against models trained on 5,861.
    # See train_mert's final_test_policy.
    X_final = X_train
    y_final = y_train
    class_weights = class_weight_vector(y_final)

    selected = validation_summary.get("best_config", {})
    learning_rate = float(selected["learning_rate"])
    batch_size = int(selected["batch_size"])
    epochs = int(selected["best_epoch"])
    seed = int(selected["seed"])

    try:
        import torch
        from instrument_robustness.mert_probe import MERTProbe
    except ImportError as error:
        raise RuntimeError(
            "MERT finalization requires PyTorch: pip install -e '.[mert]'"
        ) from error
    device = choose_device(args.device, torch)

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    test_access_count = 0
    test_evaluation_count = 0
    with status_path.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "state": "final_fit_started",
                "started_at_utc": started_at,
                "test_access_count": test_access_count,
                "test_evaluation_count": test_evaluation_count,
                "config_fingerprint": config_fingerprint(),
            },
            file,
            indent=2,
        )

    try:
        fit_started = perf_counter()
        model = train_fixed_epochs(
            X_final,
            y_final,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed,
            device=device,
            torch=torch,
            MERTProbe=MERTProbe,
            class_weights=class_weights,
        )
        fit_seconds = perf_counter() - fit_started
        torch.save(
            {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "num_classes": len(TARGET_LABELS),
                "label_order": TARGET_LABELS,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
                "layer_weights": model.layer_weights(),
                "class_weights": (
                    None if class_weights is None else class_weights.tolist()
                ),
                "embedding_schema": embedding_metadata,
                "config_fingerprint": config_fingerprint(),
            },
            final_model_path,
        )

        test_access_count = 1
        write_json(
            status_path,
            {
                "state": "extracting_test",
                "started_at_utc": started_at,
                "test_access_count": test_access_count,
                "test_evaluation_count": test_evaluation_count,
                "config_fingerprint": config_fingerprint(),
            },
        )
        extract_mert_splits(
            splits=("test",),
            data_root=Path(args.data_root),
            windows_csv=Path(args.windows_csv),
            output_dir=feature_dir,
            batch_size=args.extraction_batch_size,
            model_id=embedding_metadata["model_id"],
            revision=embedding_metadata["model_revision"],
            device=args.device,
            allow_test=True,
        )
        X_test, y_test = load_mert_embeddings("test", feature_dir=feature_dir)
        test_metadata = load_mert_embedding_metadata("test", feature_dir=feature_dir)
        if test_metadata != embedding_metadata:
            raise ValueError("Test MERT embeddings use a different extractor")

        predictions = predict(
            model,
            X_test,
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
        test_metrics = score(y_test, predictions)
        test_evaluation_count = 1

        confusion = pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=TARGET_LABELS,
            columns=TARGET_LABELS,
        )
        confusion.index.name = "actual"
        confusion.to_csv(confusion_path)

        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": config_fingerprint(),
            # Carried through from the validation summary, not hardcoded. summarize_results
            # gates on this field; without it a test summary is indistinguishable from one
            # written before the project standardised on macro-F1.
            "selection_metric": validation_summary.get("selection_metric"),
            "protocol": (
                "probe hyperparameters and epoch selected on validation; final probe "
                "fit on train only; test embeddings extracted and evaluated once"
            ),
            "model": "frozen MERT-v1-95M layer-weighted linear probe",
            "selected_config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
            },
            "model_fit_splits": ["train"],
            "backbone_frozen": True,
            "embedding_schema": embedding_metadata,
            "class_weights": (
                None
                if class_weights is None
                else {
                    label: float(class_weights[index])
                    for index, label in enumerate(TARGET_LABELS)
                }
            ),
            "training_examples": int(len(y_train)),
            "validation_examples": int(len(y_val)),
            "final_fit_examples": int(len(y_final)),
            "test_examples": int(len(y_test)),
            "fit_seconds": fit_seconds,
            "test_metrics": test_metrics,
            "label_order": TARGET_LABELS,
            "test_access_count": test_access_count,
            "test_evaluation_count": test_evaluation_count,
            "input_files": {
                split: {
                    "path": str((feature_dir / f"{split}.npz").resolve()),
                    "sha256": sha256(feature_dir / f"{split}.npz"),
                }
                for split in ("train", "val", "test")
            },
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "output_files": {
                "model": {
                    "path": str(final_model_path.resolve()),
                    "sha256": sha256(final_model_path),
                },
                "test_confusion_matrix": {
                    "path": str(confusion_path.resolve()),
                    "sha256": sha256(confusion_path),
                },
            },
        }
        write_json(test_summary_path, summary)
        write_json(
            status_path,
            {
                "state": "complete",
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_access_count": test_access_count,
                "test_evaluation_count": test_evaluation_count,
                "config_fingerprint": config_fingerprint(),
                "test_summary": {
                    "path": str(test_summary_path.resolve()),
                    "sha256": sha256(test_summary_path),
                },
            },
        )
        print("Test macro-F1:", test_metrics["macro_f1"])
        print(f"Saved final MERT outputs under {output_dir}")
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "started_at_utc": started_at,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "test_access_count": test_access_count,
                "test_evaluation_count": test_evaluation_count,
                "config_fingerprint": config_fingerprint(),
            },
        )
        raise


if __name__ == "__main__":
    main()
