from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn

from instrument_robustness.svm_model import (
    SVM_FEATURE_DIR,
    SVMConfig,
    TARGET_LABELS,
    build_svm,
    load_svm_feature_names,
    load_svm_split,
    save_svm,
)
from instrument_robustness.train_svm import DEFAULT_OUTPUT_DIR, evaluate, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the validation-selected RBF SVC on train+val and evaluate "
            "test exactly once. No feature scaling or tuning is performed."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=SVM_FEATURE_DIR,
        help="Directory containing train.npz, val.npz, and test.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing validation_summary.json and final outputs.",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    validation_summary_path = output_dir / "validation_summary.json"
    status_path = output_dir / "final_evaluation_status.json"
    model_path = output_dir / "final_model.joblib"
    confusion_path = output_dir / "test_confusion_matrix.csv"
    test_summary_path = output_dir / "test_summary.json"

    final_paths = [status_path, model_path, confusion_path, test_summary_path]
    existing_paths = [path for path in final_paths if path.exists()]
    if existing_paths:
        names = ", ".join(path.name for path in existing_paths)
        raise FileExistsError(
            "Final SVM evaluation has already started or completed; "
            f"refusing to evaluate test again. Existing: {names}"
        )

    if not validation_summary_path.exists():
        raise FileNotFoundError(
            f"Missing {validation_summary_path}. Run the validation search first."
        )

    with validation_summary_path.open(encoding="utf-8") as file:
        validation_summary = json.load(file)

    if validation_summary.get("test_evaluated") is not False:
        raise ValueError("The validation summary does not describe a sealed test set")
    if validation_summary.get("label_order") != TARGET_LABELS:
        raise ValueError("The validation summary uses an unexpected label order")

    selected = validation_summary.get("best_config", {})
    if selected.get("kernel") != "rbf":
        raise ValueError("The validation-selected model is not an RBF SVC")
    config = SVMConfig(C=float(selected["C"]), gamma=selected["gamma"])

    X_train, y_train = load_svm_split("train", feature_dir=feature_dir)
    X_val, y_val = load_svm_split("val", feature_dir=feature_dir)
    train_path = feature_dir / "train.npz"
    val_path = feature_dir / "val.npz"
    test_path = feature_dir / "test.npz"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}")

    recorded_inputs = validation_summary.get("input_files", {})
    for split, path in (("train", train_path), ("val", val_path)):
        recorded_hash = recorded_inputs.get(split, {}).get("sha256")
        if recorded_hash != sha256(path):
            raise ValueError(
                f"{split}.npz has changed since validation model selection"
            )

    train_feature_names = load_svm_feature_names(
        "train", feature_dir=feature_dir
    )
    val_feature_names = load_svm_feature_names("val", feature_dir=feature_dir)
    if train_feature_names != val_feature_names:
        raise ValueError("train.npz and val.npz use different feature orders")

    expected_labels = set(range(len(TARGET_LABELS)))
    if set(np.concatenate([y_train, y_val]).tolist()) != expected_labels:
        raise ValueError("The final development data does not contain all nine classes")

    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)
    model = build_svm(config)
    fit_started = perf_counter()
    model.fit(X_final, y_final)
    fit_seconds = perf_counter() - fit_started

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    with status_path.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "state": "started",
                "started_at_utc": started_at,
                "validation_summary": str(validation_summary_path.resolve()),
            },
            file,
            indent=2,
        )

    try:
        X_test, y_test = load_svm_split("test", feature_dir=feature_dir)
        test_metrics = evaluate(model, X_test, y_test)

        confusion_frame = pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=TARGET_LABELS,
            columns=TARGET_LABELS,
        )
        confusion_frame.index.name = "actual"
        confusion_frame.to_csv(confusion_path)
        save_svm(model, model_path)

        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "hyperparameters selected on validation; final model fit on "
                "train+val; test evaluated once"
            ),
            "selected_config": {
                "kernel": "rbf",
                "C": config.C,
                "gamma": config.gamma,
            },
            "model_fit_splits": ["train", "val"],
            "feature_preprocessing": (
                "existing train-statistics-standardized features reused; "
                "no scaler refit"
            ),
            "training_examples": int(len(y_train)),
            "validation_examples": int(len(y_val)),
            "final_fit_examples": int(len(y_final)),
            "test_examples": int(len(y_test)),
            "fit_seconds": fit_seconds,
            "test_metrics": test_metrics,
            "label_order": TARGET_LABELS,
            "feature_schema": validation_summary["feature_schema"],
            "input_files": {
                "train": {
                    "path": str(train_path.resolve()),
                    "sha256": sha256(train_path),
                },
                "val": {
                    "path": str(val_path.resolve()),
                    "sha256": sha256(val_path),
                },
                "test": {
                    "path": str(test_path.resolve()),
                    "sha256": sha256(test_path),
                    "X_shape": list(X_test.shape),
                    "X_dtype": str(X_test.dtype),
                    "y_shape": list(y_test.shape),
                    "y_dtype": str(y_test.dtype),
                },
                "validation_summary": {
                    "path": str(validation_summary_path.resolve()),
                    "sha256": sha256(validation_summary_path),
                },
            },
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "output_files": {
                "model": {
                    "path": str(model_path.resolve()),
                    "sha256": sha256(model_path),
                },
                "test_confusion_matrix": {
                    "path": str(confusion_path.resolve()),
                    "sha256": sha256(confusion_path),
                },
            },
            "test_evaluated": True,
            "test_evaluation_count": 1,
        }
        write_json(test_summary_path, summary)

        write_json(
            status_path,
            {
                "state": "complete",
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_evaluation_count": 1,
                "test_summary": {
                    "path": str(test_summary_path.resolve()),
                    "sha256": sha256(test_summary_path),
                },
            },
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "started_at_utc": started_at,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "test_evaluation_count": 1,
            },
        )
        raise

    print("Final configuration:")
    print(summary["selected_config"])
    print("Test macro-F1:", test_metrics["macro_f1"])
    print("Test accuracy:", test_metrics["accuracy"])
    print(f"Saved final outputs under {output_dir}")


if __name__ == "__main__":
    main()
