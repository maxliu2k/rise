from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import SVC

from instrument_robustness.svm_model import (
    SVM_FEATURE_DIR,
    SVMConfig,
    TARGET_LABELS,
    build_svm,
    load_svm_feature_names,
    load_svm_split,
    save_svm,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "svm"
FINAL_TEST_POLICY = {
    "parameter_selection": "train_fit_validation_macro_f1",
    "final_fit_splits": ["train", "val"],
    "feature_preprocessing": (
        "reuse the existing train-statistics-standardized arrays; "
        "do not fit another scaler"
    ),
    "test_evaluations_allowed": 1,
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def gamma_value(value: str) -> str | float:
    if value in {"scale", "auto"}:
        return value
    return positive_float(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_configs(
    *,
    c_values: list[float],
    gamma_values: list[str | float],
) -> list[SVMConfig]:
    """Return the Cartesian product of the requested RBF parameters."""

    return [
        SVMConfig(C=C, gamma=gamma)
        for C, gamma in itertools.product(
            c_values,
            gamma_values,
        )
    ]


def evaluate(
    model: SVC,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
) -> dict[str, object]:
    predictions = model.predict(X)

    return {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=list(range(len(TARGET_LABELS))),
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=list(range(len(TARGET_LABELS))),
            target_names=TARGET_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune an RBF SVC on the fixed train/validation feature splits. "
            "This command never loads or evaluates test.npz."
        )
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=SVM_FEATURE_DIR,
        help="Directory containing train.npz and val.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the CSV, JSON summary, and best model.",
    )
    parser.add_argument(
        "--C-values",
        dest="c_values",
        nargs="+",
        type=positive_float,
        default=[0.1, 1.0, 10.0, 100.0],
        metavar="C",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="+",
        type=gamma_value,
        default=["scale", 0.001, 0.01, 0.1],
        metavar="GAMMA",
        help="Values used by non-linear kernels; each must be scale, auto, or > 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train, y_train = load_svm_split(
        "train",
        feature_dir=args.feature_dir,
    )
    X_val, y_val = load_svm_split(
        "val",
        feature_dir=args.feature_dir,
    )
    train_feature_names = load_svm_feature_names(
        "train",
        feature_dir=args.feature_dir,
    )
    val_feature_names = load_svm_feature_names(
        "val",
        feature_dir=args.feature_dir,
    )
    if (
        train_feature_names is not None
        and val_feature_names is not None
        and train_feature_names != val_feature_names
    ):
        raise ValueError("train.npz and val.npz use different feature orders")
    feature_names = train_feature_names or val_feature_names

    results: list[dict[str, object]] = []
    candidates = candidate_configs(
        c_values=args.c_values,
        gamma_values=args.gamma_values,
    )

    for run_number, config in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"[{run_number}/{len(candidates)}] "
            f"kernel=rbf, C={config.C}, gamma={config.gamma}"
        )

        model = build_svm(config)
        fit_started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_started

        metrics = evaluate(
            model,
            X_val,
            y_val,
        )

        results.append(
            {
                "candidate": run_number,
                "kernel": "rbf",
                "C": config.C,
                "gamma": config.gamma,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_weighted_f1": metrics["weighted_f1"],
                "support_vectors": int(
                    model.n_support_.sum()
                ),
                "fit_seconds": fit_seconds,
            }
        )

    results_frame = pd.DataFrame(results).sort_values(
        by=[
            "validation_macro_f1",
            "validation_accuracy",
            "support_vectors",
            "candidate",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    results_frame.insert(0, "rank", np.arange(1, len(results_frame) + 1))

    best_row = results_frame.iloc[0]

    if isinstance(best_row["gamma"], str):
        best_gamma = best_row["gamma"]
    else:
        best_gamma = float(best_row["gamma"])

    best_config = SVMConfig(
        C=float(best_row["C"]),
        gamma=best_gamma,
    )

    # Refit the selected configuration on training data only.
    # The validation set remains an honest selection set.
    best_model = build_svm(best_config)
    best_model.fit(X_train, y_train)

    best_metrics = evaluate(
        best_model,
        X_val,
        y_val,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_frame.to_csv(
        output_dir / "validation_search.csv",
        index=False,
    )

    confusion_frame = pd.DataFrame(
        best_metrics["confusion_matrix"],
        index=TARGET_LABELS,
        columns=TARGET_LABELS,
    )
    confusion_frame.index.name = "actual"
    confusion_frame.to_csv(
        output_dir / "validation_confusion_matrix.csv"
    )

    save_svm(
        best_model,
        output_dir / "best_model.joblib",
    )

    feature_dir = Path(args.feature_dir)
    train_path = feature_dir / "train.npz"
    val_path = feature_dir / "val.npz"
    search_path = output_dir / "validation_search.csv"
    confusion_path = output_dir / "validation_confusion_matrix.csv"
    model_path = output_dir / "best_model.joblib"

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "validation_macro_f1",
        "best_config": {
            "kernel": "rbf",
            "C": best_config.C,
            "gamma": best_config.gamma,
        },
        "validation_metrics": best_metrics,
        "label_order": TARGET_LABELS,
        "training_examples": int(len(y_train)),
        "validation_examples": int(len(y_val)),
        "candidates_evaluated": len(candidates),
        "search_space": {
            "C": args.c_values,
            "gamma": args.gamma_values,
        },
        "model_fit_splits": ["train"],
        "feature_schema": {
            "dimension": int(X_train.shape[1]),
            "names": feature_names,
            "label_order": TARGET_LABELS,
            "standardization": "training-set statistics applied by step7_featurize",
        },
        "input_files": {
            "train": {
                "path": str(train_path.resolve()),
                "sha256": sha256(train_path),
                "X_shape": list(X_train.shape),
                "X_dtype": str(X_train.dtype),
                "y_shape": list(y_train.shape),
                "y_dtype": str(y_train.dtype),
            },
            "val": {
                "path": str(val_path.resolve()),
                "sha256": sha256(val_path),
                "X_shape": list(X_val.shape),
                "X_dtype": str(X_val.dtype),
                "y_shape": list(y_val.shape),
                "y_dtype": str(y_val.dtype),
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
            "validation_search": {
                "path": str(search_path.resolve()),
                "sha256": sha256(search_path),
            },
            "validation_confusion_matrix": {
                "path": str(confusion_path.resolve()),
                "sha256": sha256(confusion_path),
            },
            "model": {
                "path": str(model_path.resolve()),
                "sha256": sha256(model_path),
            },
        },
        "final_test_policy": FINAL_TEST_POLICY,
        "test_evaluated": False,
    }

    with (
        output_dir / "validation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nBest configuration:")
    print(summary["best_config"])
    print(
        "Validation macro-F1:",
        best_metrics["macro_f1"],
    )
    print(
        f"Saved outputs under {output_dir}"
    )


if __name__ == "__main__":
    main()
