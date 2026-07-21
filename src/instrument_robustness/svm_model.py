from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.svm import SVC

from instrument_robustness.config import FEATURES, TARGET_LABELS


SVM_FEATURE_DIR = FEATURES / "svm"


@dataclass(frozen=True)
class SVMConfig:
    """Hyperparameters for one SVM candidate."""

    C: float = 1.0
    gamma: str | float = "scale"


def load_svm_split(
    split: str,
    *,
    feature_dir: str | Path = SVM_FEATURE_DIR,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load an already processed SVM feature split."""

    if split not in {"train", "val", "test"}:
        raise ValueError(
            "split must be one of: train, val, test"
        )

    path = Path(feature_dir) / f"{split}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run "
            "`python download_data.py --features-only` first."
        )

    with np.load(path) as data:
        if "X" not in data or "y" not in data:
            raise KeyError(
                f"{path} must contain arrays named 'X' and 'y'. "
                f"Found: {data.files}"
            )

        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)

        if "label_names" in data:
            label_names = data["label_names"].tolist()
            if label_names != TARGET_LABELS:
                raise ValueError(
                    f"Unexpected label order in {path}: {label_names}. "
                    f"Expected: {TARGET_LABELS}"
                )

    if X.ndim != 2 or X.shape[1] != 88:
        raise ValueError(
            f"Expected X shape (N, 88), received {X.shape}"
        )

    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(
            f"Incompatible X and y shapes: {X.shape}, {y.shape}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{path} contains NaN or infinite values.")

    if len(X) == 0:
        raise ValueError(f"{path} contains no examples.")

    invalid_labels = sorted(
        set(y.tolist()) - set(range(len(TARGET_LABELS)))
    )

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in {path}: {invalid_labels}"
        )

    return X, y


def load_svm_feature_names(
    split: str,
    *,
    feature_dir: str | Path = SVM_FEATURE_DIR,
) -> list[str] | None:
    """Return the saved feature order when the NPZ includes it."""

    path = Path(feature_dir) / f"{split}.npz"
    with np.load(path) as data:
        if "feature_names" not in data:
            return None
        return data["feature_names"].astype(str).tolist()


def build_svm(config: SVMConfig) -> SVC:
    """Create an SVM classifier.

    The feature arrays are already standardized using training
    statistics, so no additional StandardScaler is applied.
    """

    return SVC(
        kernel="rbf",
        C=config.C,
        gamma=config.gamma,
        cache_size=2_000,
    )


def save_svm(model: SVC, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def load_svm(path: str | Path) -> SVC:
    model: Any = joblib.load(path)
    if not isinstance(model, SVC):
        raise TypeError(f"Expected an sklearn SVC in {path}, got {type(model)!r}")
    return model
