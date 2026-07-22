from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from instrument_robustness.config import FEATURES, PIPE, ROOT, TARGET_LABELS


MERT_FEATURE_DIR = FEATURES / "mert"
MERT_NUM_LAYERS = 13
MERT_HIDDEN_SIZE = 768
LABEL_TO_INDEX = {label: index for index, label in enumerate(TARGET_LABELS)}


@dataclass(frozen=True)
class MERTExample:
    window_path: Path
    window_relative_path: str
    source_path: str
    label: str
    target: int


def load_mert_examples(
    split: str,
    *,
    windows_csv: str | Path = PIPE / "windows.csv",
    data_root: str | Path = ROOT,
) -> list[MERTExample]:
    """Load one authoritative split from windows.csv."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    frame = pd.read_csv(windows_csv)
    required = {"window_path", "source_path", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"windows.csv is missing columns: {sorted(missing)}")

    rows = frame.loc[frame["split"] == split]
    if rows.empty:
        raise ValueError(f"windows.csv contains no {split!r} examples")

    unexpected = sorted(set(rows["label"]) - set(TARGET_LABELS))
    if unexpected:
        raise ValueError(f"Unexpected labels in {split}: {unexpected}")

    root = Path(data_root)
    examples = [
        MERTExample(
            window_path=root / row.window_path,
            window_relative_path=str(row.window_path),
            source_path=str(row.source_path),
            label=str(row.label),
            target=LABEL_TO_INDEX[row.label],
        )
        for row in rows.itertuples(index=False)
    ]

    missing_paths = [
        example.window_path
        for example in examples
        if not example.window_path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} window files; first missing: "
            f"{missing_paths[0]}. MERT requires the full windowed-audio download."
        )

    return examples


def load_mert_embeddings(
    split: str,
    *,
    feature_dir: str | Path = MERT_FEATURE_DIR,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load cached time-pooled hidden states for the frozen MERT probe."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    path = Path(feature_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run extract_mert first.")

    with np.load(path) as data:
        if "X" not in data or "y" not in data:
            raise KeyError(f"{path} must contain arrays named X and y")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        if "label_names" in data:
            labels = data["label_names"].astype(str).tolist()
            if labels != TARGET_LABELS:
                raise ValueError(f"Unexpected label order in {path}: {labels}")

    expected_tail = (MERT_NUM_LAYERS, MERT_HIDDEN_SIZE)
    if X.ndim != 3 or X.shape[1:] != expected_tail:
        raise ValueError(
            f"Expected X shape (N, {MERT_NUM_LAYERS}, {MERT_HIDDEN_SIZE}), "
            f"received {X.shape}"
        )
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError(f"Incompatible X and y shapes: {X.shape}, {y.shape}")
    if len(X) == 0:
        raise ValueError(f"{path} contains no examples")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{path} contains NaN or infinite values")

    invalid = sorted(set(y.tolist()) - set(range(len(TARGET_LABELS))))
    if invalid:
        raise ValueError(f"Unexpected labels in {path}: {invalid}")
    missing_labels = sorted(set(range(len(TARGET_LABELS))) - set(y.tolist()))
    if missing_labels:
        raise ValueError(f"Missing labels in {path}: {missing_labels}")

    return X, y


def load_mert_embedding_metadata(
    split: str,
    *,
    feature_dir: str | Path = MERT_FEATURE_DIR,
) -> dict[str, str]:
    path = Path(feature_dir) / f"{split}.npz"
    with np.load(path) as data:
        required = ("model_id", "model_revision", "pooling")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path} is missing MERT metadata: {missing}")
        return {key: str(data[key].item()) for key in required}
