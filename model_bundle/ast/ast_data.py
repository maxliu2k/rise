"""On-the-fly AST dataset and DataLoader for Step-5 normalized windows."""
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from instrument_robustness.config import (
    MAX_WINDOWS_PER_SOURCE,
    ROOT,
    SR,
    TARGET_LABELS,
    WINDOW_S,
    WINDOWS_CSV,
    assert_artifact_fingerprint,
)
from instrument_robustness.pretrained_extractors import ast_input, build_ast_extractor

WaveformTransform = Callable[[np.ndarray], np.ndarray]
SPLITS = frozenset(("train", "val", "test"))


def _validated_labels(labels: Sequence[str]) -> list[str]:
    validated = [str(label).strip() for label in labels]
    if not validated:
        raise ValueError("AST requires at least one instrument label")
    if any(not label for label in validated):
        raise ValueError("AST labels cannot be empty")
    if len(validated) != len(set(validated)):
        raise ValueError(f"Duplicate AST labels: {validated}")
    return validated


def resolve_ast_labels(
    manifest_path: Optional[Path] = None,
    requested_labels: Optional[Sequence[str]] = None,
) -> list[str]:
    """Require the configured label set and verify every class exists in every split."""
    path = Path(manifest_path or WINDOWS_CSV)
    assert_artifact_fingerprint(path, "step5_normalize")
    rows = pd.read_csv(path, usecols=["label", "split"])
    if rows.empty:
        raise ValueError(f"No windows found in {path}")
    if rows[["label", "split"]].isna().any().any():
        raise ValueError(f"Missing AST label or split value in {path}")

    unknown_splits = set(rows["split"]) - SPLITS
    if unknown_splits:
        raise ValueError(f"Unknown splits in {path}: {sorted(unknown_splits)}")

    label_names = _validated_labels(TARGET_LABELS if requested_labels is None else requested_labels)
    observed = set(rows["label"])
    expected = set(label_names)
    unexpected = observed - expected
    missing = expected - observed
    if unexpected or missing:
        raise ValueError(
            f"AST labels in {path} do not match the configured 12-class dataset; "
            f"unexpected={sorted(unexpected)}, missing={sorted(missing)}. "
            "Rebuild it with python -m instrument_robustness.run_pipeline."
        )

    if len(label_names) < 2:
        raise ValueError(f"AST classification requires at least two labels, got {label_names}")

    split_labels = rows.groupby("split")["label"].agg(set).to_dict()
    for split in sorted(SPLITS):
        missing = set(label_names) - split_labels.get(split, set())
        if missing:
            raise ValueError(f"Split {split!r} is missing AST labels: {sorted(missing)}")
    return label_names


def validate_ast_window_files(
    manifest_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> int:
    """Validate the canonical AST manifest and every referenced window before model download."""
    path = Path(manifest_path or WINDOWS_CSV)
    data_root = Path(root or ROOT)
    assert_artifact_fingerprint(path, "step5_normalize")
    required = ["window_path", "source_path", "start_time"]
    try:
        rows = pd.read_csv(path, usecols=required)
    except ValueError as error:
        raise ValueError(
            f"{path} is not a canonical window manifest; expected columns {required}. "
            "Rebuild it with python -m instrument_robustness.run_pipeline."
        ) from error
    if rows.empty:
        raise ValueError(f"No AST windows found in {path}")
    if rows[required].isna().any().any():
        raise ValueError(f"Missing canonical AST window metadata in {path}")
    if rows["window_path"].duplicated().any():
        raise ValueError(f"Duplicate AST window paths found in {path}")

    source_counts = rows["source_path"].value_counts()
    excessive = source_counts[source_counts > MAX_WINDOWS_PER_SOURCE]
    if not excessive.empty:
        preview = ", ".join(
            f"{source}={count}" for source, count in excessive.head(10).items()
        )
        raise ValueError(
            f"{path} has sources exceeding MAX_WINDOWS_PER_SOURCE="
            f"{MAX_WINDOWS_PER_SOURCE}: {preview}. "
            "Rebuild it with python -m instrument_robustness.run_pipeline."
        )

    start_times = pd.to_numeric(rows["start_time"], errors="coerce")
    if start_times.isna().any():
        raise ValueError(f"Invalid AST start_time values found in {path}")
    if MAX_WINDOWS_PER_SOURCE == 1 and not np.allclose(start_times.to_numpy(), 0.0):
        raise ValueError(
            f"{path} is not the onset-aligned one-window dataset: nonzero start_time found. "
            "Rebuild it with python -m instrument_robustness.run_pipeline."
        )

    missing = [
        data_root / window_path
        for window_path in rows["window_path"]
        if not (data_root / window_path).is_file()
    ]
    if missing:
        preview = "\n".join(f"  {missing_path}" for missing_path in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} AST window file(s) listed in {path} are missing:\n"
            f"{preview}{suffix}"
        )
    return len(rows)


def _load_window(path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SR:
        raise ValueError(f"Expected {SR} Hz audio at {path}, got {sample_rate} Hz")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio at {path}, got shape {waveform.shape}")

    target_samples = int(round(WINDOW_S * SR))
    if waveform.size != target_samples:
        raise ValueError(
            f"Expected exactly {target_samples} samples at {path}, got {waveform.size}. "
            "Rebuild with python -m instrument_robustness.run_pipeline --from step4_window."
        )
    return waveform


class ASTWindowDataset(Dataset):
    """Step-5 windows transformed by AST's pretrained feature extractor at access time."""

    def __init__(
        self,
        split: str,
        extractor=None,
        waveform_transform: Optional[WaveformTransform] = None,
        label_names: Optional[Sequence[str]] = None,
        manifest_path: Optional[Path] = None,
        root: Optional[Path] = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLITS)}")

        path = Path(manifest_path or WINDOWS_CSV)
        self.label_names = resolve_ast_labels(path, label_names)
        label_to_index = {label: index for index, label in enumerate(self.label_names)}

        rows = pd.read_csv(path)
        rows = rows.loc[rows["split"] == split].reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"No windows found for split {split!r} in {path}")

        unknown_labels = set(rows["label"]) - set(label_to_index)
        if unknown_labels:
            raise ValueError(f"Unknown labels in windows manifest: {sorted(unknown_labels)}")

        data_root = Path(root or ROOT)
        self.paths = [data_root / window_path for window_path in rows["window_path"]]
        self.labels = [label_to_index[label] for label in rows["label"]]
        self.class_counts = {
            label: self.labels.count(index) for index, label in enumerate(self.label_names)
        }
        self.extractor = build_ast_extractor() if extractor is None else extractor
        self.waveform_transform = waveform_transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        waveform = _load_window(self.paths[index])
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(waveform)
        waveform = np.asarray(waveform, dtype=np.float32)

        input_values = ast_input(waveform, self.extractor).squeeze(0)
        return {
            "input_values": input_values,
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def make_ast_dataloader(
    split: str,
    *,
    batch_size: int,
    extractor=None,
    waveform_transform: Optional[WaveformTransform] = None,
    shuffle: Optional[bool] = None,
    pin_memory: bool = False,
    label_names: Optional[Sequence[str]] = None,
    manifest_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> DataLoader:
    """Build a loader that keeps AST extraction in the main process and off disk."""
    dataset = ASTWindowDataset(
        split,
        extractor=extractor,
        waveform_transform=waveform_transform,
        label_names=label_names,
        manifest_path=manifest_path,
        root=root,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" if shuffle is None else shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )
