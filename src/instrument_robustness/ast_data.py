"""On-the-fly AST dataset and DataLoader for Step-5 normalized windows."""
from typing import Callable, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from instrument_robustness.config import PIPE, ROOT, SR, TARGET_LABELS, WINDOW_S
from instrument_robustness.pretrained_extractors import ast_input, build_ast_extractor

WaveformTransform = Callable[[np.ndarray], np.ndarray]
LABEL2IDX = {label: index for index, label in enumerate(TARGET_LABELS)}
SPLITS = frozenset(("train", "val", "test"))


def _load_window(path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SR:
        raise ValueError(f"Expected {SR} Hz audio at {path}, got {sample_rate} Hz")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio at {path}, got shape {waveform.shape}")

    target_samples = int(round(WINDOW_S * SR))
    if waveform.size < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.size))
    return waveform[:target_samples]


class ASTWindowDataset(Dataset):
    """Step-5 windows transformed by AST's pretrained feature extractor at access time."""

    def __init__(
        self,
        split: str,
        extractor=None,
        waveform_transform: Optional[WaveformTransform] = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLITS)}")

        rows = pd.read_csv(PIPE / "windows.csv")
        rows = rows.loc[rows["split"] == split].reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"No windows found for split {split!r} in {PIPE / 'windows.csv'}")

        unknown_labels = set(rows["label"]) - set(LABEL2IDX)
        if unknown_labels:
            raise ValueError(f"Unknown labels in windows manifest: {sorted(unknown_labels)}")

        self.paths = [ROOT / path for path in rows["window_path"]]
        self.labels = [LABEL2IDX[label] for label in rows["label"]]
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
) -> DataLoader:
    """Build a loader that keeps AST extraction in the main process and off disk."""
    dataset = ASTWindowDataset(
        split,
        extractor=extractor,
        waveform_transform=waveform_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" if shuffle is None else shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )
