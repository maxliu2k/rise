"""CNN data loader — reads the Step-7 CNN feature arrays unchanged.

Per-bin train-standardized log-mel, kept in the layout the convolutional stack expects:
(N, 1, n_mels=128, n_frames=130) = (batch, channel, freq, time).

This is the same array `crnn_data.load_crnn` reads; the only difference is layout. The CRNN
transposes to (batch, time, features) because it consumes time as a sequence. The CNN keeps the
2-D image shape and treats frequency and time as spatial axes.
"""
from __future__ import annotations

import numpy as np

from instrument_robustness.config import FEATURES, assert_serialized_fingerprint

CNN = FEATURES / "cnn"


def load_cnn(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split's CNN features.

    Preconditions: step7_featurize has written FEATURES/cnn/{split}.npz under the current config.
    Postcondition: returns (X, y) with X shaped (N, 1, 128, 130) float32 and y int64 class indices
    aligned to config.TARGET_LABELS.
    Raises: StaleArtifactError if the array carries no fingerprint or one that disagrees with the
    current config — a feature array built under a different label set or window length still
    loads and still trains, producing a plausible and meaningless model.
    """
    path = CNN / f"{split}.npz"
    with np.load(path, allow_pickle=True) as data:
        assert_serialized_fingerprint(
            data["config_fingerprint"] if "config_fingerprint" in data else None,
            str(path),
        )
        X = data["X"]                      # (N,128,130,1) as written by step7
        y = data["y"]
    X = np.transpose(X, (0, 3, 1, 2))      # -> (N,1,128,130) = (batch, channel, freq, time)
    return np.ascontiguousarray(X, dtype="float32"), y.astype("int64")
