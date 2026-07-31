"""Rebuild the CNN/CRNN log-mel input from a waveform, clean or noisy.

Separate from noise_eval_cnn, and deliberately free of any torch import, for two reasons: the
representation path is not a torch concern, and keeping it importable without the optional
pretrained extras means it can be unit-tested anywhere the core dependencies are installed.

THE INVARIANT THIS MODULE EXISTS TO HOLD. `features/cnn/{split}.npz` is the CLEAN log-mel, frozen
at Step 7. A noise evaluator that reads it would report the clean score under a noisy label. So the
noisy waveform has to travel the identical path again:

    waveform -> featurelib.logmel -> (L - mu_train) / sigma_train -> (1, n_mels, n_frames)

`featurelib.logmel` is the same function Step 6 and Step 7 call -- not a reimplementation -- which
is what makes the noisy features comparable to the clean ones. The statistics are LOADED from
norm_stats.npz and never recomputed: they belong to the trained model's contract, and refitting them
on noisy audio would rescale the very distortion being measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from instrument_robustness.config import (
    N_FRAMES,
    N_MELS,
    STATS_NPZ,
    assert_serialized_fingerprint,
)
from instrument_robustness.featurelib import logmel


def load_logmel_statistics(path: str | Path = STATS_NPZ) -> tuple[np.ndarray, np.ndarray]:
    """Load the Step-6 per-mel-bin train statistics, shaped to broadcast over frames.

    Postcondition: returns (mean, std), each (N_MELS, 1) float32, ready to broadcast against a
    (N_MELS, N_FRAMES) log-mel matrix.
    Raises: StaleArtifactError if the bundle was built under a different config; ValueError if it
    was not computed on train alone, or its shape or values are unusable.
    """
    path = Path(path)
    with np.load(path) as data:
        assert_serialized_fingerprint(
            data["config_fingerprint"] if "config_fingerprint" in data else None,
            str(path),
        )
        if data["computed_on"].item() != "train":
            raise ValueError(f"{path} was not computed from train only")
        mean = np.asarray(data["logmel_mean"], dtype=np.float32)
        std = np.asarray(data["logmel_std"], dtype=np.float32)
    if mean.shape != (N_MELS,) or std.shape != (N_MELS,):
        raise ValueError(
            f"{path} has incompatible log-mel statistics: {mean.shape}, {std.shape}; "
            f"expected ({N_MELS},)"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError(f"{path} contains non-finite log-mel statistics")
    if np.any(std <= 0):
        raise ValueError(f"{path} contains non-positive log-mel standard deviations")
    return mean[:, None], std[:, None]


def cnn_input_from_waveform(
    waveform: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """One window -> the exact tensor layout `cnn_data.load_cnn` produces, for a single example.

    Preconditions: `waveform` is a 22.05 kHz mono window; `mean` and `std` are (N_MELS, 1) as
    returned by `load_logmel_statistics`.
    Postcondition: returns (1, N_MELS, N_FRAMES) float32 = (channel, freq, time). MediumCNN and
    MediumCRNN both take this layout.
    Raises: ValueError if the log-mel comes out the wrong shape, which would mean the window was
    not the canonical length.
    """
    features = (logmel(waveform) - mean) / std
    if features.shape != (N_MELS, N_FRAMES):
        raise ValueError(f"Expected log-mel {(N_MELS, N_FRAMES)}, got {features.shape}")
    return features[None, :, :].astype(np.float32, copy=False)


def cnn_batch_from_waveforms(
    waveforms: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Stack `cnn_input_from_waveform` over a batch.

    Postcondition: returns (B, 1, N_MELS, N_FRAMES) float32, in input order, so callers may align
    rows with labels positionally.
    """
    if not waveforms:
        return np.empty((0, 1, N_MELS, N_FRAMES), dtype=np.float32)
    return np.stack(
        [cnn_input_from_waveform(w, mean, std) for w in waveforms]
    ).astype(np.float32, copy=False)
