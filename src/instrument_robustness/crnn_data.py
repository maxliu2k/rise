"""CRNN data loader — reuses the CNN log-mel features (features/cnn/{split}.npz) unchanged.

Same per-bin train-standardized log-mel as the CNN; the ONLY difference is layout: the CRNN keeps
the time axis as a sequence and must NOT collapse it to statistics. Returns X shaped
(N, n_frames=130, n_mels=128) = (batch, time, features) for the recurrent stack.
"""
import numpy as np
from instrument_robustness.config import FEATURES

CNN = FEATURES / "cnn"


def load_crnn(split):
    d = np.load(CNN / f"{split}.npz", allow_pickle=True)
    X = d["X"][..., 0]                 # (N,128,130) drop channel
    X = np.transpose(X, (0, 2, 1))     # (N,130,128) = (batch, time, features)
    return X.astype("float32"), d["y"]
