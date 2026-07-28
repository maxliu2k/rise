"""CRNN data loader — reuses the CNN log-mel features (features/cnn/{split}.npz) unchanged.

Same per-bin train-standardized log-mel as the CNN; the ONLY difference is layout: the CRNN keeps
the time axis as a sequence and must NOT collapse it to statistics. Returns X shaped
(N, n_frames=130, n_mels=128) = (batch, time, features) for the recurrent stack.
"""
import numpy as np
from instrument_robustness.config import FEATURES, assert_serialized_fingerprint

CNN = FEATURES / "cnn"


def load_crnn(split):
    path = CNN / f"{split}.npz"
    with np.load(path, allow_pickle=True) as data:
        assert_serialized_fingerprint(
            data["config_fingerprint"] if "config_fingerprint" in data else None,
            str(path),
        )
        X = data["X"][..., 0]                 # (N,128,130) drop channel
        y = data["y"]
    X = np.transpose(X, (0, 2, 1))           # (N,130,128) = (batch, time, features)
    return X.astype("float32"), y
