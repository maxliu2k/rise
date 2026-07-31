"""MediumCNN and the seed-ensemble combiners.

The model is a small 2-D convolutional classifier over log-mel: 3 conv blocks (32/64/128), global
average pooling, then a dense head. ~111k parameters, trained from scratch — it is the
non-pretrained counterweight to AST/MERT/PANNs in the model comparison.

WHY GLOBAL AVERAGE POOLING, stated because it is the architecturally load-bearing choice: GAP
aggregates the final feature map over both axes, so the model registers WHICH features are present
and not WHERE in time they occurred. That is deliberate here. 97.3% of clips in this dataset are
tiled — a short note looped to fill the 3.0 s window — and the loop period encodes the source note
length, which correlates with instrument at roughly twice chance purely as a recording artifact.
An order-sensitive readout can exploit that; GAP cannot. Measured: permuting the final feature
map's time steps moves this model's logits by ~2e-8.

Be precise about the scope of that claim. It concerns the AGGREGATION step, not the whole network.
The conv stack still encodes local temporal structure inside its receptive field, and the model is
NOT invariant to reversing the input spectrogram, because convolution with an asymmetric kernel is
not reversal-equivariant. Global ordering is what GAP discards; local pattern survives.

ENSEMBLING. Independently-initialised seeds trained on identical data disagree on roughly 5% of
test clips, so averaging them costs one inference pass and recovers part of that. Both combiners
are provided because they can disagree and neither is obviously right at small ensemble sizes:
soft voting uses confidence, hard voting uses only the argmax.

Measured on the 12-class Philharmonia test split (see the cnn-ensemble branch): the gain over the
mean single seed was +0.0074, roughly half the seed spread — real in direction, not resolvable at
that ensemble size. Ensembling here is cheap insurance, not a headline result. The ceiling is set
by how correlated the seeds' errors are (mean pairwise rho ~= 0.53), not by how many seeds are
averaged: going from 3 members to 10 removes only a further ~16% of the variance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from instrument_robustness.config import TARGET_LABELS

DROPOUT = 0.4


class MediumCNN(nn.Module):
    """3 conv blocks (32/64/128) -> GAP -> Dense 128 -> Dropout -> Dense n_classes.

    Preconditions: input is (B, 1, n_mels, n_frames) float32, per-bin standardized by Step 6/7.
    Postcondition: returns (B, n_classes) logits. Read through softmax for single-label use.
    """

    def __init__(self, n_classes: int = len(TARGET_LABELS), dropout: float = DROPOUT):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gap(self.features(x)))


def soft_vote(per_seed_probs: np.ndarray) -> np.ndarray:
    """Mean of the per-seed softmax vectors, then argmax.

    Preconditions: per_seed_probs is (n_seeds, N, n_classes) with rows summing to 1.
    Postcondition: returns (N,) predicted class indices.

    Uses confidence, so a barely-above-threshold member contributes less than a confident one.
    """
    if per_seed_probs.ndim != 3:
        raise ValueError(f"expected (n_seeds, N, n_classes), got {per_seed_probs.shape}")
    return per_seed_probs.mean(axis=0).argmax(axis=1)


def hard_vote(per_seed_probs: np.ndarray) -> np.ndarray:
    """Majority vote over per-seed argmax, ties broken by summed probability.

    Preconditions: per_seed_probs is (n_seeds, N, n_classes).
    Postcondition: returns (N,) predicted class indices.

    The tie-break is load-bearing rather than an edge case: with a handful of members and 12
    classes, an n-way split with no majority is common, and in that case this falls back to the
    soft vote.
    """
    if per_seed_probs.ndim != 3:
        raise ValueError(f"expected (n_seeds, N, n_classes), got {per_seed_probs.shape}")
    n_models, n, n_cls = per_seed_probs.shape
    votes = per_seed_probs.argmax(axis=2)
    counts = np.zeros((n, n_cls), dtype=int)
    for m in range(n_models):
        counts[np.arange(n), votes[m]] += 1
    summed = per_seed_probs.sum(axis=0)
    tied = counts == counts.max(axis=1, keepdims=True)
    return np.where(tied, summed, -np.inf).argmax(axis=1)


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 32) -> np.ndarray:
    """Softmax probabilities for X, in input order.

    Postcondition: returns (len(X), n_classes); row i corresponds to X[i], so callers may align
    predictions with labels positionally.
    """
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, len(TARGET_LABELS)), dtype="float32")
