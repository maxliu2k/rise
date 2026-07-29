"""MediumCRNN — the order-sensitive counterpart to MediumCNN.

Same input, same logits, so it shares crnn_data/cnn_data, the trainer, and the ensemble
combiners in cnn_model. Only the readout differs.

THE ONE ARCHITECTURAL DIFFERENCE. MediumCNN aggregates its final feature map with global average
pooling, which discards WHERE in time each feature occurred: permute that feature map's columns
and its output is unchanged (measured at 2e-8). This keeps the time axis as a sequence and runs a
bidirectional GRU along it, so attack -> sustain -> decay is representable (same permutation moves
its output by 1.2e-3).

Scope that claim to the AGGREGATION step, not the whole network. The conv stack encodes local
temporal structure inside its receptive field either way, and neither model is invariant to
reversing the input spectrogram, because convolution with an asymmetric kernel is not
reversal-equivariant. Global ordering is what GAP discards; local pattern survives.

TWO CONFOUNDS, to state whenever this is compared against MediumCNN. Neither is a defect, but a
CRNN win cannot be attributed to the recurrence alone:
  1. Capacity — 294,124 parameters against 110,956, on ~5.8k training windows.
  2. Temporal resolution — MediumCNN pools time three times (130 -> 65 -> 32 -> 16 frames); this
     pools once (130 -> 32), so its readout sees twice the time steps. Inherent to what a CRNN is
     for, but a second difference all the same.

ON THE TILING SHORTCUT. 97.3% of clips are tiled — a short note looped to fill the 3.0 s window —
and the loop period encodes source note length, which correlates with instrument at roughly twice
chance purely as a recording artifact. An order-sensitive readout can reach that; GAP cannot. This
was measured rather than assumed, on both sides:
  * availability — collapsing a spectrogram to its energy envelope (no timbre) predicts class at
    0.42 balanced accuracy, and the estimated loop period recovers true note length at r=+0.914.
  * use — on a trained CRNN, misclassifications favour period-matched classes at 0.4538 against a
    five-seed CNN baseline of 0.4592 (range 0.3844-0.5254). No excess. Its accuracy advantage is
    not explained by the shortcut.
That test is underpowered (26 errors, SE ~0.057) and correlational, so it rules out a large effect
rather than proving none. Re-run it whenever this model is retrained on different data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from instrument_robustness.config import TARGET_LABELS

DROPOUT = 0.4
RNN_HIDDEN = 128


class MediumCRNN(nn.Module):
    """3 conv blocks -> collapse frequency -> BiGRU over time -> mean-pool -> Dense n_classes.

    Preconditions: input is (B, 1, n_mels, n_frames) float32, per-bin standardized by Step 6/7.
    Postcondition: returns (B, n_classes) logits, interchangeable with MediumCNN's.

    Shape walk for (128 mels, 130 frames):
        input                       (B,   1, 128, 130)
        block(1,32)    pool(2,2)    (B,  32,  64,  65)
        block(32,64)   pool(2,2)    (B,  64,  32,  32)
        block(64,128)  pool(2,1)    (B, 128,  16,  32)   frequency halved, time held
        AdaptiveAvgPool2d((1,None)) (B, 128,   1,  32)   frequency collapsed, time intact
        -> sequence                 (B,  32, 128)        (batch, time, features)
        BiGRU(128 -> 128)           (B,  32, 256)
        mean over time              (B, 256)
        head                        (B, n_classes)

    Mean-pooling the GRU outputs rather than taking the final hidden state: each timestep's hidden
    state has already integrated the sequence, so order information survives the pooling, and the
    mean is less sensitive than a last-state readout to where in the window the note sits.
    """

    def __init__(self, n_classes: int = len(TARGET_LABELS), dropout: float = DROPOUT,
                 rnn_hidden: int = RNN_HIDDEN):
        super().__init__()

        def block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            block(1, 32, (2, 2)),
            block(32, 64, (2, 2)),
            block(64, 128, (2, 1)),      # frequency only -- time resolution is the point
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.GRU(128, rnn_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * rnn_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq_pool(self.features(x))     # (B, C, 1, T)
        x = x.squeeze(2).transpose(1, 2)         # (B, T, C)
        x, _ = self.rnn(x)                       # (B, T, 2*hidden)
        return self.head(x.mean(dim=1))
