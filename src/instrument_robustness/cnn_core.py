"""Shared CNN core — used by BOTH the single-instrument and multiple-instrument tasks.

Holds the model (MediumCNN), the length-bucketed batching, the training/eval primitives,
and small helpers. The task-specific code lives in single/ and multi/; anything they both
need lives here so neither depends on the other.
"""

import json
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from .config import (
    BATCH_SIZE, CLASSES, DROPOUT, MANIFEST_JSON, MAX_IMBALANCE, SPEC_DIR,
    SPECAUG_FREQ_MASKS, SPECAUG_FREQ_WIDTH, SPECAUG_TIME_MASKS, SPECAUG_TIME_WIDTH,
    SPLITS_JSON, assert_fingerprint,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agg(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


# --------------------------------------------------------------------------- data

def load_manifest():
    """Load the cache manifest and splits, refusing a cache built under a different config.

    Postcondition: returns (manifest, splits, records-by-id) whose cached arrays were produced
    by the config now in effect. Raises StaleArtifactError otherwise — silently training on a
    cache built at a different CLIP_SECONDS or SR yields a plausible, meaningless model.
    """
    if not MANIFEST_JSON.exists() or not SPLITS_JSON.exists():
        sys.exit("ERROR: cache missing — run `python -m instrument_robustness.prep_data` first.")
    manifest = json.loads(MANIFEST_JSON.read_text())
    splits = json.loads(SPLITS_JSON.read_text())
    assert_fingerprint(manifest.get("fingerprint"), str(MANIFEST_JSON))
    return manifest, splits, {r["id"]: r for r in manifest["records"]}


def load_split(split_ids, by_id):
    """Cached spectrograms for a split -> (specs, y, ids).

    specs is a LIST of (1, n_mels, frames) tensors rather than one stacked array. Clips are all
    the same length now, so stacking would work — the list is kept because it costs nothing and
    keeps this correct if a variable-length experiment is ever run again. The whole set is
    ~100 MB, so it lives in memory and epochs never touch disk.
    """
    ids = sorted(split_ids)
    specs = [torch.from_numpy(np.load(SPEC_DIR / f"{i}.npy")).float().unsqueeze(0)
             for i in ids]
    y = torch.from_numpy(np.array([by_id[i]["label"] for i in ids], dtype=np.int64))
    return specs, y, ids


class LengthBatcher:
    """Yields batches of clips that all have the SAME frame count.

    Clips are now a fixed CLIP_SECONDS, so there is exactly one bucket and this behaves as an
    ordinary shuffling batcher at the full BATCH_SIZE. It is retained because it is the thing
    that makes padding unnecessary if clip length is ever varied again: padding to the batch
    maximum would reintroduce exactly the digital silence that breaks the noise sweep, and
    would contaminate BatchNorm statistics. Grouping by exact length sidesteps both.

    Cost when lengths DO vary: batches are smaller than BATCH_SIZE where a length has few
    clips. BatchNorm2d tolerates this because it pools over height and width as well as batch,
    so even a single clip yields n_mels x frames values per channel.
    """

    def __init__(self, specs, labels, batch_size, shuffle=False, seed=None):
        self.specs, self.labels, self.batch_size, self.shuffle = specs, labels, batch_size, shuffle
        self.rng = random.Random(seed)
        self.by_len = defaultdict(list)
        for i, s in enumerate(specs):
            self.by_len[s.shape[-1]].append(i)

    def __len__(self):
        return sum(int(np.ceil(len(v) / self.batch_size)) for v in self.by_len.values())

    def __iter__(self):
        batches = []
        for idxs in self.by_len.values():
            idxs = list(idxs)
            if self.shuffle:
                self.rng.shuffle(idxs)
            batches += [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)  # else every epoch walks lengths in the same order
        for b in batches:
            yield torch.stack([self.specs[i] for i in b]), self.labels[list(b)]


def class_weights(y_train, quiet=False):
    """N / (n_classes * n_c) — upweights minority classes. Applied only if the training
    split is more imbalanced than MAX_IMBALANCE."""
    counts = np.bincount(y_train.numpy(), minlength=len(CLASSES))
    ratio = counts.max() / max(counts.min(), 1)
    if ratio <= MAX_IMBALANCE:
        if not quiet:
            print(f"class ratio {ratio:.2f}:1 <= {MAX_IMBALANCE} — no class weights")
        return None, ratio
    w = len(y_train) / (len(CLASSES) * counts)
    if not quiet:
        lo, hi = CLASSES[int(counts.argmin())], CLASSES[int(counts.argmax())]
        print(f"class ratio {ratio:.2f}:1 > {MAX_IMBALANCE} (min {lo} {counts.min()}, "
              f"max {hi} {counts.max()}) — applying class weights "
              f"[{w.min():.3f}..{w.max():.3f}]")
    return torch.tensor(w, dtype=torch.float32), ratio


# --------------------------------------------------------------------------- model

class MediumCNN(nn.Module):
    """3 conv blocks (32/64/128) -> GAP -> Dense 128 -> Dropout -> Dense n_classes.

    Shared by both tasks: single-instrument reads the n_classes logits through softmax;
    multi-instrument reads the same logits through per-class sigmoids. Only the loss and the
    interpretation differ — the architecture is identical.
    """

    def __init__(self, n_classes=len(CLASSES), dropout=DROPOUT):
        super().__init__()

        def block(cin, cout):
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

    def forward(self, x):
        return self.head(self.gap(self.features(x)))


class MediumCRNN(nn.Module):
    """3 conv blocks -> collapse frequency -> BiGRU over time -> mean-pool -> Dense n_classes.

    Drop-in interchangeable with MediumCNN: same input (B, 1, n_mels, frames), same logits out,
    so the training loop, batcher and evaluation are shared unchanged.

    THE ONE ARCHITECTURAL DIFFERENCE, and the reason this model exists: MediumCNN aggregates its
    final feature map with global average pooling, which discards WHERE in time each feature
    occurred. Permute the columns of that feature map and the CNN's output is unchanged; do the
    same to this model's sequence and the output changes. So the CNN can represent "a bright
    attack transient is present" but not "the attack precedes the decay", while this can.

    Be precise about the scope of that claim: it is about the AGGREGATION step, not the whole
    network. The conv stack still encodes local temporal structure within its receptive field, and
    a GAP-CNN is NOT invariant to reversing the input spectrogram — convolution with an asymmetric
    kernel is not reversal-equivariant, so a reversed input yields a different feature map and
    therefore a different pooled output. Global ordering is what GAP removes; local pattern
    survives. (Measured: reversing a 128-frame input moves MediumCNN's logits by ~2e-3, which is
    this effect and not order sensitivity.)

    Frequency is pooled away before the recurrence; time is deliberately pooled only once
    (130 -> 32 frames), because pooling time is exactly what would throw away the thing this model
    is for.

    !! READ BEFORE TRUSTING ANY NUMBER THIS PRODUCES !!
    97.3% of clips in this cache are TILED — a short note looped to fill 3.0 s — and source note
    length correlates with class (a length-only classifier scores 0.1977 against 0.0833 chance).
    Tiling therefore writes note length into the time axis as a periodic repetition, and an
    order-sensitive model can read that period directly. The tiling-artifact test in FINDINGS §6
    cleared this architecture's GAP-CNN sibling, NOT this model — GAP cannot see period, this can.
    Run `single/envelope_probe.py` (which bounds what any order-sensitive model can extract from
    temporal structure alone) and re-run the §6 per-class check before reporting a CRNN score.

    Shape walk, for n_mels=128, frames=130:
        input                     (B,   1, 128, 130)
        block(1,32)   pool(2,2)   (B,  32,  64,  65)
        block(32,64)  pool(2,2)   (B,  64,  32,  32)
        block(64,128) pool(2,1)   (B, 128,  16,  32)   <- freq halved, time held
        AdaptiveAvgPool2d((1,None)) (B, 128, 1, 32)     <- frequency collapsed, time intact
        -> sequence               (B,  32, 128)         (batch, time, features)
        BiGRU(128 -> 128)         (B,  32, 256)
        mean over time            (B, 256)
        head                      (B, n_classes)

    Mean-pooling the GRU outputs rather than taking the final hidden state: each timestep's hidden
    state has already integrated the sequence, so order information survives the pooling, and the
    mean is less sensitive than a last-state readout to where in the window the note happens to
    sit.

    TWO CONFOUNDS to state whenever this model is compared with MediumCNN. Neither is a bug; both
    mean a CRNN win cannot be attributed to the recurrence alone.
      1. Capacity. 294,124 parameters against MediumCNN's 110,956 — 2.6x, on 5,788 training clips.
      2. Temporal resolution. MediumCNN pools time three times (130 -> 65 -> 32 -> 16 frames);
         this pools it once (130 -> 32), so the readout sees 2x the time steps. That is inherent
         to what a CRNN is for — pooling time away would defeat the point — but it is a second
         difference between the models, not just GAP vs GRU.
    """

    def __init__(self, n_classes=len(CLASSES), dropout=DROPOUT, rnn_hidden=128):
        super().__init__()

        def block(cin, cout, pool):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            block(1, 32, (2, 2)),
            block(32, 64, (2, 2)),
            block(64, 128, (2, 1)),      # frequency only — time resolution is the point
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # collapse freq, keep every time step
        self.rnn = nn.GRU(128, rnn_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * rnn_hidden, n_classes),
        )

    def forward(self, x):
        x = self.freq_pool(self.features(x))     # (B, C, 1, T)
        x = x.squeeze(2).transpose(1, 2)         # (B, T, C)
        x, _ = self.rnn(x)                       # (B, T, 2*hidden)
        return self.head(x.mean(dim=1))


# --------------------------------------------------------------------------- loops

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, balanced_accuracy, preds, targets). Single-label (softmax/argmax).

    Balanced accuracy — the mean of per-class recall — not raw accuracy. On an imbalanced
    set, a model that collapses onto the largest class is paid its prior and posts a number
    that reads like a result; raw accuracy's floor also drifts with the split, so it is not
    comparable across configurations. Balanced accuracy scores a collapsed model
    1/n_classes, and that floor is fixed whatever the imbalance.
    """
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item() * xb.size(0)
        preds.append(out.argmax(1).cpu())
        targets.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(targets), balanced_accuracy_score(targets, preds), preds, targets


def spec_augment(x):
    """SpecAugment: zero out a few random frequency bands and time bands per clip. Applied to
    TRAINING batches only. Masking to 0 = masking to the mean (spectrograms are z-scored), the
    standard choice. Time-mask width is capped at T//2 so short clips aren't wholly erased.

    Forces the model not to depend on any single spectrogram region — the direct fix for the
    train~0.99 / val~0.92 generalisation gap, and it stays a plain CNN (training-time only)."""
    B, _, F, T = x.shape
    x = x.clone()
    for b in range(B):
        for _ in range(SPECAUG_FREQ_MASKS):
            w = random.randint(0, SPECAUG_FREQ_WIDTH)
            f0 = random.randint(0, max(0, F - w))
            x[b, :, f0:f0 + w, :] = 0.0
        for _ in range(SPECAUG_TIME_MASKS):
            w = random.randint(0, min(SPECAUG_TIME_WIDTH, max(1, T // 2)))
            t0 = random.randint(0, max(0, T - w))
            x[b, :, :, t0:t0 + w] = 0.0
    return x


def train_one_epoch(model, loader, criterion, optimizer, device, augment=None):
    """Single-label epoch (softmax/argmax accuracy). augment(xb) applies SpecAugment."""
    model.train()
    total_loss, n, preds, targets = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if augment is not None:
            xb = augment(xb)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        preds.append(out.argmax(1).detach().cpu())
        targets.append(yb.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total_loss / n, balanced_accuracy_score(targets, preds)
