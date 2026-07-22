"""Shared CNN core — used by BOTH the single-instrument and multiple-instrument tasks.

Holds the model (MediumCNN), the variable-length batching, the training/eval primitives,
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
    SPLITS_JSON,
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
    if not MANIFEST_JSON.exists() or not SPLITS_JSON.exists():
        sys.exit("ERROR: cache missing — run `python -m instrument_robustness.prep_data` first.")
    manifest = json.loads(MANIFEST_JSON.read_text())
    splits = json.loads(SPLITS_JSON.read_text())
    return manifest, splits, {r["id"]: r for r in manifest["records"]}


def load_split(split_ids, by_id):
    """Cached spectrograms for a split -> (specs, y, ids).

    Clips are variable length, so specs is a LIST of (1, n_mels, frames) tensors rather than
    one stacked array. The whole set is ~50 MB, so it lives in memory and epochs never touch
    disk.
    """
    ids = sorted(split_ids)
    specs = [torch.from_numpy(np.load(SPEC_DIR / f"{i}.npy")).float().unsqueeze(0)
             for i in ids]
    y = torch.from_numpy(np.array([by_id[i]["label"] for i in ids], dtype=np.int64))
    return specs, y, ids


class LengthBatcher:
    """Yields batches of clips that all have the SAME frame count.

    Variable-length clips cannot be stacked into one tensor, and the usual workaround —
    padding to the batch maximum — would reintroduce exactly the digital silence that
    breaks the noise sweep, and would contaminate BatchNorm statistics. Grouping by exact
    length sidesteps both: every batch is uniform by construction, with zero padding.

    Cost: batches are smaller than BATCH_SIZE when a length has few clips (here ~14 mean vs
    a 32 target). BatchNorm2d tolerates this because it pools over height and width as well
    as batch, so even a single clip yields n_mels x frames values per channel.
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
