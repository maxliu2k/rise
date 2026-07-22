"""Head-to-head: does weight decay + SpecAugment close the train~0.99 / val~0.92 gap?

Compares on the VALIDATION split (test is untouched — tuning against test would void the
0.9234 number). Baseline val scores come from the already-trained model_s{seed}.pt, so only
the treatment is trained here. Reports mean +/- std over seeds; a real gain must clear the
~0.02 seed noise. Does NOT overwrite the baseline checkpoints.

    python -m instrument_robustness.tune_experiment [--seeds 42 43 44] [--wd 1e-3]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .config import (
    BATCH_SIZE, CLASSES, EARLY_STOP_PATIENCE, LEARNING_RATE, MAX_EPOCHS, OUTPUTS,
    PLATEAU_FACTOR, PLATEAU_PATIENCE, SEEDS,
)
from .train import (agg, class_weights, evaluate, get_device, load_manifest, load_split,
                    LengthBatcher, MediumCNN, set_seed, spec_augment, train_one_epoch)


def train_val(seed, data, weight_decay, use_specaug, device):
    """Train one model and return its best val balanced accuracy. No checkpoint saved."""
    (Xtr, ytr), (Xva, yva) = data
    set_seed(seed)
    weights, _ = class_weights(ytr, quiet=True)
    train_loader = LengthBatcher(Xtr, ytr, BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = LengthBatcher(Xva, yva, BATCH_SIZE)

    model = MediumCNN().to(device)
    crit = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=PLATEAU_FACTOR,
                                                       patience=PLATEAU_PATIENCE)
    augment = spec_augment if use_specaug else None

    best_loss, best_bacc, since = float("inf"), 0.0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        train_one_epoch(model, train_loader, crit, opt, device, augment)
        vl, vb, _, _ = evaluate(model, val_loader, crit, device)
        sched.step(vl)
        if vl < best_loss:                      # select on val loss, as the baseline does
            best_loss, best_bacc, since = vl, vb, 0
        else:
            since += 1
        print(f"    s{seed} ep {epoch:>2} | {time.perf_counter()-t0:4.1f}s | "
              f"val loss {vl:.4f} bacc {vb:.4f}{' *' if since == 0 else ''}")
        if since >= EARLY_STOP_PATIENCE:
            break
    return best_bacc


def baseline_val(seed, Xva, yva, device):
    """Val balanced accuracy of the already-trained baseline model for this seed."""
    p = OUTPUTS / f"model_s{seed}.pt"
    if not p.exists():
        return None
    m = MediumCNN().to(device)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["state_dict"])
    _, bacc, _, _ = evaluate(m, LengthBatcher(Xva, yva, BATCH_SIZE), nn.CrossEntropyLoss(), device)
    return bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--wd", type=float, default=1e-3, help="weight decay for the treatment")
    args = ap.parse_args()

    device = get_device()
    _, splits, by_id = load_manifest()
    Xtr, ytr, _ = load_split(splits["train"], by_id)
    Xva, yva, _ = load_split(splits["val"], by_id)
    print(f"device: {device} | comparing on VAL ({len(Xva)} clips), {len(args.seeds)} seeds")
    print(f"treatment: weight_decay={args.wd} + SpecAugment\n")

    base, treat = [], []
    for seed in args.seeds:
        b = baseline_val(seed, Xva, yva, device)
        if b is not None:
            base.append(b)
            print(f"  baseline  s{seed}: val bacc {b:.4f}  (existing model_s{seed}.pt)")
    print()
    for seed in args.seeds:
        print(f"  training treatment, seed {seed}...")
        t = train_val(seed, ((Xtr, ytr), (Xva, yva)), args.wd, True, device)
        treat.append(t)
        print(f"  treatment s{seed}: val bacc {t:.4f}\n")

    ba, ta = agg(base), agg(treat)
    print("=" * 60)
    print("RESULT (validation balanced accuracy)")
    print("=" * 60)
    print(f"  baseline (Adam, no aug):        {ba['mean']:.4f} +/- {ba['std']:.4f}")
    print(f"  +weight_decay({args.wd}) +SpecAug: {ta['mean']:.4f} +/- {ta['std']:.4f}")
    delta = ta["mean"] - ba["mean"]
    noise = max(ba["std"], ta["std"])
    print(f"\n  delta: {delta:+.4f} | seed noise ~{noise:.4f}")
    if delta > 2 * noise:
        print("  -> real improvement (clears 2x seed noise). Confirm on TEST next.")
    elif delta > noise:
        print("  -> promising but within ~1-2x seed noise; needs more seeds to confirm.")
    elif delta > -noise:
        print("  -> no meaningful difference (within seed noise).")
    else:
        print("  -> treatment is WORSE. The baseline regularises enough already.")


if __name__ == "__main__":
    main()
