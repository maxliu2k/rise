"""Train the from-scratch CNN ensemble. VALIDATION ONLY — the test split is never read here.

    python -m instrument_robustness.train_cnn

Trains one MediumCNN per seed on the Step-7 CNN features, selecting each seed's checkpoint by
validation loss, then reports the seed ensemble on validation. `finalize_cnn` performs the single
permitted test evaluation, mirroring train_svm/finalize_svm and train_mert/finalize_mert.

WHY MULTIPLE SEEDS. A single run of this model is not a stable number: across seeds the test score
spans roughly 3.4 points, which is wider than the margin most model comparisons turn on. Anything
quoted from one seed is a draw from that distribution, not a measurement. Report mean +/- std.

WHY THE ENSEMBLE IS SELECTED ON VALIDATION. Soft and hard voting can disagree, and picking the
better of the two after seeing the test set is selection on test. The choice is made here, on
validation, and finalize_cnn reports the pre-committed combiner.

Checkpoints carry state_dict, label_order and config_fingerprint, matching the contract the noise
evaluators enforce — a checkpoint trained under a different label set or window length must fail
to load rather than silently produce plausible predictions.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

from instrument_robustness.cnn_data import load_cnn
from instrument_robustness.cnn_model import MediumCNN, hard_vote, predict_probs, soft_vote
from instrument_robustness.config import ARTIFACTS, TARGET_LABELS, config_fingerprint

DEFAULT_OUTPUT_DIR = ARTIFACTS / "cnn"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_IMBALANCE = 1.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray) -> torch.Tensor | None:
    """N / (n_classes * n_c), applied only if the split is more imbalanced than MAX_IMBALANCE."""
    counts = np.bincount(y, minlength=len(TARGET_LABELS))
    if counts.max() / max(counts.min(), 1) <= MAX_IMBALANCE:
        return None
    return torch.tensor(len(y) / (len(TARGET_LABELS) * np.maximum(counts, 1)), dtype=torch.float32)


def iterate(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, rng: random.Random):
    idx = list(range(len(X)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        yield torch.from_numpy(X[b]), torch.from_numpy(y[b])


def train_one_seed(seed, Xtr, ytr, Xva, yva, device, weights, out_dir):
    """Train one seed, keeping the best-validation-loss weights.

    Postcondition: writes out_dir/model_s{seed}.pt and returns a per-seed record. The test split
    is not touched.
    """
    set_seed(seed)
    rng = random.Random(seed)
    model = MediumCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    epoch_times = []
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in iterate(Xtr, ytr, BATCH_SIZE, True, rng):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in iterate(Xva, yva, BATCH_SIZE, False, rng):
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
                n += xb.size(0)
        val_loss /= max(n, 1)
        epoch_times.append(time.perf_counter() - t0)
        scheduler.step(val_loss)

        flag = ""
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            stale += 1
        print(f"  s{seed} | ep {epoch:>2}/{MAX_EPOCHS} | {epoch_times[-1]:5.1f}s | "
              f"val loss {val_loss:.4f}{flag}")
        if stale >= EARLY_STOP_PATIENCE:
            print(f"  s{seed} | early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    path = out_dir / f"model_s{seed}.pt"
    torch.save({"state_dict": best_state, "seed": seed, "label_order": list(TARGET_LABELS),
                "config_fingerprint": config_fingerprint()}, path)

    probs = predict_probs(model, Xva, device, BATCH_SIZE)
    preds = probs.argmax(axis=1)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        "val_balanced_accuracy": float(balanced_accuracy_score(yva, preds)),
        "val_mcc": float(matthews_corrcoef(yva, preds)),
        "mean_epoch_s": float(np.mean(epoch_times)),
        "checkpoint": str(path),
    }, probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the from-scratch CNN ensemble (validation only).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_cnn("train")
    Xva, yva = load_cnn("val")
    print(f"device: {args.device} | train {len(Xtr)} | val {len(Xva)} | "
          f"{len(TARGET_LABELS)} classes | input {Xtr.shape[1:]}")
    print(f"seeds: {args.seeds}\n")

    weights = class_weights(ytr)
    print("class weights: " + ("none" if weights is None else
                               f"[{weights.min():.3f}..{weights.max():.3f}]"))
    print(f"params: {sum(p.numel() for p in MediumCNN().parameters()):,}\n")

    records, per_seed_probs = [], []
    for seed in args.seeds:
        rec, probs = train_one_seed(seed, Xtr, ytr, Xva, yva, args.device, weights, args.output_dir)
        records.append(rec)
        per_seed_probs.append(probs)
        print(f"  s{seed} | val balanced acc {rec['val_balanced_accuracy']:.4f}\n")

    stacked = np.stack(per_seed_probs)
    singles = [r["val_balanced_accuracy"] for r in records]
    combiners = {
        "soft_vote": float(balanced_accuracy_score(yva, soft_vote(stacked))),
        "hard_vote": float(balanced_accuracy_score(yva, hard_vote(stacked))),
    }
    # Chosen HERE, on validation. Selecting the better combiner after seeing test would be
    # selection on test, and would inflate whatever finalize_cnn reports.
    chosen = max(combiners, key=combiners.get)

    summary = {
        "config_fingerprint": config_fingerprint(),
        "label_order": list(TARGET_LABELS),
        "seeds": list(args.seeds),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "n_params": int(sum(p.numel() for p in MediumCNN().parameters())),
        "per_seed": records,
        "single_seed_val": {"mean": float(np.mean(singles)),
                            "std": float(np.std(singles, ddof=1)) if len(singles) > 1 else 0.0,
                            "min": float(np.min(singles)), "max": float(np.max(singles))},
        "ensemble_val": combiners,
        "selected_combiner": chosen,
    }
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 66)
    print(f"single seed (val)  {summary['single_seed_val']['mean']:.4f} "
          f"+/- {summary['single_seed_val']['std']:.4f}")
    for name, score in combiners.items():
        print(f"ensemble {name:<10} {score:.4f}")
    print(f"selected combiner: {chosen}  (chosen on validation, before any test evaluation)")
    print("=" * 66)
    print(f"\nwrote {args.output_dir / 'validation_summary.json'}")
    print("next: python -m instrument_robustness.finalize_cnn")


if __name__ == "__main__":
    main()
