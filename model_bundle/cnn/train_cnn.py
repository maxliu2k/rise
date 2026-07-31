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

RESUMING. Re-running this command retrains only the seeds that are not already finished, so a run
killed partway costs at most one partial seed. Just run it again:

    python -m instrument_robustness.train_cnn

Each finished seed persists three files — the checkpoint, its validation probabilities, and a
record carrying the config fingerprint, the architecture, and the hashes of the arrays it trained
on. A seed is reused only when all four match the run now starting; otherwise it is retrained.
`validation_summary.json` names the reused seeds so a resumed run is visible rather than implied.

This exists because the ensemble vote used to be assembled from probabilities held in memory, so
losing the process meant losing every completed seed. That happened twice, both times at seed 43,
and the response both times was to spend the CPU again rather than to make the work durable.

SELECTION METRIC IS macro-F1, not balanced accuracy, as of the standardisation across all six
models. SVM and MERT already selected on macro-F1; noise_eval_common's clean-parity gate and
every noise metric compare on macro-F1; a CNN/CRNN result selected on something else could not
honestly be placed beside them. Balanced accuracy and MCC are still recorded in full for every
seed and every combiner -- CLAUDE.md's own rule is that they, not macro-F1, are the right metric
under class imbalance, and this file does not resolve that tension, it standardises on the metric
the rest of the project already committed to. See docs/AUDIT_CHECKLIST.md #10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from instrument_robustness.cnn_data import CNN as CNN_FEATURE_DIR
from instrument_robustness.cnn_data import load_cnn
from instrument_robustness.cnn_model import MediumCNN, hard_vote, predict_probs, soft_vote
from instrument_robustness.config import (ARTIFACTS, MAX_IMBALANCE, TARGET_LABELS,
                                          config_fingerprint)

DEFAULT_OUTPUT_DIR = ARTIFACTS / "cnn"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
# MAX_IMBALANCE is NOT redefined here. It used to be, at the same value config already held, while
# train_mert imported it -- so changing config would have silently given the two models different
# class weighting, in the head-to-head comparison that is this project's entire output.


def sha256(path: Path) -> str:
    """Content hash of a feature array, so finalize_* can prove the test split it evaluates is the
    same data validation was selected against."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _stats(values: list[float]) -> dict:
    """mean/std/min/max over a per-seed metric. std is 0.0 for a single seed, not NaN or undefined
    -- an n=1 std has no meaning, and CLAUDE.md's own rule is never to quote it as if it did, but
    the field must still exist for --seeds with one entry not to crash the caller."""
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)), "max": float(np.max(values))}


def seed_store_paths(out_dir: Path, seed: int) -> tuple[Path, Path, Path]:
    """(checkpoint, record, validation-probabilities) paths for one seed."""
    return (out_dir / f"model_s{seed}.pt",
            out_dir / f"seed_s{seed}.json",
            out_dir / f"val_probs_s{seed}.npy")


def write_seed_store(out_dir: Path, seed: int, record: dict, probs: np.ndarray,
                     provenance: dict) -> None:
    """Persist one finished seed so a later run can reuse it instead of retraining.

    Postcondition: the record file exists only if the probabilities were fully written first.
    Order is the whole point -- the record is the commit marker, so a run killed mid-write leaves
    no record and the seed is simply retrained. Checking three files for mutual consistency after
    the fact would be guesswork.
    """
    _, record_path, probs_path = seed_store_paths(out_dir, seed)
    np.save(probs_path, probs.astype(np.float32))
    record_path.write_text(json.dumps({**record, **provenance}, indent=2))


def read_seed_store(out_dir: Path, seed: int, provenance: dict, n_val: int):
    """A finished seed's (record, validation probabilities), or None if it must be retrained.

    Preconditions: `provenance` describes the run now in progress (config fingerprint,
    architecture, and the hashes of the feature arrays being trained on).
    Postcondition: returns None unless all three files exist, every provenance field matches
    exactly, and the stored probabilities have one row per current validation window.

    Anything short of that is reported as absent rather than repaired. Reusing probabilities that
    were computed against different features would corrupt the ensemble vote silently, and it is
    the ensemble -- not any single seed -- that finalize_cnn reports.
    """
    checkpoint, record_path, probs_path = seed_store_paths(out_dir, seed)
    if not (checkpoint.exists() and record_path.exists() and probs_path.exists()):
        return None
    try:
        record = json.loads(record_path.read_text())
        probs = np.load(probs_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if any(record.get(key) != value for key, value in provenance.items()):
        return None
    if probs.shape != (n_val, len(TARGET_LABELS)):
        return None
    return record, probs


def train_one_seed(seed, Xtr, ytr, Xva, yva, device, weights, out_dir, model_cls):
    """Train one seed, keeping the best-validation-loss weights.

    Postcondition: writes out_dir/model_s{seed}.pt and returns a per-seed record. The test split
    is not touched.
    """
    set_seed(seed)
    rng = random.Random(seed)
    model = model_cls().to(device)
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
                "architecture": model_cls.__name__,
                "config_fingerprint": config_fingerprint()}, path)

    probs = predict_probs(model, Xva, device, BATCH_SIZE)
    preds = probs.argmax(axis=1)
    labels = list(range(len(TARGET_LABELS)))
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        # macro_f1 is the SELECTION metric (see run_training); balanced_accuracy and mcc are kept
        # alongside per CLAUDE.md's own reporting rule (they do not reward a collapsed classifier
        # under imbalance the way macro-F1 can) -- recording them costs nothing and lets a reader
        # who distrusts the standardisation check it themselves.
        "val_macro_f1": float(f1_score(yva, preds, labels=labels, average="macro",
                                       zero_division=0)),
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


def run_training(model_cls, output_dir: Path, seeds, device: str) -> dict:
    """Train `model_cls` once per seed on train, select on validation, write the summary.

    Preconditions: Step-7 features exist for train and val under the current config.
    Postcondition: writes output_dir/model_s{seed}.pt per seed plus validation_summary.json, and
    returns that summary. The TEST split is never read here — finalize_* spends the single
    permitted evaluation.

    Shared by train_cnn and train_crnn, which differ only in architecture and output directory.
    Duplicating a whole trainer would mean fixing every future bug in two places.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load_cnn("train")
    Xva, yva = load_cnn("val")
    print(f"device: {device} | train {len(Xtr)} | val {len(Xva)} | "
          f"{len(TARGET_LABELS)} classes | input {Xtr.shape[1:]}")
    print(f"architecture: {model_cls.__name__} -> {output_dir}")
    print(f"seeds: {list(seeds)}\n")

    weights = class_weights(ytr)
    print("class weights: " + ("none" if weights is None else
                               f"[{weights.min():.3f}..{weights.max():.3f}]"))
    print(f"params: {sum(p.numel() for p in model_cls().parameters()):,}\n")

    # Per-seed provenance. Reusing a seed is only safe if it was trained by this architecture, on
    # these exact arrays, under this config -- so all four travel with the stored seed and must
    # match before it is accepted.
    provenance = {
        "config_fingerprint": config_fingerprint(),
        "architecture": model_cls.__name__,
        "train_sha256": sha256(CNN_FEATURE_DIR / "train.npz"),
        "val_sha256": sha256(CNN_FEATURE_DIR / "val.npz"),
    }

    records, per_seed_probs, reused = [], [], []
    for seed in seeds:
        stored = read_seed_store(output_dir, seed, provenance, len(Xva))
        if stored is not None:
            rec, probs = stored
            reused.append(seed)
            print(f"  s{seed} | reusing completed seed, val balanced acc "
                  f"{rec['val_balanced_accuracy']:.4f}\n")
        else:
            rec, probs = train_one_seed(seed, Xtr, ytr, Xva, yva, device, weights, output_dir,
                                        model_cls)
            write_seed_store(output_dir, seed, rec, probs, provenance)
            print(f"  s{seed} | val balanced acc {rec['val_balanced_accuracy']:.4f}\n")
        records.append(rec)
        per_seed_probs.append(probs)
    if reused:
        print(f"reused {len(reused)} of {len(seeds)} seeds from a previous run: {reused}\n")

    # Recomputed from probs rather than read out of `records`, uniformly for reused AND
    # freshly-trained seeds. A seed reused from an older run (before val_macro_f1 existed in the
    # per-seed store) would otherwise KeyError here, or -- worse -- silently mix an old-format
    # record's balanced accuracy into a macro-F1 average. probs is always present either way.
    label_ids = list(range(len(TARGET_LABELS)))
    stacked = np.stack(per_seed_probs)
    singles_macro_f1 = [float(f1_score(yva, p.argmax(axis=1), labels=label_ids,
                                       average="macro", zero_division=0))
                        for p in per_seed_probs]
    singles_balanced_accuracy = [r["val_balanced_accuracy"] for r in records]

    # SELECTION METRIC: macro-F1. This is the project's standardised comparison metric -- SVM and
    # MERT already select on it, PANNs' probe is scored on it, and it is what noise_eval_common's
    # clean-parity gate compares against, so a CNN/CRNN summary selected on anything else could not
    # honestly enter the noise sweep. Balanced accuracy is still recorded below, in full, because
    # macro-F1 rewards a collapsed classifier more as imbalance grows (see docs/FINDINGS.md S7) --
    # this is a standardisation choice for cross-model comparability, not a claim that macro-F1 is
    # the better metric in the abstract.
    combiners_macro_f1 = {
        "soft_vote": float(f1_score(yva, soft_vote(stacked), labels=label_ids,
                                    average="macro", zero_division=0)),
        "hard_vote": float(f1_score(yva, hard_vote(stacked), labels=label_ids,
                                    average="macro", zero_division=0)),
    }
    combiners_balanced_accuracy = {
        "soft_vote": float(balanced_accuracy_score(yva, soft_vote(stacked))),
        "hard_vote": float(balanced_accuracy_score(yva, hard_vote(stacked))),
    }
    # Chosen HERE, on validation. Selecting the better combiner after seeing test would be
    # selection on test, and would inflate whatever finalize_cnn reports.
    chosen = max(combiners_macro_f1, key=combiners_macro_f1.get)

    summary = {
        "config_fingerprint": config_fingerprint(),
        "architecture": model_cls.__name__,
        "label_order": list(TARGET_LABELS),
        "seeds": list(seeds),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "n_params": int(sum(p.numel() for p in model_cls().parameters())),
        # finalize_* refuses to run unless this is False, and re-hashes these inputs. Together
        # they make "the test split was sealed while validation was selected" checkable rather
        # than merely asserted.
        "test_evaluated": False,
        "inputs": {s: {"path": str(CNN_FEATURE_DIR / f"{s}.npz"),
                       "sha256": sha256(CNN_FEATURE_DIR / f"{s}.npz")}
                   for s in ("train", "val")},
        # Which seeds came off disk rather than being trained in this process. Every seed is
        # provenance-checked before reuse, but a reader deserves to know the run was resumed.
        "reused_seeds": list(reused),
        "per_seed": records,
        "selection_metric": "validation_macro_f1",
        # single_seed_val is macro-F1 -- it is what selection used and what summarize_results.py
        # reads as this row's headline number. single_seed_val_balanced_accuracy is the same five
        # seeds scored the other way, kept in full rather than reduced to "recorded but unused".
        "single_seed_val": _stats(singles_macro_f1),
        "single_seed_val_balanced_accuracy": _stats(singles_balanced_accuracy),
        "ensemble_val": combiners_macro_f1,
        "ensemble_val_balanced_accuracy": combiners_balanced_accuracy,
        "selected_combiner": chosen,
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 66)
    print(f"single seed (val)  macro-F1 {summary['single_seed_val']['mean']:.4f} "
          f"+/- {summary['single_seed_val']['std']:.4f}"
          f"  (balanced acc {summary['single_seed_val_balanced_accuracy']['mean']:.4f} "
          f"+/- {summary['single_seed_val_balanced_accuracy']['std']:.4f})")
    for name in combiners_macro_f1:
        print(f"ensemble {name:<10} macro-F1 {combiners_macro_f1[name]:.4f}"
              f"  (balanced acc {combiners_balanced_accuracy[name]:.4f})")
    print(f"selected combiner: {chosen}  (chosen on validation macro-F1, "
          f"before any test evaluation)")
    print("=" * 66)
    print(f"\nwrote {output_dir / 'validation_summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    run_training(MediumCNN, args.output_dir, args.seeds, args.device)
    print("next: python -m instrument_robustness.finalize_cnn")


if __name__ == "__main__":
    main()
