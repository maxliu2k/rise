"""Ensemble the per-seed CNN checkpoints and evaluate on the clean test set.

    python -m instrument_robustness.single.ensemble_eval

The seeds already trained for the mean +/- std report are three independently-initialised models
on identical data. Averaging them costs one extra inference pass and is the cheapest accuracy the
study can buy, so it is worth knowing whether it buys any.

TWO COMBINERS, both reported:
  * soft vote  -- mean of the softmax probability vectors, then argmax. Uses confidence, so a
                  model that is barely above threshold contributes less than a confident one.
  * hard vote  -- per-model argmax, then majority. Ties broken by summed probability, since with
                  3 models and 12 classes a 1-1-1 split is common.

HOW TO READ THE COMPARISON. The honest baseline is the MEAN single-seed score (0.9600 +/- 0.0138),
not the best seed. The best of three (0.9757) is a maximum selected on the test set: it is not a
number any single training run can be relied on to reproduce, and beating it is not the bar.
Beating the mean by more than the seed spread is.

PRE-REGISTERED INTERPRETATION (written before running):
  * ensemble > mean single seed by more than the seed spread (0.0138)
        -> ensembling buys real accuracy; report it as the headline for this branch.
  * ensemble > mean but within the spread
        -> the expected small gain from variance reduction, not a resolvable effect at 3 seeds.
           Report the number, do not claim an improvement.
  * ensemble <= mean single seed
        -> the seeds are making correlated errors, so averaging cannot help. That is itself worth
           knowing: it would say the remaining errors are data-limited, not initialisation-limited.

Clean audio only. Noise is deliberately out of scope here.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, classification_report, matthews_corrcoef

from ..cnn_core import BATCH_SIZE, LengthBatcher, MediumCNN, get_device, load_manifest, load_split
from ..config import CLASSES, OUTPUTS, SEEDS, assert_fingerprint, config_fingerprint

RESULTS_JSON = OUTPUTS / "ensemble_metrics.json"


@torch.no_grad()
def predict_probs(model, loader, device):
    """Softmax probabilities for every clip in `loader`, in loader order.

    Postcondition: returns (probs (N, n_classes), targets (N,)). Rows sum to 1.
    """
    model.eval()
    probs, targets = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        probs.append(F.softmax(out, dim=1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def load_models(device):
    """Every seed's checkpoint, fingerprint-checked.

    Raises: SystemExit if any seed in config.SEEDS is missing. Ensembling whatever happens to be
    on disk would silently report a 2-model ensemble as a 3-model one.
    """
    missing = [s for s in SEEDS if not (OUTPUTS / f"model_s{s}.pt").exists()]
    if missing:
        sys.exit(f"ERROR: no checkpoint for seed(s) {missing}. An ensemble of "
                 f"{[s for s in SEEDS if s not in missing]} would be reported as if complete.\n"
                 f"  Run: python -m instrument_robustness.single.train --seeds "
                 f"{' '.join(str(s) for s in missing)}")
    models = {}
    for seed in SEEDS:
        path = OUTPUTS / f"model_s{seed}.pt"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        assert_fingerprint(ckpt.get("fingerprint"), f"outputs/model_s{seed}.pt")
        m = MediumCNN().to(device)
        m.load_state_dict(ckpt["state_dict"])
        models[seed] = m
    return models


def hard_vote(per_seed_probs):
    """Majority vote over per-model argmax, ties broken by summed probability.

    Preconditions: per_seed_probs is (n_models, N, n_classes).
    Postcondition: returns (N,) predicted class indices.

    With 3 models and 12 classes, three-way disagreement is common, so the tie-break is load-
    bearing rather than a rare edge case — it is the summed probability, i.e. the soft vote.
    """
    n_models, n, n_cls = per_seed_probs.shape
    votes = per_seed_probs.argmax(axis=2)                       # (n_models, N)
    counts = np.zeros((n, n_cls), dtype=int)
    for m in range(n_models):
        counts[np.arange(n), votes[m]] += 1
    summed = per_seed_probs.sum(axis=0)                          # (N, n_classes)
    best = counts.max(axis=1, keepdims=True)
    tied = counts == best                                        # candidates per row
    return np.where(tied, summed, -np.inf).argmax(axis=1)


def scored(preds, targets):
    return {"balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
            "mcc": float(matthews_corrcoef(targets, preds))}


def main():
    device = get_device()
    manifest, splits, by_id = load_manifest()
    Xte, yte, test_ids = load_split(splits["test"], by_id)
    loader = LengthBatcher(Xte, yte, BATCH_SIZE)
    print(f"device: {device} | {len(Xte)} test clips | {len(CLASSES)} classes")

    models = load_models(device)
    print(f"ensembling seeds {list(models)}\n")

    per_seed, targets = [], None
    for seed, model in models.items():
        p, t = predict_probs(model, loader, device)
        if targets is None:
            targets = t
        else:
            # LengthBatcher is deterministic without shuffle, but if that ever changed the rows
            # would silently misalign and the ensemble would average mismatched clips.
            assert np.array_equal(t, targets), f"seed {seed} saw the clips in a different order"
        per_seed.append(p)
        s = scored(p.argmax(axis=1), t)
        print(f"  s{seed} alone | bacc {s['balanced_accuracy']:.4f} | mcc {s['mcc']:.4f}")
    per_seed = np.stack(per_seed)

    soft = scored(per_seed.mean(axis=0).argmax(axis=1), targets)
    hard = scored(hard_vote(per_seed), targets)

    singles = [scored(p.argmax(axis=1), targets)["balanced_accuracy"] for p in per_seed]
    mean_single = float(np.mean(singles))
    std_single = float(np.std(singles, ddof=1))
    best_single = float(np.max(singles))
    gain = soft["balanced_accuracy"] - mean_single

    rep = classification_report(targets, per_seed.mean(axis=0).argmax(axis=1),
                                labels=list(range(len(CLASSES))), target_names=list(CLASSES),
                                output_dict=True, zero_division=0)

    RESULTS_JSON.write_text(json.dumps({
        "fingerprint": config_fingerprint(),
        "seeds": list(SEEDS),
        "n_test": int(len(targets)),
        "classes": list(CLASSES),
        "single_seed": {"per_seed": singles, "mean": mean_single, "std": std_single,
                        "best": best_single},
        "soft_vote": soft,
        "hard_vote": hard,
        "gain_over_mean_single": gain,
        "exceeds_seed_spread": bool(abs(gain) > std_single),
        "per_class_soft_vote": {c: {"precision": float(rep[c]["precision"]),
                                    "recall": float(rep[c]["recall"]),
                                    "support": int(rep[c]["support"])} for c in CLASSES},
    }, indent=2))

    print("\n" + "=" * 66)
    print(f"single seed (mean)  {mean_single:.4f} +/- {std_single:.4f}")
    print(f"single seed (best)  {best_single:.4f}   <- selected on test, not a target")
    print(f"ensemble soft vote  {soft['balanced_accuracy']:.4f} | mcc {soft['mcc']:.4f}")
    print(f"ensemble hard vote  {hard['balanced_accuracy']:.4f} | mcc {hard['mcc']:.4f}")
    print(f"\ngain over mean single seed: {gain:+.4f}  (seed spread {std_single:.4f})")
    if abs(gain) <= std_single:
        print("-> within the seed spread. Report the number; do not claim an improvement.")
    elif gain > 0:
        print(f"-> real gain, {gain / std_single:.1f}x the spread.")
    else:
        print("-> ensemble is WORSE than the average single seed: the seeds are making correlated\n"
              "   errors, so the remaining errors are data-limited rather than init-limited.")
    print("=" * 66)
    print(f"\nwrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
