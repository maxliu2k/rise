"""Does a model's MISTAKES track tiling period? Tests use, not availability.

    python -m instrument_robustness.single.period_error_probe

`envelope_probe` measured what an order-sensitive model COULD extract from temporal structure
(period recovers source note length at r=+0.914; period alone predicts class at ~2x chance). It
cannot say whether a trained model actually uses it. This does.

THE IDEA. Within any one class, source note lengths vary — not every tuba note is 0.575 s. So when
a model gets a clip wrong, ask which class it guessed. If it is leaning on tiling period, its wrong
guesses should be biased toward classes whose TYPICAL period matches THIS clip's ACTUAL period. If
it is leaning on timbre, its wrong guesses should be biased toward timbre neighbours instead —
which is what the CNN's confusions already look like (double-bass -> cello, trumpet -> trombone,
all same-family).

WHY NOT JUST SHUFFLE THE CLIP. The obvious causal test — destroy the periodicity and watch accuracy
fall — was designed and discarded. Shuffling time chunks also destroys the attack/decay envelope,
and attack/decay is legitimate timbre (arguably the dominant instrument cue). A drop would be
uninterpretable: period or envelope, no way to tell. This test manipulates nothing, so it has no
such confound.

THE STATISTIC. For each misclassified clip, rank the 11 wrong classes by how close their median
period is to this clip's period, and record where the PREDICTED class falls, normalised to [0,1]:

    0.0  = the model picked the closest-period wrong class
    0.5  = period played no role (uniform over wrong classes)
    1.0  = the model picked the furthest-period wrong class

The CNN supplies the baseline. It cannot read period (GAP discards time order at the readout;
measured at 2e-8), so whatever period-matching ITS errors show is the background level from period
and timbre being correlated in the data anyway. The CRNN's departure from that baseline is the
signal. Running all CNN seeds gives that baseline a spread rather than a single number.

PRE-REGISTERED INTERPRETATION (fixed in conversation before the CRNN's score was known, and before
this file existed):
  * CRNN mean rank inside the CNN seeds' range
        -> not using period. Its accuracy advantage stands on its own.
  * CRNN mean rank clearly below the CNN range
        -> using period, by roughly that margin. Footnote it against any CRNN number.
  * below the CNN range AND concentrated on the hard classes (trumpet, oboe, double-bass)
        -> using note length exactly where timbre fails. The most damaging version; more than a
           footnote.

LIMITATION, stated plainly: this is correlational. It shows the errors are CONSISTENT WITH using
period; it cannot prove use. A causal test needs period changed while attack/decay is preserved,
and every method for that perturbs timbre somewhat.
"""
import json
import sys

import numpy as np
import torch

from ..cnn_core import BATCH_SIZE, LengthBatcher, MediumCNN, MediumCRNN, get_device, load_manifest, load_split
from ..config import CLASSES, OUTPUTS, SEEDS, assert_fingerprint, config_fingerprint
from .envelope_probe import autocorr, dominant_lag, envelope

RESULTS_JSON = OUTPUTS / "period_error_probe.json"
ARCHES = {"MediumCNN": MediumCNN, "MediumCRNN": MediumCRNN}


def periods(specs):
    """Estimated tiling period, in frames, for each spectrogram."""
    return np.array([dominant_lag(autocorr(envelope(s))) for s in specs], dtype=float)


@torch.no_grad()
def predict(path, specs, labels, device):
    """Class predictions from one checkpoint, loaded under its recorded architecture.

    Raises: SystemExit if the checkpoint is missing; StaleArtifactError on a config mismatch.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert_fingerprint(ckpt.get("fingerprint"), str(path))
    arch = ckpt.get("arch", "MediumCNN")     # checkpoints predating --model are all CNNs
    model = ARCHES[arch]().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out = []
    for xb, _ in LengthBatcher(specs, labels, BATCH_SIZE):
        out.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out), arch


def period_rank(preds, targets, clip_periods, class_median):
    """Mean normalised period-rank of the predicted class, over misclassified clips.

    Preconditions: class_median maps class index -> median period in frames, computed on TRAIN.
    Postcondition: returns (mean_rank, n_errors, per_true_class_mean_rank). 0.5 is the no-effect
    value; below 0.5 means errors favour period-matched classes.
    """
    ranks, by_class = [], {c: [] for c in range(len(CLASSES))}
    for p, t, q in zip(preds, targets, clip_periods):
        if p == t:
            continue
        wrong = [c for c in range(len(CLASSES)) if c != t]
        dist = sorted(wrong, key=lambda c: abs(class_median[c] - q))
        r = dist.index(p) / (len(wrong) - 1)      # 0 = closest period, 1 = furthest
        ranks.append(r)
        by_class[t].append(r)
    return (float(np.mean(ranks)) if ranks else float("nan"),
            len(ranks),
            {CLASSES[c]: float(np.mean(v)) for c, v in by_class.items() if v})


def main():
    device = get_device()
    manifest, splits, by_id = load_manifest()
    Xtr, ytr, _ = load_split(splits["train"], by_id)
    Xte, yte, _ = load_split(splits["test"], by_id)
    ytr, yte = ytr.numpy(), yte.numpy()

    ptr, pte = periods(Xtr), periods(Xte)
    class_median = {c: float(np.median(ptr[ytr == c])) for c in range(len(CLASSES))}
    print(f"test {len(Xte)} clips | per-class median period (frames), from TRAIN:")
    for c in sorted(class_median, key=lambda c: class_median[c]):
        print(f"  {CLASSES[c]:<14}{class_median[c]:6.1f}")

    crnn_path = OUTPUTS / "crnn" / "model_s42.pt"
    if not crnn_path.exists():
        sys.exit(f"ERROR: {crnn_path} missing — train it first:\n"
                 f"  python -m instrument_robustness.single.train --model crnn --seeds 42")

    print("\nmodel                mean rank   n errors   (0.5 = period played no role)")
    results = {}
    cnn_ranks = []
    for seed in SEEDS:
        path = OUTPUTS / f"model_s{seed}.pt"
        if not path.exists():
            continue
        preds, arch = predict(path, Xte, torch.from_numpy(yte), device)
        mean_rank, n_err, per_class = period_rank(preds, yte, pte, class_median)
        cnn_ranks.append(mean_rank)
        results[f"cnn_s{seed}"] = {"arch": arch, "mean_rank": mean_rank, "n_errors": n_err,
                                   "per_true_class": per_class}
        print(f"  CNN  s{seed}          {mean_rank:.4f}      {n_err:>4}")

    preds, arch = predict(crnn_path, Xte, torch.from_numpy(yte), device)
    crnn_rank, crnn_err, crnn_per_class = period_rank(preds, yte, pte, class_median)
    results["crnn_s42"] = {"arch": arch, "mean_rank": crnn_rank, "n_errors": crnn_err,
                           "per_true_class": crnn_per_class}
    print(f"  CRNN s42          {crnn_rank:.4f}      {crnn_err:>4}")

    lo, hi = min(cnn_ranks), max(cnn_ranks)
    inside = lo <= crnn_rank <= hi
    RESULTS_JSON.write_text(json.dumps({
        "fingerprint": config_fingerprint(),
        "class_median_period_frames": {CLASSES[c]: v for c, v in class_median.items()},
        "cnn_seeds": list(SEEDS), "cnn_range": [lo, hi],
        "crnn_mean_rank": crnn_rank,
        "crnn_inside_cnn_range": bool(inside),
        "results": results,
    }, indent=2))

    print("\n" + "=" * 72)
    print(f"CNN baseline range: {lo:.4f} – {hi:.4f}   (n={len(cnn_ranks)} seeds)")
    print(f"CRNN:               {crnn_rank:.4f}")
    if inside:
        print("-> INSIDE the CNN range. No evidence the CRNN's errors track tiling period;")
        print("   its accuracy advantage is not explained by the tiling shortcut.")
    elif crnn_rank < lo:
        print("-> BELOW the CNN range. The CRNN's errors DO favour period-matched classes.")
        worst = sorted(crnn_per_class.items(), key=lambda kv: kv[1])[:3]
        print("   most period-biased true classes: "
              + ", ".join(f"{k} {v:.3f}" for k, v in worst))
    else:
        print("-> ABOVE the CNN range: errors avoid period-matched classes. Not the predicted")
        print("   failure mode; treat as noise unless it replicates.")
    print("=" * 72)
    print(f"\nwrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
