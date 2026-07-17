"""SNR sweep: evaluate the trained CNN on the test split under additive white Gaussian noise.

Noise is injected at the **waveform** level, before spectrogram generation, and the
spectrogram is then regenerated through prep_data.wav_to_logmel — the exact function that
built the training cache. Reimplementing it here would mean the sweep tests a different
pipeline than the one that was trained, which is the bug this pilot exists to rule out.

SNR is measured over each clip's **active span** rather than the whole 2s clip. Median
active fraction is ~0.47, so whole-clip power would size the noise against a mostly-silent
signal and deliver a true SNR at the note roughly 3dB higher than requested — which shows
up as a suspiciously flat degradation curve.
"""

import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef)

from .config import (
    BATCH_SIZE, CLASSES, MODEL_PATH, NOISE_SEED, OUTPUTS, SNR_LEVELS_DB, WAVE_DIR,
)
from .prep_data import wav_to_logmel
from .train import get_device, load_manifest, MediumCNN, set_seed
from .config import SEED


def add_noise_at_snr(y, snr_db, rng):
    """Additive white Gaussian noise at `snr_db`, measured over the whole clip.

    Whole-clip power is the true signal power here: clips are variable length and contain
    no padding, so every sample is real audio. (Under the old fixed-length zero-padding this
    was not true, and whole-clip power overstated the SNR at the note by ~3dB.)

    Returns (noisy waveform, achieved SNR in dB).
    """
    p_sig = float(np.mean(y ** 2))
    if p_sig <= 0:
        return y.copy(), float("nan")
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=y.shape).astype(np.float32)
    achieved = 10.0 * np.log10(p_sig / float(np.mean(noise ** 2)))
    return (y + noise).astype(np.float32), achieved


@torch.no_grad()
def predict(model, specs, device):
    """Variable-length specs -> predictions, batched by exact frame count (see
    train.LengthBatcher for why padding is not an option here)."""
    model.eval()
    by_len = defaultdict(list)
    for i, s in enumerate(specs):
        by_len[s.shape[-1]].append(i)
    out = np.empty(len(specs), dtype=np.int64)
    for idxs in by_len.values():
        for i in range(0, len(idxs), BATCH_SIZE):
            b = idxs[i:i + BATCH_SIZE]
            X = torch.stack([torch.from_numpy(specs[j]).float().unsqueeze(0) for j in b])
            out[b] = model(X.to(device)).argmax(1).cpu().numpy()
    return out


def run_condition(model, records, snr_db, device, cond_idx):
    """One sweep condition. snr_db=None means clean (no noise added)."""
    specs, achieved = [], []
    for clip_idx, r in enumerate(records):
        y = np.load(WAVE_DIR / f"{r['id']}.npy")
        if snr_db is None:
            specs.append(wav_to_logmel(y))
            continue
        # seeded per (condition, clip) so the sweep is bit-for-bit reproducible
        rng = np.random.default_rng([NOISE_SEED, cond_idx, clip_idx])
        y_noisy, ach = add_noise_at_snr(y, snr_db, rng)
        specs.append(wav_to_logmel(y_noisy))
        achieved.append(ach)

    targets = np.array([r["label"] for r in records])
    preds = predict(model, specs, device)
    # full per-class precision/recall/F1: accuracy and F1 alone cannot distinguish "degraded
    # evenly" from "collapsed to one class", which is exactly the distinction this sweep exists
    # to make. A collapse reads as minority recall -> 0 with majority precision -> the class prior.
    rep = classification_report(targets, preds, labels=list(range(len(CLASSES))),
                                target_names=list(CLASSES), output_dict=True, zero_division=0)
    return {
        "condition": "clean" if snr_db is None else f"{snr_db}dB",
        "snr_db": snr_db,
        # Raw accuracy is deliberately absent. A model predicting 'cello' unconditionally
        # scores 0.62 on this split — it looks like a result and is a collapsed classifier
        # being paid the class prior. It also drifts with the split (0.65 before chunking),
        # so the apparent "floor" moves for reasons unrelated to the model.
        # balanced accuracy: 0.5 = chance at any imbalance. MCC: 0.0 = no information.
        #
        # F1 is deliberately absent, for the same reason accuracy is. Macro F1's floor
        # tracks the class prior (0.33 at 50/50, 0.47 at 90/10), so a collapsed model scores
        # HIGHER the more imbalanced the data — and our own two splits scored the identical
        # collapse at 0.3941 then 0.3844. F1 also discards true negatives by construction:
        # that is sound for retrieval, where a TN is one of a billion documents you rightly
        # ignored, but here a TN is a correctly identified cello. Precision and recall below
        # say everything F1 would, without averaging it into a respectable-looking number.
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "mcc": float(matthews_corrcoef(targets, preds)),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "per_class": {
            c: {"precision": float(rep[c]["precision"]),
                "recall": float(rep[c]["recall"]),
                "support": int(rep[c]["support"])}
            for c in CLASSES
        },
        "confusion_matrix": confusion_matrix(
            targets, preds, labels=list(range(len(CLASSES)))).tolist(),
        "achieved_snr_db_mean": float(np.mean(achieved)) if achieved else None,
        "achieved_snr_db_std": float(np.std(achieved)) if achieved else None,
        # Clips are variable length, so a short clip carries less evidence than a long one
        # at the same SNR. Without this breakdown, that difference would be invisible inside
        # the headline accuracy and could masquerade as a noise effect.
        "by_length": length_breakdown(records, preds, targets),
    }


def length_breakdown(records, preds, targets):
    """Balanced accuracy per clip-length bucket, so length is not confounded with the SNR
    effect. Buckets containing only one class report None — balanced accuracy is undefined
    there, and raw accuracy would be actively misleading."""
    edges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.01)]
    out = {}
    lens = np.array([r["clip_seconds"] for r in records])
    for lo, hi in edges:
        mask = (lens >= lo) & (lens < hi)
        if not mask.sum():
            continue
        both = len(np.unique(targets[mask])) == len(CLASSES)
        out[f"{lo:.1f}-{hi:.1f}s"] = {
            "n": int(mask.sum()),
            "balanced_accuracy": (float(balanced_accuracy_score(targets[mask], preds[mask]))
                                  if both else None),
        }
    return out


def plot_acc_vs_snr(results, path):
    """Two panels. Left: balanced accuracy, where 0.5 is chance no matter the imbalance —
    a collapsed model lands on the chance line instead of at a flattering 0.62. Right:
    per-class precision/recall, which names which class was abandoned."""
    noisy = [r for r in results if r["snr_db"] is not None]
    clean = next(r for r in results if r["snr_db"] is None)
    xs = [r["snr_db"] for r in noisy]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ys = [r["balanced_accuracy"] for r in noisy]
    ax1.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=7, label="additive white noise")
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8)
    # clean is not an SNR value, so it gets a reference line rather than an x position
    ax1.axhline(clean["balanced_accuracy"], ls="--", color="#2ca02c", lw=1.5,
                label=f"clean ({clean['balanced_accuracy']:.3f})")
    ax1.axhline(0.5, ls=":", color="#d62728", lw=1.5, label="chance / collapsed (0.50)")
    ax1.set(xlabel="SNR (dB)", ylabel="balanced accuracy",
            title="Balanced accuracy vs. SNR")

    for c, col in zip(CLASSES, ("#1f77b4", "#d62728")):
        ax2.plot(xs, [r["per_class"][c]["recall"] for r in noisy], "o-", color=col,
                 lw=2, ms=6, label=f"{c} recall")
        ax2.plot(xs, [r["per_class"][c]["precision"] for r in noisy], "s--", color=col,
                 lw=1.2, ms=4, alpha=0.6, label=f"{c} precision")
    ax2.set(xlabel="SNR (dB)", ylabel="score", title="Per-class precision / recall")

    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_ylim(-0.03, 1.05)
        ax.invert_xaxis()  # left-to-right = cleaner-to-noisier reads naturally
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Trumpet vs. cello, medium CNN — robustness to additive white noise")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not MODEL_PATH.exists():
        sys.exit("ERROR: no trained model — run `python train.py` first.")
    set_seed(SEED)
    device = get_device()

    manifest, splits, by_id = load_manifest()
    records = [by_id[i] for i in sorted(splits["test"])]
    print(f"device: {device} | test clips: {len(records)}\n")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = MediumCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])

    conditions = [None] + list(SNR_LEVELS_DB)
    results = [run_condition(model, records, snr, device, i)
               for i, snr in enumerate(conditions)]

    print("=" * 78)
    print("SNR SWEEP")
    print("=" * 78)
    print("balanced accuracy: 0.5 = chance at any class imbalance | MCC: 0.0 = no information")
    print("(accuracy and F1 are not reported — both pay a collapsed classifier for the")
    print(" class prior, and both have floors that drift with the split)\n")
    print(f"{'condition':<11}{'bal acc':>9}{'MCC':>8}{'macro P':>9}{'macro R':>9}"
          f"{'achieved SNR':>18}")
    for r in results:
        ach = ("clean" if r["achieved_snr_db_mean"] is None
               else f"{r['achieved_snr_db_mean']:.2f} ± {r['achieved_snr_db_std']:.2f} dB")
        print(f"{r['condition']:<11}{r['balanced_accuracy']:>9.4f}{r['mcc']:>8.4f}"
              f"{r['macro_precision']:>9.4f}{r['macro_recall']:>9.4f}{ach:>18}")

    print(f"\nper-class breakdown (recall -> 0 for one class is the collapse signature):")
    print(f"{'condition':<11}{'class':<10}{'precision':>11}{'recall':>9}{'support':>9}")
    for r in results:
        for i, c in enumerate(CLASSES):
            p = r["per_class"][c]
            print(f"{r['condition'] if i == 0 else '':<11}{c:<10}{p['precision']:>11.4f}"
                  f"{p['recall']:>9.4f}{p['support']:>9}")

    buckets = list(results[0]["by_length"])
    print(f"\nbalanced accuracy by clip length:")
    print(f"{'condition':<11}" + "".join(f"{b:>12}" for b in buckets))
    print(f"{'':<11}" + "".join(f"{'n=' + str(results[0]['by_length'][b]['n']):>12}"
                                for b in buckets))
    for r in results:
        row = f"{r['condition']:<11}"
        for b in buckets:
            v = r["by_length"].get(b, {}).get("balanced_accuracy")
            row += f"{v:>12.4f}" if v is not None else f"{'—':>12}"
        print(row)

    # --- correctness checks on the hook itself
    clean_bacc = results[0]["balanced_accuracy"]
    ref = json.loads((OUTPUTS / "metrics.json").read_text())["test_balanced_accuracy"]
    print(f"\nclean path check: noise_eval {clean_bacc:.4f} vs train.py {ref:.4f}", end=" ")
    print("— match" if abs(clean_bacc - ref) < 1e-9
          else "— MISMATCH: the spectrogram path diverged between training and eval")

    for r in results[1:]:
        err = abs(r["achieved_snr_db_mean"] - r["snr_db"])
        status = "ok" if err < 0.5 else "OFF TARGET"
        print(f"achieved SNR at {r['snr_db']:>2}dB target: "
              f"{r['achieved_snr_db_mean']:.2f}dB (err {err:.3f}) — {status}")

    accs = [r["balanced_accuracy"] for r in results[1:]]
    monotone = all(a >= b - 1e-9 for a, b in zip(accs, accs[1:]))
    print(f"\ndegradation monotone as SNR falls: {monotone}")
    print(f"drop from clean to {results[-1]['condition']}: "
          f"{clean_bacc - accs[-1]:+.4f} balanced accuracy")

    # A collapse is not the same failure as even degradation, and accuracy alone hides it.
    for r in results[1:]:
        dead = [c for c in CLASSES if r["per_class"][c]["recall"] == 0.0]
        if dead:
            print(f"  COLLAPSE at {r['condition']}: recall 0.00 for {', '.join(dead)} "
                  f"— predicting one class regardless of input")

    (OUTPUTS / "snr_results.json").write_text(json.dumps({
        "seed": SEED, "noise_seed": NOISE_SEED, "device": str(device),
        "n_test": len(records), "results": results,
    }, indent=2))
    plot_acc_vs_snr(results, OUTPUTS / "acc_vs_snr.png")
    print(f"\nwrote {OUTPUTS / 'snr_results.json'} and {OUTPUTS / 'acc_vs_snr.png'}")


if __name__ == "__main__":
    main()
