"""SNR sweep: how fast does 12-class instrument ID fall apart under additive white noise?

Noise is injected at the **waveform** level, before spectrogram generation, and the
spectrogram is regenerated through prep_data.wav_to_logmel — the exact function that built
the training cache. Reimplementing it here would test a different pipeline than the one that
was trained, which is the bug this sweep exists to rule out (the clean condition is checked
against train.py's per-seed test score each run).

Multi-seed: the sweep runs against every model_s{seed}.pt and reports mean +/- std, matching
train.py. The noise is seeded per (condition, clip) and is IDENTICAL across model seeds, so
the reported spread is model variance, not noise variance. Each condition's noisy
spectrograms are built once and reused across all seed models — spectrogram generation is
the expensive step.

SNR is measured over the whole clip: clips are variable length and contain no padding, so
every sample is real audio and whole-clip power is the true signal power.
"""

import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef

from .config import (
    BATCH_SIZE, CLASSES, NOISE_SEED, OUTPUTS, SEED, SEEDS, SNR_LEVELS_DB, WAVE_DIR,
)
from .prep_data import wav_to_logmel
from .train import agg, get_device, load_manifest, MediumCNN, set_seed


def add_noise_at_snr(y, snr_db, rng):
    """Additive white Gaussian noise at `snr_db`, measured over the whole clip.

    Returns (noisy waveform, achieved SNR in dB).
    """
    p_sig = float(np.mean(y ** 2))
    if p_sig <= 0:
        return y.copy(), float("nan")
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=y.shape).astype(np.float32)
    achieved = 10.0 * np.log10(p_sig / float(np.mean(noise ** 2)))
    return (y + noise).astype(np.float32), achieved


def build_specs(records, snr_db, cond_idx):
    """Spectrograms for one condition, built once and shared across all seed models.
    snr_db=None means clean. Returns (specs, achieved_snr_list)."""
    specs, achieved = [], []
    for clip_idx, r in enumerate(records):
        y = np.load(WAVE_DIR / f"{r['id']}.npy")
        if snr_db is None:
            specs.append(wav_to_logmel(y))
            continue
        # seeded per (condition, clip), independent of model seed, so every model sees the
        # same corrupted inputs and the seed spread is pure model variance
        rng = np.random.default_rng([NOISE_SEED, cond_idx, clip_idx])
        y_noisy, ach = add_noise_at_snr(y, snr_db, rng)
        specs.append(wav_to_logmel(y_noisy))
        achieved.append(ach)
    return specs, achieved


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


def score(preds, targets):
    """Per-seed metrics for one condition. Balanced accuracy and MCC only — accuracy and F1
    both pay a collapsed classifier the class prior (see train.py / FINDINGS §7)."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "mcc": float(matthews_corrcoef(targets, preds)),
        "per_class_recall": {
            c: float((preds[targets == i] == i).mean()) if (targets == i).any() else None
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(
            targets, preds, labels=list(range(len(CLASSES)))),
    }


def run_condition(models, records, targets, snr_db, cond_idx, device):
    """Build this condition's spectrograms once, evaluate every seed model, aggregate."""
    specs, achieved = build_specs(records, snr_db, cond_idx)
    per_seed = {seed: score(predict(m, specs, device), targets) for seed, m in models.items()}

    baccs = [s["balanced_accuracy"] for s in per_seed.values()]
    mccs = [s["mcc"] for s in per_seed.values()]
    cm_sum = np.sum([s["confusion_matrix"] for s in per_seed.values()], axis=0)
    # per-class recall averaged across seeds — names which instruments fall first
    recall = {c: agg([per_seed[s]["per_class_recall"][c] for s in per_seed
                      if per_seed[s]["per_class_recall"][c] is not None])
              for c in CLASSES}

    return {
        "condition": "clean" if snr_db is None else f"{snr_db}dB",
        "snr_db": snr_db,
        "balanced_accuracy": agg(baccs),
        "mcc": agg(mccs),
        "per_class_recall": recall,
        "confusion_matrix_summed": cm_sum.tolist(),
        "achieved_snr_db_mean": float(np.mean(achieved)) if achieved else None,
        "achieved_snr_db_std": float(np.std(achieved)) if achieved else None,
        "per_seed": {seed: {"balanced_accuracy": s["balanced_accuracy"], "mcc": s["mcc"]}
                     for seed, s in per_seed.items()},
    }


def plot_sweep(results, path):
    """Left: balanced accuracy vs SNR, mean line + per-seed spread, with the clean and chance
    references. Right: per-class recall for all 12, so you can see which instruments the noise
    takes down first (legend ordered by how far each falls from clean to the noisiest level)."""
    noisy = [r for r in results if r["snr_db"] is not None]
    clean = next(r for r in results if r["snr_db"] is None)
    xs = [r["snr_db"] for r in noisy]
    chance = 1.0 / len(CLASSES)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    ys = [r["balanced_accuracy"]["mean"] for r in noisy]
    lo = [r["balanced_accuracy"]["min"] for r in noisy]
    hi = [r["balanced_accuracy"]["max"] for r in noisy]
    ax1.fill_between(xs, lo, hi, color="#1f77b4", alpha=0.18, label="seed min–max")
    ax1.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=6, label="balanced accuracy")
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7)
    ax1.axhline(clean["balanced_accuracy"]["mean"], ls="--", color="#2ca02c", lw=1.5,
                label=f"clean ({clean['balanced_accuracy']['mean']:.3f})")
    ax1.axhline(chance, ls=":", color="#d62728", lw=1.5,
                label=f"chance / collapsed ({chance:.3f})")
    ax1.set(xlabel="SNR (dB)", ylabel="balanced accuracy", title="Balanced accuracy vs. SNR")

    cmap = plt.get_cmap("turbo")
    fall = sorted(CLASSES,
                  key=lambda c: clean["per_class_recall"][c]["mean"]
                  - noisy[-1]["per_class_recall"][c]["mean"], reverse=True)
    for rank, c in enumerate(fall):
        col = cmap(rank / max(len(CLASSES) - 1, 1))
        ax2.plot(xs, [r["per_class_recall"][c]["mean"] for r in noisy], "o-",
                 color=col, lw=1.3, ms=3, label=c)
    ax2.axhline(chance, ls=":", color="#888", lw=1)
    ax2.set(xlabel="SNR (dB)", ylabel="recall (mean over seeds)",
            title="Per-class recall — legend ordered by how far each falls")

    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_ylim(-0.03, 1.05)
        ax.invert_xaxis()  # left-to-right = cleaner-to-noisier
        ax.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower left", fontsize=6, ncol=2)
    fig.suptitle(f"{len(CLASSES)}-class instrument ID — robustness to additive white noise "
                 f"({len(clean['per_seed'])} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    set_seed(SEED)
    device = get_device()

    models = {}
    for seed in SEEDS:
        p = OUTPUTS / f"model_s{seed}.pt"
        if not p.exists():
            continue
        m = MediumCNN().to(device)
        m.load_state_dict(torch.load(p, map_location=device)["state_dict"])
        models[seed] = m
    if not models:
        sys.exit("ERROR: no model_s*.pt in outputs/ — run `python -m instrument_robustness.train` first.")

    manifest, splits, by_id = load_manifest()
    records = [by_id[i] for i in sorted(splits["test"])]
    targets = np.array([r["label"] for r in records])
    print(f"device: {device} | {len(CLASSES)} classes | {len(records)} test clips | "
          f"{len(models)} seeds {list(models)}\n")

    conditions = [None] + list(SNR_LEVELS_DB)
    results = []
    for i, snr in enumerate(conditions):
        r = run_condition(models, records, targets, snr, i, device)
        results.append(r)
        ba, mc = r["balanced_accuracy"], r["mcc"]
        ach = "     clean" if r["achieved_snr_db_mean"] is None else f"{r['achieved_snr_db_mean']:6.2f}dB"
        print(f"  {r['condition']:<7} | bal acc {ba['mean']:.4f} +/- {ba['std']:.4f} | "
              f"MCC {mc['mean']:.4f} | achieved {ach}")

    chance = 1.0 / len(CLASSES)
    print("\n" + "=" * 74)
    print(f"SNR SWEEP — {len(CLASSES)} classes, {len(models)} seeds")
    print("=" * 74)
    print(f"balanced accuracy: chance = 1/{len(CLASSES)} = {chance:.4f} | MCC: 0.0 = no information\n")
    print(f"{'condition':<10}{'bal acc':>9}{'std':>8}{'MCC':>9}{'vs clean':>10}{'achieved SNR':>16}")
    clean_bacc = results[0]["balanced_accuracy"]["mean"]
    for r in results:
        ba = r["balanced_accuracy"]
        ach = "clean" if r["achieved_snr_db_mean"] is None else f"{r['achieved_snr_db_mean']:.2f}dB"
        drop = "" if r["snr_db"] is None else f"{ba['mean'] - clean_bacc:+.4f}"
        print(f"{r['condition']:<10}{ba['mean']:>9.4f}{ba['std']:>8.4f}{r['mcc']['mean']:>9.4f}"
              f"{drop:>10}{ach:>16}")

    # where does "minimal noise" start to bite? first condition losing >2% and >5% of clean
    print("\nDEGRADATION ONSET (how little noise it takes):")
    for thresh in (0.02, 0.05, 0.10):
        hit = next((r for r in results[1:]
                    if clean_bacc - r["balanced_accuracy"]["mean"] >= thresh), None)
        if hit:
            print(f"  first drop of {thresh:.0%}+: at {hit['condition']} "
                  f"(bal acc {hit['balanced_accuracy']['mean']:.4f})")
        else:
            print(f"  first drop of {thresh:.0%}+: never within the swept range")

    # which classes fall first — recall at the mildest noise level vs clean
    mild = results[1]  # highest SNR
    print(f"\nMOST FRAGILE CLASSES at {mild['condition']} (mildest noise), recall drop from clean:")
    drops = sorted(CLASSES, key=lambda c: mild["per_class_recall"][c]["mean"]
                   - results[0]["per_class_recall"][c]["mean"])
    for c in drops[:5]:
        d = mild["per_class_recall"][c]["mean"] - results[0]["per_class_recall"][c]["mean"]
        print(f"  {c:<13} {results[0]['per_class_recall'][c]['mean']:.3f} -> "
              f"{mild['per_class_recall'][c]['mean']:.3f}  ({d:+.3f})")

    # --- correctness checks
    metrics = json.loads((OUTPUTS / "metrics.json").read_text())
    ref = {s["seed"]: s["test_balanced_accuracy"] for s in metrics["per_seed"]}
    print("\nclean-path check (noise_eval clean vs train.py test, per seed):")
    ok = True
    for seed, s in results[0]["per_seed"].items():
        match = abs(s["balanced_accuracy"] - ref[seed]) < 1e-9
        ok &= match
        print(f"  seed {seed}: {s['balanced_accuracy']:.4f} vs {ref[seed]:.4f} "
              f"{'match' if match else 'MISMATCH — spectrogram path diverged'}")

    achieved_ok = all(abs(r["achieved_snr_db_mean"] - r["snr_db"]) < 0.5 for r in results[1:])
    baccs = [r["balanced_accuracy"]["mean"] for r in results[1:]]
    monotone = all(a >= b - 1e-9 for a, b in zip(baccs, baccs[1:]))
    print(f"achieved SNR on target (<0.5dB): {achieved_ok} | monotone degradation: {monotone}")

    (OUTPUTS / "snr_results.json").write_text(json.dumps({
        "classes": list(CLASSES), "n_classes": len(CLASSES),
        "chance_balanced_accuracy": chance,
        "seeds": list(models), "noise_seed": NOISE_SEED, "device": str(device),
        "n_test": len(records), "clean_path_check_passed": bool(ok),
        "results": results,
    }, indent=2))
    plot_sweep(results, OUTPUTS / "acc_vs_snr.png")
    print(f"\nwrote {OUTPUTS / 'snr_results.json'} and {OUTPUTS / 'acc_vs_snr.png'}")


if __name__ == "__main__":
    main()
