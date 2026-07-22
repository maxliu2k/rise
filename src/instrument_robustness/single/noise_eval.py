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

from ..config import (
    BATCH_SIZE, CLASSES, IN_BAND_HZ, NOISE_COLORS, NOISE_SEED, OUTPUTS, SEED, SEEDS,
    SNR_LEVELS_DB, SR, WAVE_DIR,
)
from ..prep_data import wav_to_logmel
from ..cnn_core import agg, get_device, load_manifest, MediumCNN, set_seed


def colored_noise(n, exponent, rng):
    """Unit-variance noise with power spectral density ~ 1/f**exponent.
    exponent 0 = white, 1 = pink, 2 = brown. Built in the rfft domain, then inverted."""
    f = np.fft.rfftfreq(n, d=1.0 / SR)
    scale = np.ones_like(f)
    scale[1:] = 1.0 / (f[1:] ** (exponent / 2.0))
    scale[0] = scale[1] if scale.size > 1 else 1.0  # DC: avoid divide-by-zero blowup
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * scale
    x = np.fft.irfft(spec, n=n).astype(np.float32)
    return x / (x.std() + 1e-12)


def _inband_snr(y, noise):
    """SNR measured only over IN_BAND_HZ — the honest figure for coloured noise, whose
    energy may sit largely outside the band where the music lives."""
    f = np.fft.rfftfreq(y.size, d=1.0 / SR)
    band = (f >= IN_BAND_HZ[0]) & (f <= IN_BAND_HZ[1])
    ps = float((np.abs(np.fft.rfft(y)) ** 2)[band].sum())
    pn = float((np.abs(np.fft.rfft(noise)) ** 2)[band].sum())
    return 10.0 * np.log10(ps / pn) if pn > 0 else float("nan")


def add_noise_at_snr(y, snr_db, rng, exponent=0.0):
    """Additive noise at `snr_db` over total clip power. exponent selects the colour
    (0=white, 1=pink, 2=brown). Returns (noisy waveform, achieved nominal dB, in-band dB)."""
    p_sig = float(np.mean(y ** 2))
    if p_sig <= 0:
        return y.copy(), float("nan"), float("nan")
    noise = colored_noise(y.size, exponent, rng)          # unit variance
    noise = noise * np.sqrt(p_sig / (10 ** (snr_db / 10.0)))  # scale to total-power SNR
    nominal = 10.0 * np.log10(p_sig / float(np.mean(noise ** 2)))
    return (y + noise).astype(np.float32), nominal, _inband_snr(y, noise)


def build_specs(records, snr_db, cond_idx, exponent):
    """Spectrograms for one condition, built once and shared across all seed models.
    snr_db=None means clean. Returns (specs, nominal_snr_list, inband_snr_list)."""
    specs, nominal, inband = [], [], []
    for clip_idx, r in enumerate(records):
        y = np.load(WAVE_DIR / f"{r['id']}.npy")
        if snr_db is None:
            specs.append(wav_to_logmel(y))
            continue
        # seeded per (condition, clip), independent of model seed, so every model sees the
        # same corrupted inputs and the seed spread is pure model variance
        rng = np.random.default_rng([NOISE_SEED, cond_idx, clip_idx])
        y_noisy, nom, ib = add_noise_at_snr(y, snr_db, rng, exponent)
        specs.append(wav_to_logmel(y_noisy))
        nominal.append(nom)
        inband.append(ib)
    return specs, nominal, inband


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


def run_condition(models, records, targets, snr_db, cond_idx, exponent, device):
    """Build this condition's spectrograms once, evaluate every seed model, aggregate."""
    specs, nominal, inband = build_specs(records, snr_db, cond_idx, exponent)
    per_seed = {seed: score(predict(m, specs, device), targets) for seed, m in models.items()}

    baccs = [s["balanced_accuracy"] for s in per_seed.values()]
    mccs = [s["mcc"] for s in per_seed.values()]
    cm_sum = np.sum([s["confusion_matrix"] for s in per_seed.values()], axis=0)
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
        "nominal_snr_db": float(np.mean(nominal)) if nominal else None,
        "inband_snr_db": float(np.nanmean(inband)) if inband else None,
        "per_seed": {seed: {"balanced_accuracy": s["balanced_accuracy"], "mcc": s["mcc"]}
                     for seed, s in per_seed.items()},
    }


def plot_colors(by_color, clean_bacc, path):
    """Two panels, both plotting balanced accuracy vs SNR with one line per noise colour.
    LEFT uses NOMINAL SNR (total power), RIGHT uses IN-BAND SNR (200Hz-8kHz). The whole
    point: if the colours separate on the left but collapse onto one curve on the right, the
    'colour matters' effect was an artifact of measuring SNR over total power. If they stay
    apart on the right, spectral shape has a real effect beyond where the energy sits."""
    chance = 1.0 / len(CLASSES)
    cols = {"white": "#444444", "pink": "#e377c2", "brown": "#8c564b"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for color, results in by_color.items():
        noisy = [r for r in results if r["snr_db"] is not None]
        y = [r["balanced_accuracy"]["mean"] for r in noisy]
        c = cols.get(color, None)
        ax1.plot([r["nominal_snr_db"] for r in noisy], y, "o-", color=c, lw=2, ms=5, label=color)
        ax2.plot([r["inband_snr_db"] for r in noisy], y, "o-", color=c, lw=2, ms=5, label=color)

    for ax, lab in ((ax1, "nominal SNR (total power)"), (ax2, "in-band SNR (200Hz–8kHz)")):
        ax.axhline(clean_bacc, ls="--", color="#2ca02c", lw=1.3, label=f"clean ({clean_bacc:.3f})")
        ax.axhline(chance, ls=":", color="#d62728", lw=1.3, label=f"chance ({chance:.3f})")
        ax.set(xlabel=lab, ylabel="balanced accuracy", ylim=(-0.03, 1.02))
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    ax1.set_title("vs nominal SNR — colours look different")
    ax2.set_title("vs in-band SNR — the honest comparison")
    fig.suptitle(f"{len(CLASSES)}-class instrument ID — white / pink / brown noise "
                 f"({len(next(iter(by_color.values()))[0]['per_seed'])} seeds)")
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
    chance = 1.0 / len(CLASSES)
    print(f"device: {device} | {len(CLASSES)} classes | {len(records)} test clips | "
          f"{len(models)} seeds {list(models)}")
    print(f"colours: {', '.join(NOISE_COLORS)} | chance = {chance:.4f}\n")

    # clean once (colour-independent); then each colour's noisy sweep. Distinct cond_idx per
    # (colour, level) keeps the noise realisations independent across the whole run.
    clean = run_condition(models, records, targets, None, 0, 0.0, device)
    clean_bacc = clean["balanced_accuracy"]["mean"]
    print(f"clean: balanced acc {clean_bacc:.4f} +/- {clean['balanced_accuracy']['std']:.4f}\n")

    by_color = {}
    cond_idx = 1
    for color, exponent in NOISE_COLORS.items():
        results = [clean]
        for snr in SNR_LEVELS_DB:
            r = run_condition(models, records, targets, snr, cond_idx, exponent, device)
            results.append(r)
            cond_idx += 1
            ba = r["balanced_accuracy"]
            print(f"  {color:<6} {r['condition']:<6} | bal acc {ba['mean']:.4f} | "
                  f"nominal {r['nominal_snr_db']:6.2f}dB | in-band {r['inband_snr_db']:6.2f}dB")
        by_color[color] = results
        print()

    # --- comparison table: balanced accuracy at each NOMINAL level, per colour
    print("=" * 78)
    print(f"NOISE-COLOUR COMPARISON — {len(CLASSES)} classes, {len(models)} seeds")
    print("=" * 78)
    print(f"balanced accuracy (chance {chance:.4f}). clean = {clean_bacc:.4f}\n")
    print(f"{'nominal':<9}" + "".join(f"{c:>10}" for c in NOISE_COLORS))
    for i, snr in enumerate(SNR_LEVELS_DB):
        row = f"{str(snr) + 'dB':<9}"
        for color in NOISE_COLORS:
            row += f"{by_color[color][i + 1]['balanced_accuracy']['mean']:>10.4f}"
        print(row)

    # the honest cut: at a fixed IN-BAND SNR, are the colours still different?
    print(f"\nsame data, but showing IN-BAND SNR per cell (nominal -> in-band):")
    print(f"{'nominal':<9}" + "".join(f"{c:>12}" for c in NOISE_COLORS))
    for i, snr in enumerate(SNR_LEVELS_DB):
        row = f"{str(snr) + 'dB':<9}"
        for color in NOISE_COLORS:
            row += f"{by_color[color][i + 1]['inband_snr_db']:>11.1f} "
        print(row)
    print("\nread: if brown's column above is shifted ~+20dB, its apparent robustness is just")
    print("that a 'nominal 0dB' brown clip is really ~+20dB where the music actually is.")

    # --- correctness checks
    metrics = json.loads((OUTPUTS / "metrics.json").read_text())
    ref = {s["seed"]: s["test_balanced_accuracy"] for s in metrics["per_seed"]}
    ok = all(abs(clean["per_seed"][seed]["balanced_accuracy"] - ref[seed]) < 1e-9 for seed in ref)
    print(f"\nclean-path check vs train.py (all seeds match): {ok}")
    for color in NOISE_COLORS:
        errs = [abs(by_color[color][i + 1]["nominal_snr_db"] - snr)
                for i, snr in enumerate(SNR_LEVELS_DB)]
        print(f"  {color:<6}: nominal SNR on target (<0.5dB): {all(e < 0.5 for e in errs)}")

    (OUTPUTS / "snr_results.json").write_text(json.dumps({
        "classes": list(CLASSES), "n_classes": len(CLASSES),
        "chance_balanced_accuracy": chance, "clean_balanced_accuracy": clean["balanced_accuracy"],
        "seeds": list(models), "noise_seed": NOISE_SEED, "in_band_hz": list(IN_BAND_HZ),
        "device": str(device), "n_test": len(records), "clean_path_check_passed": bool(ok),
        "by_color": by_color,
    }, indent=2))
    plot_colors(by_color, clean_bacc, OUTPUTS / "noise_colors.png")
    print(f"\nwrote {OUTPUTS / 'snr_results.json'} and {OUTPUTS / 'noise_colors.png'}")


if __name__ == "__main__":
    main()
