"""noise_sweep.py — build the shared noisy TEST windows, ONCE, for every model to reuse.

DATASET-AGNOSTIC: driven by config's DATA_ROOT, so the identical protocol (same SNR levels,
same noise types, same SNR math, same seed scheme) runs on TinySOL and on Philharmonia. Only
the data root changes. That is what makes the degradation curves comparable across datasets.

DESIGN RULES (these keep the experiment valid):
  * Generated ONCE and written to disk so every model featurizes the SAME noisy audio ->
    predictions stay paired for McNemar / cluster bootstrap across models.
  * TEST SPLIT ONLY. train/val are never touched; models stay clean-trained.
  * ONE noise realization per (window, noise_type), SCALED to every SNR. The seed deliberately
    excludes the SNR: if each level drew fresh noise, the degradation curve would confound
    "more noise" with "different noise", and part of the drop between 20 dB and 0 dB would be
    noise variability rather than level. Scaling a single realization isolates the SNR axis.
  * The seed also folds in the dataset/config fingerprint, so noise cannot be silently carried
    across two different builds of the data.
  * NO re-normalization after mixing. Note this is NOT because scaling changes the SNR -- SNR is
    a power RATIO and is invariant to scaling the whole mixture. The reason is that the clean
    signal has a fixed reference gain (Step 5 normalized every window to TARGET_RMS), and models
    were trained at that gain; rescaling the mixture would present a different amplitude
    distribution than training. Clipping is the genuinely destructive operation -- it is
    nonlinear and really does corrupt the SNR -- hence float32 output.
  * Featurization later must use the EXISTING train-only stats — never recompute on noisy data.

SNR math (power, not amplitude):
    P = mean(x**2)
    alpha = sqrt( P_signal / (P_noise * 10**(SNR_dB/10)) )
    x_noisy = x_clean + alpha * noise
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, hashlib, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, librosa
from instrument_robustness.config import (ROOT, WORK, SR, TARGET_RMS, WINDOWS_CSV,
                                          config_fingerprint_json)
warnings.filterwarnings("ignore")

SNRS = [20, 10, 5, 0, -5]
NOISE_TYPES = ["white", "natural", "mechanical"]
NOISY_DIR = WORK / "windows_noisy"
NOISE_ROOT = Path(os.environ.get("RISE_NOISE_ROOT", Path.home() / "Downloads/noise_sources"))
ESC50_DIR = NOISE_ROOT / "ESC-50-master" / "audio"
ESC50_META = NOISE_ROOT / "ESC-50-master" / "meta" / "esc50.csv"
CLIP_LEN = int(round(3.0 * SR))

# ESC-50 orders its 50 categories in five blocks of ten. We use two of them as distinct real-noise
# characters and deliberately drop the third:
#   0-9   animals            -> natural
#   10-19 natural soundscapes/water -> natural
#   20-29 human non-speech   -> EXCLUDED: neither ambient-natural nor mechanical, and speech-like
#                               transients would be a third condition rather than part of either.
#   30-39 interior/domestic  -> mechanical
#   40-49 exterior/urban     -> mechanical
# That leaves 800 clips per category, so neither is better sampled than the other.
ESC50_TARGETS = {"natural": range(0, 20), "mechanical": range(30, 50)}


def dataset_fingerprint():
    """Short digest of the config that produced this data root's windows.

    Folded into every noise seed so a noisy set generated against one build of the dataset can
    never be silently reused against a different one.
    """
    return hashlib.sha256(config_fingerprint_json().encode()).hexdigest()[:16]


def window_seed(window_id, noise_type, fingerprint=None):
    """Stable 32-bit seed from (dataset fingerprint, window, noise type).

    NOTE the absence of SNR: one realization is drawn per window and noise type, then scaled to
    every SNR, so the only thing changing along the curve is level.
    """
    fp = dataset_fingerprint() if fingerprint is None else fingerprint
    key = f"{fp}|{window_id}|{noise_type}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def load_esc50_index():
    """{noise_type: [paths]} for the real-noise categories, from ESC-50's own metadata."""
    if not ESC50_DIR.exists():
        raise SystemExit(f"ESC-50 audio not found at {ESC50_DIR}. Set RISE_NOISE_ROOT or download it.")
    if not ESC50_META.exists():
        raise SystemExit(
            f"ESC-50 metadata not found at {ESC50_META}. It maps clips to categories and is what\n"
            "splits natural from mechanical. Fetch it with:\n"
            "  curl -sL -o '{}' https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
            .format(ESC50_META))
    meta = pd.read_csv(ESC50_META)
    index = {}
    for nt, targets in ESC50_TARGETS.items():
        sel = meta[meta.target.isin(list(targets))]
        paths = [ESC50_DIR / f for f in sorted(sel.filename)]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} ESC-50 clips listed in metadata are missing, e.g. {missing[0].name}")
        index[nt] = paths
    return index


def draw_noise(noise_type, rng, esc_index):
    """One CLIP_LEN noise realization for this window+type. Called ONCE per window, not per SNR."""
    if noise_type == "white":
        return rng.standard_normal(CLIP_LEN).astype(np.float32)
    paths = esc_index[noise_type]
    p = paths[rng.integers(len(paths))]
    n, sr_n = sf.read(str(p), dtype="float32", always_2d=False)
    if n.ndim > 1:
        n = n.mean(axis=1)
    if sr_n != SR:
        n = librosa.resample(n, orig_sr=sr_n, target_sr=SR)
    if n.size < CLIP_LEN:                      # tile short noise clips to length
        n = np.tile(n, int(np.ceil(CLIP_LEN / max(n.size, 1))))
    start = rng.integers(0, max(n.size - CLIP_LEN, 0) + 1)
    seg = n[start:start + CLIP_LEN]
    if np.sqrt(np.mean(seg ** 2)) < 1e-8:      # avoid an all-silent noise segment
        seg = seg + 1e-6 * rng.standard_normal(seg.size)
    return seg.astype(np.float32)


def mix_at_snr(clean, noise, snr_db):
    """Scale `noise` so the mixture sits at exactly `snr_db`, then add. No re-normalization."""
    p_sig = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    alpha = np.sqrt(p_sig / (p_noise * (10 ** (snr_db / 10.0))))
    return (clean + alpha * noise).astype(np.float32), alpha, p_sig


def measured_snr(clean, noisy):
    """Recover the achieved SNR from the mixture (added noise = noisy - clean)."""
    added = noisy - clean
    return 10 * np.log10(np.mean(clean ** 2) / np.mean(added ** 2))


def load_clean(rel_path):
    y, _ = sf.read(str(ROOT / rel_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - y.size))
    return y[:CLIP_LEN]


def test_windows():
    w = pd.read_csv(WINDOWS_CSV)
    return w[w.split == "test"].reset_index(drop=True)


def out_path(noise_type, snr, window_id):
    return NOISY_DIR / noise_type / f"snr{snr}" / f"{window_id}.wav"


def window_id_of(rel_path):
    return Path(rel_path).stem


# --------------------------------------------------------------------------- validate
def validate(n_samples=5):
    w = test_windows()
    esc = load_esc50_index()
    fp = dataset_fingerprint()
    print(f"data root  : {ROOT}")
    print(f"fingerprint: {fp}")
    print(f"test windows: {len(w)} | ESC-50: " +
          ", ".join(f"{k} {len(v)}" for k, v in esc.items()))
    print(f"\nexpected P_signal = TARGET_RMS^2 = {TARGET_RMS**2:.6f} (windows are RMS-normalized)")
    rows = []
    rng_pick = np.random.default_rng(0)
    idx = rng_pick.choice(len(w), size=min(n_samples, len(w)), replace=False)
    for i in idx:
        rel = w.iloc[i]["window_path"]
        wid = window_id_of(rel)
        clean = load_clean(rel)
        p_sig = float(np.mean(clean ** 2))
        assert abs(np.sqrt(p_sig) - TARGET_RMS) < 0.05, \
            f"{wid}: RMS {np.sqrt(p_sig):.4f} far from TARGET_RMS {TARGET_RMS}"
        for nt in NOISE_TYPES:
            # ONE realization, reused across every SNR (see module docstring)
            noise = draw_noise(nt, np.random.default_rng(window_seed(wid, nt, fp)), esc)
            for snr in SNRS:
                noisy, _, _ = mix_at_snr(clean, noise, snr)
                rows.append({"window": wid[:30], "noise": nt, "target_dB": snr,
                             "measured_dB": round(measured_snr(clean, noisy), 4),
                             "err_dB": round(measured_snr(clean, noisy) - snr, 5),
                             "peak": round(float(np.abs(noisy).max()), 3)})
    df = pd.DataFrame(rows)
    print("\nmeasured vs target SNR (all sampled windows):")
    print(df.groupby(["noise", "target_dB"])
            .agg(measured_dB=("measured_dB", "mean"),
                 max_abs_err_dB=("err_dB", lambda s: s.abs().max()),
                 max_peak=("peak", "max")).round(5).to_string())
    worst = df["err_dB"].abs().max()
    print(f"\nworst |measured - target| = {worst:.6f} dB  -> {'PASS' if worst < 0.1 else 'FAIL'} (<0.1 dB)")

    # the realization must be identical across SNRs -- that is the point of the seed change
    rel = w.iloc[idx[0]]["window_path"]; wid = window_id_of(rel); clean = load_clean(rel)
    for nt in NOISE_TYPES:
        noise = draw_noise(nt, np.random.default_rng(window_seed(wid, nt, fp)), esc)
        added = [(mix_at_snr(clean, noise, s)[0] - clean) for s in SNRS]
        # each added signal must be a pure rescaling of the same waveform
        base = added[0] / (np.linalg.norm(added[0]) + 1e-12)
        cos = [float(abs(np.dot(a / (np.linalg.norm(a) + 1e-12), base))) for a in added]
        ok = all(c > 1 - 1e-6 for c in cos)
        print(f"  {nt:<11} same realization across all SNRs: {ok}  (min cos {min(cos):.8f})")

    listen = NOISY_DIR / "_validation_samples"
    listen.mkdir(parents=True, exist_ok=True)
    sf.write(str(listen / f"{wid}__clean.wav"), clean, SR, subtype="FLOAT")
    for nt in NOISE_TYPES:
        noise = draw_noise(nt, np.random.default_rng(window_seed(wid, nt, fp)), esc)
        noisy, _, _ = mix_at_snr(clean, noise, 0)
        sf.write(str(listen / f"{wid}__{nt}_snr0.wav"), noisy, SR, subtype="FLOAT")
    print(f"\nlisten to these (0 dB = instrument and noise at EQUAL power):\n  {listen}")


# --------------------------------------------------------------------------- generate
def generate():
    w = test_windows()
    esc = load_esc50_index()
    fp = dataset_fingerprint()
    total = len(w) * len(NOISE_TYPES) * len(SNRS)
    print(f"generating {total} noisy windows ({len(w)} test windows x "
          f"{len(NOISE_TYPES)} noise types x {len(SNRS)} SNRs) under {NOISY_DIR}")
    print(f"dataset fingerprint: {fp}")
    for nt in NOISE_TYPES:
        for snr in SNRS:
            (NOISY_DIR / nt / f"snr{snr}").mkdir(parents=True, exist_ok=True)
    made = 0
    for rel in w["window_path"]:
        wid = window_id_of(rel)
        clean = load_clean(rel)
        for nt in NOISE_TYPES:
            # draw ONCE per (window, noise type), then scale to every SNR
            noise = draw_noise(nt, np.random.default_rng(window_seed(wid, nt, fp)), esc)
            for snr in SNRS:
                noisy, _, _ = mix_at_snr(clean, noise, snr)
                sf.write(str(out_path(nt, snr, wid)), noisy, SR, subtype="FLOAT")
                made += 1
        if made % 3000 < len(NOISE_TYPES) * len(SNRS):
            print(f"  {made}/{total}")
    meta = {"dataset_fingerprint": fp, "snrs": SNRS, "noise_types": NOISE_TYPES,
            "n_test_windows": int(len(w)), "n_files": made,
            "esc50_counts": {k: len(v) for k, v in esc.items()},
            "esc50_targets": {k: [min(v), max(v)] for k, v in ESC50_TARGETS.items()},
            "seed_scheme": "sha256(dataset_fingerprint|window_id|noise_type)[:4] — SNR excluded",
            "one_realization_scaled_to_all_snrs": True}
    (NOISY_DIR / "noise_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"done: {made} files. 'clean' is NOT duplicated — it points at {WORK/'windows'}.")
    print(f"wrote {NOISY_DIR/'noise_manifest.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="check SNR math on a few windows, then stop")
    ap.add_argument("--generate", action="store_true", help="write all conditions")
    a = ap.parse_args()
    if a.validate:
        validate()
    elif a.generate:
        generate()
    else:
        ap.error("pass --validate or --generate")


if __name__ == "__main__":
    main()
