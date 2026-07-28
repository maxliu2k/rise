"""noise_sweep.py — build the shared noisy TEST windows, ONCE, for every model to reuse.

DATASET-AGNOSTIC: driven by config's DATA_ROOT, so the identical protocol (same SNR levels,
same noise types, same SNR math, same seed scheme) runs on TinySOL and on Philharmonia. Only
the data root changes. That is what makes the degradation curves comparable across datasets.

DESIGN RULES (these keep the experiment valid):
  * Generated ONCE and written to disk so every model featurizes the SAME noisy audio ->
    predictions stay paired for McNemar / bootstrap across models.
  * TEST SPLIT ONLY. train/val are never touched; models stay clean-trained.
  * NO re-normalization after mixing (that would change the effective SNR). Written as
    float32 WAV so values beyond +-1.0 keep headroom instead of clipping at low SNR.
  * Deterministic per (window_id, noise_type, snr): the same window always gets the same
    noise realization, reproducible across reruns and identical for every model.
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

import argparse, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, librosa
from instrument_robustness.config import ROOT, PIPE, WORK, SR, TARGET_RMS
warnings.filterwarnings("ignore")

SNRS = [20, 10, 0, -5]                 # plus "clean", which is shared (not duplicated)
NOISE_TYPES = ["white", "real"]
NOISY_DIR = WORK / "windows_noisy"
ESC50_DIR = Path(os.environ.get("RISE_NOISE_ROOT", Path.home() / "Downloads/noise_sources")) \
    / "ESC-50-master" / "audio"
CLIP_LEN = int(round(3.0 * SR))


def window_seed(window_id, noise_type, snr):
    """Stable 32-bit seed from (window, noise type, SNR) — reproducible across machines/reruns."""
    key = f"{window_id}|{noise_type}|{snr}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def load_esc50_paths():
    if not ESC50_DIR.exists():
        raise SystemExit(f"ESC-50 not found at {ESC50_DIR}. Set RISE_NOISE_ROOT or download it.")
    return sorted(ESC50_DIR.glob("*.wav"))


def get_real_noise(rng, esc_paths):
    """Random 3 s segment from a random ESC-50 clip, resampled to the working rate."""
    p = esc_paths[rng.integers(len(esc_paths))]
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
    w = pd.read_csv(PIPE / "windows.csv")
    return w[w.split == "test"].reset_index(drop=True)


def out_path(noise_type, snr, window_id):
    return NOISY_DIR / noise_type / f"snr{snr}" / f"{window_id}.wav"


def window_id_of(rel_path):
    return Path(rel_path).stem


# --------------------------------------------------------------------------- validate (B5)
def validate(n_samples=5):
    w = test_windows()
    esc = load_esc50_paths()
    print(f"data root : {ROOT}")
    print(f"test windows: {len(w)} | ESC-50 clips: {len(esc)}")
    print(f"\nexpected P_signal = TARGET_RMS^2 = {TARGET_RMS**2:.6f} (windows are RMS-normalized)")
    rows = []
    rng_pick = np.random.default_rng(0)
    idx = rng_pick.choice(len(w), size=min(n_samples, len(w)), replace=False)
    for i in idx:
        rel = w.iloc[i]["window_path"]
        wid = window_id_of(rel)
        clean = load_clean(rel)
        p_sig = float(np.mean(clean ** 2))
        # B3 assert: signal power should match the RMS-normalization contract
        assert abs(np.sqrt(p_sig) - TARGET_RMS) < 0.05, \
            f"{wid}: RMS {np.sqrt(p_sig):.4f} far from TARGET_RMS {TARGET_RMS}"
        for nt in NOISE_TYPES:
            for snr in SNRS:
                rng = np.random.default_rng(window_seed(wid, nt, snr))
                noise = (rng.standard_normal(CLIP_LEN).astype(np.float32) if nt == "white"
                         else get_real_noise(rng, esc))
                noisy, alpha, _ = mix_at_snr(clean, noise, snr)
                rows.append({"window": wid[:34], "noise": nt, "target_dB": snr,
                             "measured_dB": round(measured_snr(clean, noisy), 4),
                             "err_dB": round(measured_snr(clean, noisy) - snr, 5),
                             "peak": round(float(np.abs(noisy).max()), 3)})
    df = pd.DataFrame(rows)
    print("\nmeasured vs target SNR (all sampled windows):")
    print(df.groupby(["noise", "target_dB"])
            .agg(measured_dB=("measured_dB", "mean"), max_abs_err_dB=("err_dB", lambda s: s.abs().max()),
                 max_peak=("peak", "max")).round(5).to_string())
    worst = df["err_dB"].abs().max()
    print(f"\nworst |measured - target| = {worst:.6f} dB  -> {'PASS' if worst < 0.1 else 'FAIL'} (<0.1 dB)")

    # B5: write a 0 dB sample to listen to (one white, one real)
    listen = NOISY_DIR / "_validation_samples"
    listen.mkdir(parents=True, exist_ok=True)
    rel = w.iloc[idx[0]]["window_path"]
    wid = window_id_of(rel)
    clean = load_clean(rel)
    sf.write(str(listen / f"{wid}__clean.wav"), clean, SR, subtype="FLOAT")
    for nt in NOISE_TYPES:
        rng = np.random.default_rng(window_seed(wid, nt, 0))
        noise = (rng.standard_normal(CLIP_LEN).astype(np.float32) if nt == "white"
                 else get_real_noise(rng, esc))
        noisy, _, _ = mix_at_snr(clean, noise, 0)
        sf.write(str(listen / f"{wid}__{nt}_snr0.wav"), noisy, SR, subtype="FLOAT")
    print(f"\nlisten to these (0 dB = instrument and noise at EQUAL power):\n  {listen}")
    for f in sorted(listen.glob("*.wav")):
        print(f"    {f.name}")


# --------------------------------------------------------------------------- generate (B6)
def generate():
    w = test_windows()
    esc = load_esc50_paths()
    total = len(w) * len(NOISE_TYPES) * len(SNRS)
    print(f"generating {total} noisy windows ({len(w)} test windows x "
          f"{len(NOISE_TYPES)} noise types x {len(SNRS)} SNRs) under {NOISY_DIR}")
    made = 0
    for nt in NOISE_TYPES:
        for snr in SNRS:
            (NOISY_DIR / nt / f"snr{snr}").mkdir(parents=True, exist_ok=True)
            for rel in w["window_path"]:
                wid = window_id_of(rel)
                clean = load_clean(rel)
                rng = np.random.default_rng(window_seed(wid, nt, snr))
                noise = (rng.standard_normal(CLIP_LEN).astype(np.float32) if nt == "white"
                         else get_real_noise(rng, esc))
                noisy, _, _ = mix_at_snr(clean, noise, snr)
                # float32 WAV: no re-normalization, no clipping at low SNR
                sf.write(str(out_path(nt, snr, wid)), noisy, SR, subtype="FLOAT")
                made += 1
            print(f"  {nt} snr{snr}: {len(w)} files")
    print(f"done: {made} files. 'clean' is NOT duplicated — it points at {WORK/'windows'}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="B5 check on a few windows, then stop")
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
