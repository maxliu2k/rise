"""Step 1 - Resample everything to a common rate (kills Flag 1) and go mono.

- Decode every file, resample to SR (22050 Hz), mono.
- NO loudness normalization here (that is per-window, later).
- Write .wav to work/resampled/, preserving the source's relative path.
- Persist source -> resampled mapping in manifest_9_resampled.csv.
- Sanity check: re-measure the per-instrument high-frequency ceiling AFTER resampling.
  All instruments should now be capped at/below Nyquist (11025 Hz) with no differential
  brick wall. If one instrument still shows a distinctly lower wall, STOP and investigate.
"""
import warnings, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from instrument_robustness.config import ROOT, RESAMPLED, MANIFEST_9, PIPE, SR, TARGET_LABELS
warnings.filterwarnings("ignore")

def resample_one(rel_path):
    src = ROOT / rel_path
    dst = RESAMPLED / (rel_path.rsplit(".", 1)[0] + ".wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        y, _ = librosa.load(str(src), sr=SR, mono=True)
        if y.size == 0:
            return (rel_path, None, 0, 0.0, "empty")
        sf.write(str(dst), y, SR, subtype="PCM_16")
        return (rel_path, str(dst.relative_to(ROOT)), int(y.size), round(y.size / SR, 4), "ok")
    except Exception as e:
        return (rel_path, None, 0, 0.0, f"error:{type(e).__name__}")

def brickwall_hz(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < sr * 0.2:
        return None
    p = (np.abs(librosa.stft(y, n_fft=4096)) ** 2).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    p_db = 10 * np.log10(p / (p.max() + 1e-12) + 1e-12)
    idx = np.where(p_db > -75)[0]
    return freqs[idx.max()] if len(idx) else None

def sanity_check(df):
    rng = np.random.default_rng(0)
    print("\n=== post-resample frequency-ceiling check (Nyquist =", SR // 2, "Hz) ===")
    rows = []
    for inst in TARGET_LABELS:
        sub = df[(df.label == inst) & (df.status == "ok")]
        bright = sub[sub.dynamic.astype(str).str.contains("forte|fortissimo", case=False, na=False)]
        pool = (bright if len(bright) >= 30 else sub)["resampled_path"].tolist()
        sample = rng.choice(pool, size=min(40, len(pool)), replace=False)
        ceils = np.array([c for p in sample if (c := brickwall_hz(str(ROOT / p), SR)) is not None])
        rows.append((inst, round(np.percentile(ceils, 90)), round(ceils.max())))
    rep = pd.DataFrame(rows, columns=["instrument", "ceil_p90_Hz", "ceil_max_Hz"])
    print(rep.to_string(index=False))
    spread = rep.ceil_p90_Hz.max() - rep.ceil_p90_Hz.min()
    print(f"\np90 ceiling spread across instruments: {spread} Hz")
    ok = rep.ceil_max_Hz.max() <= SR // 2 + 5
    print("all ceilings <= Nyquist:", ok, "-> Flag 1 defused" if ok else "-> INVESTIGATE")
    return rep

def main():
    df = pd.read_csv(MANIFEST_9)
    paths = df["path"].tolist()
    print(f"resampling {len(paths)} files to {SR} Hz mono ...")
    results = {}
    done = 0
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(resample_one, p) for p in paths]
        for f in as_completed(futs):
            rel, rpath, n, dur, status = f.result()
            results[rel] = (rpath, n, dur, status)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(paths)}")
    df["resampled_path"] = df["path"].map(lambda p: results[p][0])
    df["resampled_dur_s"] = df["path"].map(lambda p: results[p][2])
    df["status"] = df["path"].map(lambda p: results[p][3])
    n_ok = (df.status == "ok").sum()
    print(f"\nresampled ok: {n_ok} | failures: {len(df) - n_ok}")
    if (df.status != "ok").any():
        print(df[df.status != "ok"][["path", "status"]].to_string(index=False))
    out = PIPE / "manifest_9_resampled.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    sanity_check(df)

if __name__ == "__main__":
    main()
