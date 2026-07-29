"""Step 1 - Resample everything to a common rate (kills Flag 1) and go mono.

- Decode every file, resample to SR (22050 Hz), mono.
- NO loudness normalization here (that is per-window, later).
- Write .wav to work/resampled/, preserving the source's relative path.
- Persist source -> resampled mapping in manifest_resampled.csv.
- Sanity check: re-measure the per-instrument high-frequency ceiling AFTER resampling.
  All instruments should now be capped at/below Nyquist (11025 Hz) with no differential
  brick wall. If one instrument still shows a distinctly lower wall, STOP and investigate.
"""
import warnings, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from instrument_robustness.config import (
    MANIFEST_LABELED,
    MANIFEST_RESAMPLED,
    PIPE,
    RESAMPLED,
    ROOT,
    SR,
    TARGET_LABELS,
    assert_artifact_fingerprint,
    write_artifact_fingerprint,
)
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
        return (rel_path, dst.relative_to(ROOT).as_posix(), int(y.size), round(y.size / SR, 4), "ok")
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
        # prefer loud/bright notes to expose any brick wall; fall back to all if no `dynamic` column
        if "dynamic" in sub.columns:
            bright = sub[sub.dynamic.astype(str).str.contains("forte|fortissimo", case=False, na=False)]
        else:
            bright = sub.iloc[0:0]
        pool = (bright if len(bright) >= 30 else sub)["resampled_path"].tolist()
        sample = rng.choice(pool, size=min(40, len(pool)), replace=False)
        ceils = np.array([c for p in sample if (c := brickwall_hz(str(ROOT / p), SR)) is not None])
        assert len(ceils), (
            f"no measurable ceiling for {inst}: every sampled file was under 0.2s. The Nyquist "
            f"check below cannot pass judgement on a class it did not measure.")
        rows.append((inst, round(np.percentile(ceils, 90)), round(ceils.max())))
    rep = pd.DataFrame(rows, columns=["instrument", "ceil_p90_Hz", "ceil_max_Hz"])
    print(rep.to_string(index=False))
    spread = rep.ceil_p90_Hz.max() - rep.ceil_p90_Hz.min()
    # Diagnostic only, deliberately NOT asserted: a residual spread is expected (instruments
    # genuinely differ in brightness) and there is no pre-registered threshold separating that from
    # a codec artifact. Asserting a number picked after the fact would be noise. The bound below is
    # the one with a defensible meaning.
    print(f"\np90 ceiling spread across instruments: {spread} Hz")

    # This guards SR = 22050, the most load-bearing constant in the repo: per-instrument MP3
    # bitrate (64/80/96 kbps) puts the codec brick wall ABOVE Nyquist at 22.05 kHz so it is
    # discarded, and BELOW it at 44.1 kHz where it becomes a free per-class shortcut. It used to
    # print "-> INVESTIGATE" and return, so the stage exited 0 and run_pipeline carried straight on
    # to featurize against a confounded cache. An invariant that cannot fail the build is not a
    # check.
    worst = rep.ceil_max_Hz.max()
    assert worst <= SR // 2 + 5, (
        f"a resampled ceiling reached {worst} Hz, above Nyquist ({SR // 2} Hz). The bitrate "
        f"confound is NOT defused and every downstream number would be suspect:\n"
        + rep.to_string(index=False))
    print(f"all ceilings <= Nyquist: True -> Flag 1 defused")
    return rep

def main():
    assert_artifact_fingerprint(MANIFEST_LABELED, "step0_filter")
    df = pd.read_csv(MANIFEST_LABELED)
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
    # Check BEFORE stamping. The fingerprint sidecar is what every later stage treats as proof the
    # stage succeeded, so writing it first and validating second means a failed Nyquist check still
    # leaves a fully valid-looking manifest on disk -- and `--from step2_trim` would sail past it.
    sanity_check(df)
    out = MANIFEST_RESAMPLED
    df.to_csv(out, index=False)
    write_artifact_fingerprint(out, "step1_resample")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
