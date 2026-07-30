"""Step 2 - Trim leading/trailing silence from each resampled file.

- librosa.effects.trim with top_db=TRIM_TOP_DB (~30): energy-based, conservative.
- Keep note onsets: if trimming would leave < MIN_TRIM_S of audio (e.g. very soft notes
  where the whole thing reads as "quiet"), keep the untrimmed resampled audio and flag it.
- Recompute and store post-trim duration.
Output: work/trimmed/*.wav and manifest_trimmed.csv
"""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from pathlib import Path

from instrument_robustness.config import (ROOT, RESAMPLED, TRIMMED, PIPE, SR, TRIM_TOP_DB,
                                          MIN_TRIM_S, MANIFEST_RESAMPLED, MANIFEST_TRIMMED,
                                          assert_artifact_fingerprint, worker_count,
                                          write_artifact_fingerprint)
warnings.filterwarnings("ignore")

def trim_one(rel_resampled):
    src = ROOT / rel_resampled
    # pathlib, not str.split("resampled/"): step1 stores paths with the OS separator, so a
    # hard-coded "/" raises IndexError on Windows and the whole stage dies at the first file.
    dst = TRIMMED / Path(rel_resampled).relative_to(RESAMPLED.relative_to(ROOT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        y, _ = librosa.load(str(src), sr=SR, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        flag = "ok"
        if yt.size < int(MIN_TRIM_S * SR):     # too aggressive -> fall back to untrimmed
            yt, flag = y, "kept_untrimmed"
        sf.write(str(dst), yt, SR, subtype="PCM_16")
        return (rel_resampled, dst.relative_to(ROOT).as_posix(), round(yt.size / SR, 4), flag)
    except Exception as e:
        return (rel_resampled, None, 0.0, f"error:{type(e).__name__}")

def main():
    assert_artifact_fingerprint(MANIFEST_RESAMPLED, "step1_resample")
    df = pd.read_csv(MANIFEST_RESAMPLED)
    df = df[df.status == "ok"].copy()
    paths = df["resampled_path"].tolist()
    print(f"trimming {len(paths)} files (top_db={TRIM_TOP_DB}) ...")
    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=worker_count()) as ex:
        futs = [ex.submit(trim_one, p) for p in paths]
        for f in as_completed(futs):
            rp, tp, dur, flag = f.result()
            res[rp] = (tp, dur, flag)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(paths)}")
    df["trimmed_path"] = df["resampled_path"].map(lambda p: res[p][0])
    df["trimmed_dur_s"] = df["resampled_path"].map(lambda p: res[p][1])
    df["trim_flag"] = df["resampled_path"].map(lambda p: res[p][2])

    print("\ntrim flags:", df["trim_flag"].value_counts().to_dict())

    # Step 1 filters its own failures (status == "ok"); this stage never filtered or checked its
    # own. A row that failed to decode here kept trimmed_path = None, flowed into step 3 -- where
    # it counted toward the 70/15/15 fractions as a phantom source -- and then took step 4 down two
    # stages later with a TypeError on `ROOT / nan`. Fail here, where the cause is legible.
    failed = df[df["trim_flag"].str.startswith("error:")]
    assert failed.empty, (
        f"{len(failed)} file(s) failed to trim. They would be split as phantom sources and crash "
        f"step 4:\n" + failed[["resampled_path", "trim_flag"]].head(10).to_string(index=False))
    print(f"duration  resampled -> trimmed (median): "
          f"{df.resampled_dur_s.median():.3f}s -> {df.trimmed_dur_s.median():.3f}s")
    print("median trimmed duration per class:")
    print(df.groupby("label").trimmed_dur_s.median().round(3).to_string())
    out = MANIFEST_TRIMMED
    df.to_csv(out, index=False)
    write_artifact_fingerprint(out, "step2_trim")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
