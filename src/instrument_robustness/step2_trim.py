"""Step 2 - Trim leading/trailing silence from each resampled file.

- librosa.effects.trim with top_db=TRIM_TOP_DB (~30): energy-based, conservative.
- Keep note onsets: if trimming would leave < MIN_TRIM_S of audio (e.g. very soft notes
  where the whole thing reads as "quiet"), keep the untrimmed resampled audio and flag it.
- Recompute and store post-trim duration.
Output: work/trimmed/*.wav and manifest_9_trimmed.csv
"""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from instrument_robustness.config import ROOT, TRIMMED, PIPE, SR, TRIM_TOP_DB, MIN_TRIM_S
warnings.filterwarnings("ignore")

def trim_one(rel_resampled):
    src = ROOT / rel_resampled
    dst = TRIMMED / rel_resampled.split("resampled/", 1)[1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        y, _ = librosa.load(str(src), sr=SR, mono=True)
        yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
        flag = "ok"
        if yt.size < int(MIN_TRIM_S * SR):     # too aggressive -> fall back to untrimmed
            yt, flag = y, "kept_untrimmed"
        sf.write(str(dst), yt, SR, subtype="PCM_16")
        return (rel_resampled, str(dst.relative_to(ROOT)), round(yt.size / SR, 4), flag)
    except Exception as e:
        return (rel_resampled, None, 0.0, f"error:{type(e).__name__}")

def main():
    df = pd.read_csv(PIPE / "manifest_9_resampled.csv")
    df = df[df.status == "ok"].copy()
    paths = df["resampled_path"].tolist()
    print(f"trimming {len(paths)} files (top_db={TRIM_TOP_DB}) ...")
    res, done = {}, 0
    with ProcessPoolExecutor() as ex:
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
    print(f"duration  resampled -> trimmed (median): "
          f"{df.resampled_dur_s.median():.3f}s -> {df.trimmed_dur_s.median():.3f}s")
    print("median trimmed duration per class:")
    print(df.groupby("label").trimmed_dur_s.median().round(3).to_string())
    out = PIPE / "manifest_9_trimmed.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
