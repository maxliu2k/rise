"""Step 5 - Per-window loudness normalization.

RMS-normalize each window to TARGET_RMS (with a peak guard against clipping), IN PLACE in
work/windows/. Done AFTER windowing so each window is individually normalized and later
SNR math (noise experiments) is well-defined against a known per-window RMS.
RMS-to-target is idempotent, so re-running is safe. Adds pre/post RMS columns to windows.csv.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from instrument_robustness.config import ROOT, PIPE, SR, TARGET_RMS
warnings.filterwarnings("ignore")

def norm_one(wrel):
    y, _ = librosa.load(str(ROOT / wrel), sr=SR, mono=True)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:                      # silent window: leave as-is
        return (wrel, rms, rms)
    scale = TARGET_RMS / rms
    peak = np.abs(y).max() * scale
    if peak > 0.99:                     # peak guard: don't clip
        scale *= 0.99 / peak
    yn = y * scale
    sf.write(str(ROOT / wrel), yn, SR, subtype="PCM_16")
    post = float(np.sqrt(np.mean(yn ** 2)))
    return (wrel, rms, post)

def main():
    win = pd.read_csv(PIPE / "windows.csv")
    paths = win["window_path"].tolist()
    print(f"normalizing {len(paths)} windows to RMS={TARGET_RMS} ...")
    res, done = {}, 0
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(norm_one, p) for p in paths]
        for f in as_completed(futs):
            wrel, pre, post = f.result()
            res[wrel] = (pre, post)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(paths)}")
    win["pre_norm_rms"] = win["window_path"].map(lambda p: round(res[p][0], 5))
    win["post_norm_rms"] = win["window_path"].map(lambda p: round(res[p][1], 5))
    win.to_csv(PIPE / "windows.csv", index=False)
    peaked = (np.abs(win["post_norm_rms"] - TARGET_RMS) > 1e-3).sum()
    print(f"\ndone. windows below target due to peak-guard: {peaked} "
          f"({peaked/len(win)*100:.1f}%)")
    print("post-norm RMS: median = %.4f (target %.4f)" %
          (win["post_norm_rms"].median(), TARGET_RMS))
    print(f"updated {PIPE / 'windows.csv'} with pre/post RMS columns")

if __name__ == "__main__":
    main()
