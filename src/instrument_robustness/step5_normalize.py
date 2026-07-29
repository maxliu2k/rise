"""Step 5 - Per-window loudness normalization.

RMS-normalize each window to TARGET_RMS (with a peak guard against clipping), IN PLACE in
work/windows/. Done AFTER windowing so each window is individually normalized and later
SNR math (noise experiments) is well-defined against a known per-window RMS.
RMS-to-target is idempotent, so re-running is safe. Adds pre/post RMS columns to windows.csv.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from instrument_robustness.config import (
    ROOT,
    SR,
    TARGET_RMS,
    WINDOWS_CSV,
    assert_artifact_fingerprint,
    write_artifact_fingerprint,
)
warnings.filterwarnings("ignore")

def norm_one(wrel):
    """Scale one window to TARGET_RMS. Returns (path, pre_rms, post_rms, status).

    status is "ok", "peak_guarded" (scaled down to avoid clipping, so post < target), or "silent"
    (left untouched). The three were previously indistinguishable downstream, and main() reported
    all of them as peak-guarded.
    """
    y, _ = librosa.load(str(ROOT / wrel), sr=SR, mono=True)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:                      # silent window: leave as-is
        return (wrel, rms, rms, "silent")
    scale = TARGET_RMS / rms
    peak = np.abs(y).max() * scale
    status = "ok"
    if peak > 0.99:                     # peak guard: don't clip
        scale *= 0.99 / peak
        status = "peak_guarded"
    yn = y * scale
    sf.write(str(ROOT / wrel), yn, SR, subtype="PCM_16")
    post = float(np.sqrt(np.mean(yn ** 2)))
    return (wrel, rms, post, status)

def main():
    # Accepts its own stage as well as step 4's. This stage edits windows.csv IN PLACE and
    # re-stamps the sidecar as step5_normalize, so demanding "step4_window" made
    # `run_pipeline --from step5_normalize` fail permanently after any successful run -- and that
    # is the exact command the runner prints when a later stage fails, so the suggested fix looped.
    # Re-running is safe: RMS-to-target converges (a peak-guarded window re-scales to itself).
    assert_artifact_fingerprint(WINDOWS_CSV, ("step4_window", "step5_normalize"))
    win = pd.read_csv(WINDOWS_CSV)
    paths = win["window_path"].tolist()
    print(f"normalizing {len(paths)} windows to RMS={TARGET_RMS} ...")
    res, done = {}, 0
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(norm_one, p) for p in paths]
        for f in as_completed(futs):
            wrel, pre, post, status = f.result()
            res[wrel] = (pre, post, status)
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(paths)}")
    win["pre_norm_rms"] = win["window_path"].map(lambda p: round(res[p][0], 5))
    win["post_norm_rms"] = win["window_path"].map(lambda p: round(res[p][1], 5))
    norm_status = win["window_path"].map(lambda p: res[p][2])
    WINDOWS_CSV.parent.mkdir(parents=True, exist_ok=True)
    win.to_csv(WINDOWS_CSV, index=False)
    write_artifact_fingerprint(WINDOWS_CSV, "step5_normalize")

    counts = norm_status.value_counts().to_dict()
    peaked, silent = counts.get("peak_guarded", 0), counts.get("silent", 0)
    print(f"\ndone. at target: {counts.get('ok', 0)} | "
          f"below target (peak-guard, legitimate): {peaked} ({peaked/len(win)*100:.1f}%) | "
          f"silent: {silent}")

    # A silent window is not a loudness edge case, it is a defective source: step 2 trims to the
    # attack and step 4 tiles real signal, so digital silence here means the recording has none.
    # It would also make the noise experiment's SNR undefined (signal power zero), and the old
    # combined counter reported it as an ordinary peak-guard.
    assert silent == 0, (
        f"{silent} window(s) are digitally silent after trimming and tiling -- a defective source, "
        f"and an undefined SNR for the noise sweep:\n"
        + "\n".join(f"    {p}" for p in win.loc[norm_status == 'silent', 'window_path'][:10]))
    print("post-norm RMS: median = %.4f (target %.4f)" %
          (win["post_norm_rms"].median(), TARGET_RMS))
    print(f"updated {WINDOWS_CSV} with pre/post RMS columns")

if __name__ == "__main__":
    main()
