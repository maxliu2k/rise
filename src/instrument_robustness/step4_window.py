"""Step 4 - Window every (resampled, trimmed) file to fixed 3.0 s (kills Flag 2).

- 3.0 s windows, NO overlap (hop 3.0 s). See config for rationale.
- Every window inherits its source's label AND its source's split tag from Step 3.
  Never re-split at window level.
- Short/only windows are TILED (looped) to 3.0 s, never zero-padded. A trailing window with
  < MIN_WINDOW_CONTENT_S of real content is dropped, unless it is the source's only window.

  Zero-padding is not a stylistic choice here, it is a correctness bug for any noise experiment:
  librosa's power_to_db(ref=np.max) clamps digital silence to the -80 dB floor, injected noise
  fills that region, and the window lands outside the training distribution. Measured on the
  cnn-ensemble branch, a padded cache collapses to the majority class at EVERY SNR -- the noise
  sweep stops measuring anything. Tiling keeps every sample real signal.

  The obvious objection -- that a looped note's repeated attacks encode the source duration, which
  correlates with instrument -- was pre-registered and tested: if the model read tiling period,
  the extremes of the length distribution would be the most noise-robust. They are not. Tuba
  (shortest, ~5x repeats) and clarinet (longest, least tiled) both floor at 0.000 recall under
  noise, with no monotone length-vs-survival relationship. See FINDINGS S6 on cnn-ensemble.
Output: work/windows/*.wav and windows.csv (window_path,label,split,source_path,start_time,content_s)
Also writes per-class WINDOW counts per split into the report block returned to caller.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf

from instrument_robustness.config import (
    HOP_S,
    MIN_WINDOW_CONTENT_S,
    PIPE,
    ROOT,
    SPLITS_CSV,
    SR,
    TARGET_LABELS,
    WINDOWS,
    WINDOWS_CSV,
    WINDOW_S,
    assert_artifact_fingerprint,
    write_artifact_fingerprint,
)
warnings.filterwarnings("ignore")

WIN = int(round(WINDOW_S * SR))
HOP = int(round(HOP_S * SR))
MIN_CONTENT = int(round(MIN_WINDOW_CONTENT_S * SR))

def tile_to_length(seg, n):
    """Loop `seg` until it is exactly `n` samples. Every sample stays real signal.

    Preconditions: len(seg) >= 1 and len(seg) <= n.
    Postcondition: returns an array of exactly n samples, containing no synthesised silence.
    Raises: ValueError on an empty segment -- an empty window means the caller has a bug, and
    padding it to 3 s of zeros would hide that behind a plausible-looking file.
    """
    if len(seg) == 0:
        raise ValueError("cannot tile an empty segment")
    if len(seg) >= n:
        return seg[:n]
    reps = int(np.ceil(n / len(seg)))
    return np.tile(seg, reps)[:n]


def window_one(args):
    trimmed_rel, label, split, source_path = args
    y, _ = librosa.load(str(ROOT / trimmed_rel), sr=SR, mono=True)
    n = len(y)
    stem = trimmed_rel.split("trimmed/", 1)[1].rsplit(".", 1)[0]
    out = []
    idx = 0
    starts = list(range(0, max(1, n), HOP))
    for wi, start in enumerate(starts):
        seg = y[start:start + WIN]
        content = len(seg)
        if content < MIN_CONTENT and wi != 0:      # drop tiny trailing window (never the only one)
            continue
        if content < WIN:                          # TILE short/only/final window -- never pad
            seg = tile_to_length(seg, WIN)
        wpath = WINDOWS / f"{stem}_w{idx:03d}.wav"
        wpath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wpath), seg, SR, subtype="PCM_16")
        out.append((str(wpath.relative_to(ROOT)), label, split, source_path,
                    round(start / SR, 4), round(content / SR, 4)))
        idx += 1
    return out

def main():
    assert_artifact_fingerprint(SPLITS_CSV, "step3_split")
    sp = pd.read_csv(SPLITS_CSV)
    args = list(zip(sp["trimmed_path"], sp["label"], sp["split"], sp["source_path"]))
    print(f"windowing {len(args)} source files -> {WINDOW_S}s windows (no overlap) ...")
    rows, done = [], 0
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(window_one, a) for a in args]
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 1500 == 0:
                print(f"  {done}/{len(args)} sources")
    win = pd.DataFrame(rows, columns=["window_path", "label", "split",
                                      "source_path", "start_time", "content_s"])
    win = win.sort_values(["source_path", "start_time"]).reset_index(drop=True)
    PIPE.mkdir(parents=True, exist_ok=True)
    win.to_csv(WINDOWS_CSV, index=False)

    print(f"\ntotal windows: {len(win)}  (from {len(args)} sources)")
    counts = pd.crosstab(win["label"], win["split"])[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    counts = counts.loc[TARGET_LABELS]
    print("\nper-class WINDOW counts per split:")
    print(counts.to_string())
    imb = counts["total"].max() / counts["total"].min()
    print(f"\nwindow-level imbalance (max/min total): {imb:.1f}x  "
          f"[{counts['total'].idxmax()} {counts['total'].max()} vs "
          f"{counts['total'].idxmin()} {counts['total'].min()}]")

    # verify split inheritance: each source's windows all share one split tag
    bad = win.groupby("source_path")["split"].nunique().max()
    print("max distinct split tags within a single source (must be 1):", bad)
    assert bad == 1

    # persist the report block for pipeline_report.txt
    block = ["STEP 4 — WINDOW  (-> windows.csv, work/windows/)",
             f"{WINDOW_S}s windows, no overlap. total windows: {len(win)} from {len(args)} sources.",
             "", "per-class WINDOW counts per split:", counts.to_string(),
             f"\nwindow-level imbalance (max/min total): {imb:.1f}x",
             "-> lean on class weights / window-count capping accordingly.",
             "-> report BALANCED ACCURACY and MCC, not accuracy or macro-F1: macro-F1 RISES with",
             "   imbalance for a collapsed classifier (0.3333 at a 0.50 prior, 0.4737 at 0.90),",
             "   so it pays a dead model more on more imbalanced data. See FINDINGS S7."]
    (PIPE / "_step4_report_block.txt").write_text("\n".join(block))
    write_artifact_fingerprint(WINDOWS_CSV, "step4_window")
    print(f"\nwrote {WINDOWS_CSV} and report block")

if __name__ == "__main__":
    main()
