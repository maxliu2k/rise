"""Step 4 - Window every (resampled, trimmed) file to fixed 3.0 s (kills Flag 2).

- 3.0 s windows, NO overlap (hop 3.0 s). At most MAX_WINDOWS_PER_SOURCE per file, which is
  1 by default: only the first window is guaranteed to start at a note onset. See config.
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
from pathlib import Path

import numpy as np, pandas as pd, librosa, soundfile as sf

from instrument_robustness.config import (
    worker_count,
    HOP_S,
    MAX_WINDOWS_PER_SOURCE,
    MIN_WINDOW_CONTENT_S,
    PIPE,
    ROOT,
    SPLITS_CSV,
    TRIMMED,
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
    trimmed_rel, label, note, split, source_path = args
    y, _ = librosa.load(str(ROOT / trimmed_rel), sr=SR, mono=True)
    n = len(y)
    # pathlib rather than splitting on "trimmed/": see step2. Windows separators break the
    # string form and take the stage down at the first file.
    stem = Path(trimmed_rel).relative_to(TRIMMED.relative_to(ROOT)).with_suffix("").as_posix()
    out = []
    idx = 0
    starts = list(range(0, max(1, n), HOP))[:MAX_WINDOWS_PER_SOURCE]
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
        out.append((wpath.relative_to(ROOT).as_posix(), label, note, split, source_path,
                    round(start / SR, 4), round(content / SR, 4)))
        idx += 1
    return out

def main():
    assert_artifact_fingerprint(SPLITS_CSV, "step3_split")
    sp = pd.read_csv(SPLITS_CSV)
    args = list(zip(sp["trimmed_path"], sp["label"], sp["note"], sp["split"], sp["source_path"]))
    print(f"windowing {len(args)} source files -> {WINDOW_S}s windows (no overlap) ...")
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=worker_count()) as ex:
        futs = [ex.submit(window_one, a) for a in args]
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 1500 == 0:
                print(f"  {done}/{len(args)} sources")
    win = pd.DataFrame(rows, columns=["window_path", "label", "note", "split",
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

    # Verify the leak guarantee at WINDOW level -- the level the models actually see.
    #
    # The previous check here was `win.groupby("source_path")["split"].nunique().max() == 1`. With
    # MAX_WINDOWS_PER_SOURCE = 1 each source yields exactly one window, so that maximum is 1 by
    # construction and the assert could not fail for any input. It is the same dead check step 3's
    # docstring retired, surviving one stage later and reading as reassurance.
    #
    # Grouping by PITCH instead can fail: it catches a window mis-tagged against its source, a
    # step-3 grouping regression, and any future windowing scheme that re-splits.
    grp = win["label"].astype(str) + "_" + win["note"].astype(str)
    spans = win.groupby(grp)["split"].nunique()
    leaked = spans[spans > 1]
    assert leaked.empty, (
        f"{len(leaked)} pitch-group(s) span more than one split AT WINDOW LEVEL:\n"
        + "\n".join(f"    {k}" for k in list(leaked.index)[:10])
        + ("\n    ..." if len(leaked) > 10 else ""))
    print(f"window-level leak check passed: {len(spans)} pitch-groups, none spanning splits")

    # Persist the compact Step-4 report block beside the authoritative windows manifest.
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
