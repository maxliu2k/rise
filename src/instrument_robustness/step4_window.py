"""Step 4 - Window every (resampled, trimmed) file to fixed 3.0 s (kills Flag 2).

- 3.0 s windows, NO overlap (hop 3.0 s). See config for rationale.
- Every window inherits its source's label AND its source's split tag from Step 3.
  Never re-split at window level.
- Short/only windows are zero-padded to 3.0 s. A trailing window with < MIN_WINDOW_CONTENT_S
  of real content is dropped, unless it is the source's only window.
Output: work/windows/*.wav and windows.csv (window_path,label,split,source_path,start_time,content_s)
Also writes per-class WINDOW counts per split into the report block returned to caller.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, librosa, soundfile as sf
from instrument_robustness.config import (ROOT, WINDOWS, PIPE, SR, WINDOW_S, HOP_S,
                    MIN_WINDOW_CONTENT_S, TARGET_LABELS)
warnings.filterwarnings("ignore")

WIN = int(round(WINDOW_S * SR))
HOP = int(round(HOP_S * SR))
MIN_CONTENT = int(round(MIN_WINDOW_CONTENT_S * SR))

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
        if content < WIN:                          # zero-pad short/only/final window
            seg = np.pad(seg, (0, WIN - content))
        wpath = WINDOWS / f"{stem}_w{idx:03d}.wav"
        wpath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wpath), seg, SR, subtype="PCM_16")
        out.append((str(wpath.relative_to(ROOT)), label, split, source_path,
                    round(start / SR, 4), round(content / SR, 4)))
        idx += 1
    return out

def main():
    sp = pd.read_csv(PIPE / "splits.csv")
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
    win.to_csv(PIPE / "windows.csv", index=False)

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
             "-> lean on class weights / window-count capping accordingly; report macro-F1."]
    (PIPE / "_step4_report_block.txt").write_text("\n".join(block))
    print(f"\nwrote {PIPE / 'windows.csv'} and report block")

if __name__ == "__main__":
    main()
