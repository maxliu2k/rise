"""Render test clips at each SNR level to WAV, so a human can hear what the model is fed.

Reuses noise_eval.add_noise_at_snr — the exact function the sweep uses — so the audio is
bit-for-bit what the model saw, not a lookalike. Renders from the cached waveforms (already
trimmed to 22.05 kHz mono), for the same reason.

The point of the exercise: 40 dB SNR sounds essentially clean to a listener, yet the model
has already lost ~15 points of balanced accuracy there. Listen and confirm.

    python -m instrument_robustness.audio_demo

Writes outputs/audio_demo/:
    <instrument>_<snr>dB.wav      one file per (instrument, level)
    <instrument>_montage.wav      clean -> noisiest in one file, short gaps between
"""

import sys

import numpy as np
import soundfile as sf

from ..config import CLASSES, OUTPUTS, SEED, SR, WAVE_DIR
from .noise_eval import add_noise_at_snr
from ..cnn_core import load_manifest

# The levels worth hearing: 60 dB (no audible change), down through the knee (40-30), to
# 0 dB (noise as loud as the note). Ordered loud-signal -> loud-noise for the montage.
DEMO_SNRS = (None, 60, 40, 30, 20, 10, 0)

# One clip per instrument. A spread of registers plus the story's protagonists: trumpet is
# the most fragile class, tuba the most robust (see FINDINGS §2-3).
DEMO_INSTRUMENTS = ("flute", "trumpet", "cello", "tuba")

GAP_S = 0.4          # silence between montage segments
MIN_CLIP_S = 1.2     # pick a clip with enough note to actually hear
# Prefer loud dynamics. A pianissimo note strips the very harmonics that identify the
# instrument — a soft trumpet has none of its brassy blare and genuinely sounds reed-like
# (both a human AND the model mistake it for a clarinet). "longest clip" alone biases toward
# soft sustained notes, which are unrepresentative; requiring forte/fortissimo fixes that.
LOUD_DYNAMICS = {"fortissimo", "forte"}


def pick_clip(records, instrument):
    """A loud, long-enough, representative test clip for an instrument. Prefers fortissimo/
    forte at >= MIN_CLIP_S; falls back progressively so a clip is always returned."""
    clips = [r for r in records if r["instrument"] == instrument]
    if not clips:
        return None
    long_enough = [r for r in clips if r["clip_seconds"] >= MIN_CLIP_S]
    loud_long = [r for r in long_enough if r["dynamics"] in LOUD_DYNAMICS]
    pool = loud_long or long_enough or clips
    return max(pool, key=lambda r: r["clip_seconds"])


def render_levels(y, clip_idx):
    """(label, waveform) for each SNR level, all at a shared gain so relative loudness is
    preserved and 0 dB doesn't clip. Louder noise therefore *sounds* louder, as it should."""
    out = []
    for cond_idx, snr in enumerate(DEMO_SNRS):
        if snr is None:
            out.append(("clean", y.copy()))
        else:
            rng = np.random.default_rng([SEED, cond_idx, clip_idx])
            noisy, _ = add_noise_at_snr(y, snr, rng)
            out.append((f"{snr}dB", noisy))
    # one gain across all levels of this clip: the 0 dB version has the largest peak, so
    # normalising the whole set by the global peak keeps the montage clip-free and honest
    peak = max(float(np.abs(w).max()) for _, w in out)
    g = 0.97 / peak if peak > 0 else 1.0
    return [(lab, (w * g).astype(np.float32)) for lab, w in out]


def main():
    manifest, splits, by_id = load_manifest()
    records = [by_id[i] for i in sorted(splits["test"])]

    out_dir = OUTPUTS / "audio_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)

    print(f"rendering {len(DEMO_INSTRUMENTS)} instruments x {len(DEMO_SNRS)} levels "
          f"-> {out_dir}\n")
    print("  SNR reminder: 60 dB = noise at 0.1% of signal amplitude (inaudible),")
    print("                40 dB = 1% (barely audible), 0 dB = noise as loud as the note\n")

    written = 0
    for instrument in DEMO_INSTRUMENTS:
        if instrument not in CLASSES:
            print(f"  {instrument}: not in CLASSES, skipping")
            continue
        rec = pick_clip(records, instrument)
        if rec is None:
            print(f"  {instrument}: no test clip found, skipping")
            continue
        clip_idx = sorted(splits["test"]).index(rec["id"])
        y = np.load(WAVE_DIR / f"{rec['id']}.npy")
        levels = render_levels(y, clip_idx)

        for lab, w in levels:
            sf.write(out_dir / f"{instrument}_{lab}.wav", w, SR, subtype="PCM_16")
            written += 1

        montage = []
        for lab, w in levels:
            montage.append(w)
            montage.append(gap)
        sf.write(out_dir / f"{instrument}_montage.wav",
                 np.concatenate(montage), SR, subtype="PCM_16")

        labels = " -> ".join(lab for lab, _ in levels)
        print(f"  {instrument:<9} ({rec['clip_seconds']:.2f}s clip): {labels}")

    print(f"\nwrote {written} clips + {len(DEMO_INSTRUMENTS)} montages to {out_dir}")
    print("start with the *_montage.wav files: clean first, then progressively louder noise.")


if __name__ == "__main__":
    main()
