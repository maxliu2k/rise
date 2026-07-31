"""Shared constants for the instrument-classification study.

Everything tunable lives here so prep_data / train / noise_eval cannot drift apart.
Change CLASSES to rescope the study; nothing else should need editing.
"""

from pathlib import Path

# --- paths ---
# This file lives at src/instrument_robustness/config.py, so the repo root is two levels up.
# data/ and outputs/ stay at the repo root, not inside the package.
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
WAVE_DIR = DATA_CACHE / "wave"
SPEC_DIR = DATA_CACHE / "spec"
SPLITS_JSON = DATA_CACHE / "splits.json"
MANIFEST_JSON = DATA_CACHE / "manifest.json"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = OUTPUTS / "model.pt"
# One JSON per trained seed. metrics.json is rebuilt by aggregating ALL of these, so seeds may
# be trained in separate invocations without the aggregate silently covering only the last one.
SEED_METRICS_DIR = OUTPUTS / "seed_metrics"

# --- data source ---
# The official philharmonia.co.uk/assets/audio/samples/... URLs predate their site
# redesign and no longer resolve. This Internet Archive mirror is the working source.
# License: CC Attribution-ShareAlike 4.0.
ARCHIVE_BASE = "https://archive.org/download/philharmonicorchestrasamples"

# The orchestral core: 4 strings, 4 woodwinds, 4 brass. Every family represented, every
# class has >=433 clips after strict articulation filtering, imbalance ~1.97:1.
#
# Tuba earns its place beyond family balance: at As0-F4 it overlaps double-bass (C1-G4)
# almost exactly, giving one same-register / different-family pair. Most of this set is
# separable on pitch alone, so that pair is the closest thing here to a real test of whether
# the model learned timbre.
#
# Keep alphabetical: the ordering fixes the label indices, and a reordering would silently
# invalidate every saved checkpoint. (main standardised on a 9-class subset dropping
# double-bass/oboe/french-horn; the team reverted to these 12.)
CLASSES = (
    "bassoon", "cello", "clarinet", "double-bass", "flute", "french-horn",
    "oboe", "trombone", "trumpet", "tuba", "viola", "violin",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

FAMILY = {
    "violin": "strings", "viola": "strings", "cello": "strings", "double-bass": "strings",
    "flute": "woodwind", "oboe": "woodwind", "clarinet": "woodwind", "bassoon": "woodwind",
    "trumpet": "brass", "trombone": "brass", "french-horn": "brass", "tuba": "brass",
}

# The archive's zip names are NOT the instrument field inside the filenames: the zip uses
# spaces, the files use hyphens, and `cor anglais.zip` contains `english-horn_*.mp3`.
# Mapped explicitly rather than derived, so a bad guess fails loudly at download.
ZIP_NAME = {
    "double-bass": "double bass",
    "french-horn": "french horn",
    "bass-clarinet": "bass clarinet",
    "english-horn": "cor anglais",   # the one true mismatch
}

# --- articulation filtering ---
# Strict: one articulation per instrument — the plain, sustained, ordinary tone. Bowed
# strings call it `arco-normal`; everything else calls it `normal`. This dominates the
# library rather than being a slice of it (84-89% of files), so filtering costs little.
_ARCO = {"violin", "viola", "cello", "double-bass"}
STRICT_ARTICULATIONS = {c: ({"arco-normal"} if c in _ARCO else {"normal"}) for c in CLASSES}

# Fallback: the plain-sustained family. Still excludes tremolo, glissando, pizz, col legno,
# trills, and tonguing effects — only the timbrally-neutral ones.
SUSTAINED_ARTICULATIONS = {
    c: ({"arco-normal", "arco-detache", "arco-legato", "non-vibrato"} if c in _ARCO
        else {"normal", "tenuto", "nonlegato"})
    for c in CLASSES
}
MIN_STRICT_N = 200      # per-class floor below which strict is abandoned
MAX_IMBALANCE = 1.5     # above this ratio, train.py applies class weights

# --- audio ---
# SR is load-bearing for reasons unrelated to why it was chosen. The library's classes are
# encoded at three different bitrates (64/80/96 kbps) that cut across instrument families,
# so the MP3 encoder leaves a class-correlated spectral edge. Measured: every codec brick
# wall sits above 19 kHz, and the class-correlated difference above ~14 kHz. At SR=22050
# the Nyquist is 11025 Hz and all of it is discarded before the model sees anything.
# Raising SR toward 44100 puts the encoder INSIDE the analysis band and hands the model a
# perfect non-timbral shortcut. prep_data.check_bitrates() enforces this each run.
SR = 22050
CLIP_SECONDS = 3.0           # FIXED window: every clip is exactly this long
CLIP_SAMPLES = int(SR * CLIP_SECONDS)  # 66150
# 3.0s matches the team's shared window, so our CNN clips line up with the other models'
# without anyone reconciling two window sizes. It is a whole multiple of the library's coded
# note durations (0.25/0.5/1.0/1.5s), and truncates only the longest ~3% of notes (vs ~20%
# at 1.5s); those lose redundant sustain tails, not timbre (attack + early sustain carry it).
# Most notes are shorter than the window (median ~0.91s) and get TILED to fill it — see below.
TRIM_TOP_DB = 30

# Clips are FIXED LENGTH (CLIP_SAMPLES). A note shorter than the window is TILED — looped
# end-to-start until it fills the window — and a file longer than the window is truncated
# (chunked, see below). Every clip is the same size, which the shared .npz stacking requires
# and which removes clip-width as a class-correlated cue the model could otherwise exploit.
#
# Why tiling and not zero-padding:
#   zeros   — centered zero-padding actively BREAKS the noise sweep: power_to_db clamps
#             digital silence to the -80dB floor, that floor is ~61% of a median image, and
#             added noise fills it, collapsing the spectrogram's std and pushing the clip
#             outside the training distribution. Measured: majority-class collapse (0.65
#             acc, trumpet F1 0.00) at every SNR including a mild 20dB.
#   tiling  — looping the trimmed note keeps the window 100% real signal, so noise lands on
#             music at every point, not on a silence floor. The trimmed note starts near its
#             onset and ends near-zero (top_db=30), so each loop re-introduces the recorded
#             attack: the seam discontinuity measured ~0.2x a normal sample step (no click),
#             and the result resembles a re-articulated / tongued note of the same pitch — an
#             in-class sound, not an alien one. The loop is class-neutral (applied identically
#             to all 12 instruments); its only per-clip parameter, the repeat period = source
#             note length, carries just the weak note-length signal, which MediumCNN's
#             AdaptiveAvgPool2d averages over the time axis and largely discards.

# A file of duration d yields min(floor(d / CLIP_SECONDS), MAX_CHUNKS_PER_FILE) chunks, taken
# from the start — so chunk 0 always contains the note onset (attack).
#
# Set to 1: keep only the onset chunk, drop the later ones. At the previous cap of 4, the
# later chunks were ~5% of the set and were near-duplicate ATTACK-LESS sustains of ~196
# long/very-long notes — and disproportionately the soft (piano) notes that are the most
# timbrally ambiguous (a soft sustained trumpet reads as a clarinet). Over-weighting the
# hardest, most redundant material ~3-4x is low-value at best. cap=1 gives every file one
# clip that includes the attack; long files keep their first CLIP_SECONDS. (Train/test
# leakage was never the concern here — the pitch-grouped split keeps a note's chunks in one
# split — but a cap is still cleaner.)
MAX_CHUNKS_PER_FILE = 1

# --- spectrogram ---
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = SR // 2
# Time axis is now FIXED: every clip is CLIP_SAMPLES @ hop 512, center=True -> 1 + n//512.
MAX_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH  # 130 at 3.0s
MIN_FRAMES = 8  # floor guard: 3 MaxPool2d(2) stages must leave a non-empty time axis (8->4->2->1)

# --- split ---
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --- training ---
# SEEDS drives the multi-seed runs; SEED is the single canonical seed used for data prep
# (so the split is identical across seeds — only model init and batch order vary, which is
# what we want to measure). Single-seed numbers were misleading: the learning-rate probe
# showed non-monotonic behaviour at intermediate settings that was pure seed noise.
# Five seeds, not three. Three is the bare minimum for a std at all, and the estimate it gives is
# itself very noisy; five is the common reporting convention and materially tightens it.
#
# APPEND-ONLY. SEED below is SEEDS[0] and is the seed prep_data uses to build the pitch-group
# split, so reordering this tuple would silently change the split and invalidate the cache, every
# checkpoint, and every recorded number. Adding seeds at the end is safe: the per-seed store in
# outputs/seed_metrics/ means existing seeds are not retrained.
SEEDS = (42, 43, 44, 45, 46)
SEED = SEEDS[0]
BATCH_SIZE = 32
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8
PLATEAU_PATIENCE = 4
PLATEAU_FACTOR = 0.5
DROPOUT = 0.4

# Regularisation / augmentation. Both default to the committed baseline (0.9234): weight
# decay 0 makes AdamW identical to Adam, and SpecAugment off means no masking. The model is
# generalisation-limited (train ~0.99 vs val ~0.92), so these two currently-zero levers are
# the first things to try within a plain CNN. Tune on VAL; a gain must clear the ~0.02 seed
# noise to be real.
WEIGHT_DECAY = 0.0           # AdamW L2 penalty
SPECAUGMENT = False          # time/frequency masking on TRAINING batches only
SPECAUG_FREQ_MASKS = 2       # masked frequency bands per clip
SPECAUG_FREQ_WIDTH = 15      # max width of each (mel bins, of 128)
SPECAUG_TIME_MASKS = 2       # masked time bands per clip
SPECAUG_TIME_WIDTH = 12      # max width of each (frames); capped at T//2 for short clips

# --- noise sweep ---
# "clean" is reported separately, not as an x value.
# 20/10/0 are the levels the pilot spec asked for. 60-30 were added after the sweep showed
# a clean-trained model already pinned to the majority class by 15dB: the entire 20/10/0
# band sits in a dead zone, and every interesting transition happens between 60 and 25dB.
# Keeping the full range makes the knee visible instead of plotting three identical points.
# Weighted toward the HIGH-SNR (minimal-noise) end, because that is where instrument ID
# breaks: the 2-class pilot's knee was 40-50dB — inaudible noise — and 12 confusable classes
# fail even earlier.
#
# Re-gridded 2026-07-28 for MODEL DISCRIMINATION rather than curve description. The measured
# 3-seed sweep showed three regimes (FINDINGS §5a): 60-30dB the model still uses all 12 classes
# (top-class share 10.7-21.5% against a uniform 8.3%); at 20dB it has shed 5 classes; by 10dB
# it predicts one class 46.8% of the time and by 0dB 75.4%. A collapsed model cannot rank
# models — every architecture scores near chance — so the old grid spent 3 of its 9 points
# where no comparison is possible and only 4 in the band that resolves anything.
#
# 2.5dB steps from 50 to 25 put 11 points across the steep region (balanced accuracy falls
# 0.90 -> ~0.50 there). 60/55 anchor the near-clean end. 20/10/0 are RETAINED to document the
# collapse, not to resolve it.
#
# CAVEAT on comparing to the previous sweep: noise_eval seeds its per-clip RNG on the condition's
# INDEX in this tuple (default_rng([NOISE_SEED, cond_idx, clip_idx])), so inserting levels shifts
# the noise realisation of every point after the first. Numbers at a shared SNR will therefore
# differ slightly from the old grid — same target SNR, different draw, averaged over 1284 clips x
# 3 seeds. Statistically comparable, NOT bit-identical. Seeding on the SNR value rather than the
# position would decouple the grid from the realisation and is worth doing before the six-model
# comparison, so that grid edits stop perturbing every other point.
SNR_LEVELS_DB = (60, 55, 50, 47.5, 45, 42.5, 40, 37.5, 35, 32.5, 30, 27.5, 25, 20, 10, 0)
NOISE_SEED = 1234

# Noise colours as 1/f**exponent power spectra: 0 = white (flat), 1 = pink (-3dB/oct,
# equal power per octave, ~natural ambient noise), 2 = brown (-6dB/oct, mostly rumble).
NOISE_COLORS = {"white": 0.0, "pink": 1.0, "brown": 2.0}

# The band where the music actually lives. SNR set over total power is misleading for
# coloured noise — brown noise at a nominal 0dB is ~+20dB *in this band* because almost all
# its energy sits below it. Reporting in-band SNR makes the colours comparable on an honest
# axis. (200Hz-8kHz spans the fundamentals and the harmonics that carry timbre.)
IN_BAND_HZ = (200, 8000)

# --- multi-label mixtures (multilabel.py) ---
# Path A toward polyphony: sum k isolated notes into one clip; the label is the SET of
# instruments present (a 12-dim multi-hot vector), trained with per-class sigmoid + BCE.
# Sources are drawn PER SPLIT from the pitch-grouped splits.json, so the train/test leak
# guarantee carries over — no source note appears in both.
MIX_POLYPHONY = (1, 2, 3)        # instruments per mixture, drawn uniformly; 1 keeps solo detection
MIX_COUNTS = {"train": 3600, "val": 800, "test": 800}
MIX_SEED = 2024
# CAVEAT: summed studio notes are NOT real polyphony — no reverb interaction, aligned
# onsets, uncorrelated parts. This validates the multi-label machinery and lets mixing be
# studied cleanly; it is not a substitute for IRMAS (configs/data/irmas.yaml).


# --- provenance ---

def config_fingerprint():
    """The data-processing settings that define what a cached array or checkpoint MEANS.

    Every artifact (manifest, checkpoint, metrics) records this; every consumer asserts it
    matches before using them together. This exists because a stale checkpoint evaluated
    against a freshly rebuilt cache produces a plausible-looking, entirely meaningless number
    — and nothing else in the pipeline would notice. Caught once by reading file timestamps
    by hand, which is exactly the vigilance this replaces.

    Only fields that change the meaning of the data belong here. Training hyperparameters
    (LR, epochs, dropout) deliberately do NOT: a model trained at a different learning rate is
    still a valid model for this cache, so including them would fire on harmless differences
    and train everyone to ignore the check.

    Postcondition: returns a JSON-serialisable dict, stable across runs for a given config.
    """
    return {
        "sr": SR,
        "clip_seconds": CLIP_SECONDS,
        "max_chunks_per_file": MAX_CHUNKS_PER_FILE,
        "trim_top_db": TRIM_TOP_DB,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "classes": list(CLASSES),
    }


class StaleArtifactError(RuntimeError):
    """An artifact was built under a different config than the one now in effect."""


def assert_fingerprint(found, source, expected=None):
    """Crash unless `found` matches the current config fingerprint.

    Precondition: `found` is a fingerprint dict previously produced by config_fingerprint(),
    or None for an artifact predating fingerprinting.
    Postcondition: returns None; raises otherwise. Never returns "close enough".

    Raises StaleArtifactError, naming the differing fields and how to fix it — a bare
    assertion here would print a 40-line dict diff and teach the reader nothing.
    """
    expected = expected if expected is not None else config_fingerprint()
    if found is None:
        raise StaleArtifactError(
            f"{source} predates config fingerprinting, so it cannot be checked against the "
            f"current config. Rebuild it: python -m instrument_robustness.prep_data")
    diffs = [f"    {k}: artifact={found.get(k, '<missing>')!r} current={v!r}"
             for k, v in expected.items() if found.get(k) != v]
    if diffs:
        raise StaleArtifactError(
            f"{source} was built under a different config:\n" + "\n".join(diffs) +
            "\n  Rebuild the cache and retrain, or check out the config that made it.")
