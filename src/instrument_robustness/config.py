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
# invalidate every saved checkpoint.
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
CLIP_SECONDS = 2.0            # MAXIMUM clip length, not a fixed one — see below
CLIP_SAMPLES = int(SR * CLIP_SECONDS)  # 44100
TRIM_TOP_DB = 30

# Clips are VARIABLE LENGTH. Nothing is ever padded or tiled: a note shorter than
# CLIP_SECONDS is kept at its true length, and a file longer than CLIP_SECONDS is cut into
# chunks of exactly CLIP_SECONDS (capped, see below). Every sample the model sees is real
# recorded audio.
#
# This is viable because MediumCNN ends in AdaptiveAvgPool2d, which collapses any time
# axis — verified from 12 to 500 frames. Batching is what needs care: train.py groups
# clips by exact frame count so every batch is uniform without padding (73 groups here,
# median 10 clips, mean batch ~14 vs the 32 target).
#
# Rejected alternatives, both measured:
#   tiling  — repeat a short note to fill CLIP_SECONDS. Produces no click artifact (seam
#             discontinuity is 0.2x a normal sample step, since trimmed notes start and end
#             near zero) but fabricates 2s of audio from as little as 0.26s. 79% of clips
#             would have been tiled.
#   zeros   — centered zero-padding. Actively breaks the noise sweep: power_to_db clamps
#             digital silence to the -80dB floor, that floor is ~61% of a median image, and
#             added noise fills it, collapsing the spectrogram's std and pushing the clip
#             outside the training distribution. Measured: majority-class collapse (0.65
#             acc, trumpet F1 0.00) at every SNR including a mild 20dB.
#
# Known cost: clip length now varies 11-87 frames, so a short clip carries less evidence
# than a long one at the same SNR. noise_eval.py reports the sweep per length bucket to
# keep that from being confounded with the noise effect itself.

# A file of duration d yields min(floor(d / CLIP_SECONDS), MAX_CHUNKS_PER_FILE) chunks.
# The cap exists because chunks of one sustained note are near-identical: uncapped, a
# single 70.66s trumpet recording yields 35 chunks and would be weighted 35x any other
# note. Pseudo-replication inflates the dataset without adding information. Train/test
# leakage is separately prevented by the pitch-grouped split — every chunk of a file shares
# its {instrument}_{note} group, so chunks cannot span splits.
MAX_CHUNKS_PER_FILE = 4

# --- spectrogram ---
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = SR // 2
# Time axis is variable. A clip of n samples @ hop 512, center=True -> 1 + n//512 frames,
# so 2.0s -> 87 (the maximum) and the shortest note here, 0.26s, -> 12.
MAX_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH  # 87
MIN_FRAMES = 8  # 3 MaxPool2d(2) stages must leave a non-empty time axis: 8 -> 4 -> 2 -> 1

# --- split ---
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --- training ---
# SEEDS drives the multi-seed runs; SEED is the single canonical seed used for data prep
# (so the split is identical across seeds — only model init and batch order vary, which is
# what we want to measure). Single-seed numbers were misleading: the learning-rate probe
# showed non-monotonic behaviour at intermediate settings that was pure seed noise.
SEEDS = (42, 43, 44)
SEED = SEEDS[0]
BATCH_SIZE = 32
MAX_EPOCHS = 40
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8
PLATEAU_PATIENCE = 4
PLATEAU_FACTOR = 0.5
DROPOUT = 0.4

# --- noise sweep ---
# "clean" is reported separately, not as an x value.
# 20/10/0 are the levels the pilot spec asked for. 60-30 were added after the sweep showed
# a clean-trained model already pinned to the majority class by 15dB: the entire 20/10/0
# band sits in a dead zone, and every interesting transition happens between 60 and 25dB.
# Keeping the full range makes the knee visible instead of plotting three identical points.
# Weighted toward the HIGH-SNR (minimal-noise) end, because that is where instrument ID
# breaks: the 2-class pilot's knee was 40-50dB — inaudible noise — and 12 confusable classes
# fail even earlier. Below ~20dB everything has long since collapsed, so the low levels are
# kept only to confirm the floor, not to resolve it.
SNR_LEVELS_DB = (60, 50, 45, 40, 35, 30, 20, 10, 0)
NOISE_SEED = 1234

# Noise colours as 1/f**exponent power spectra: 0 = white (flat), 1 = pink (-3dB/oct,
# equal power per octave, ~natural ambient noise), 2 = brown (-6dB/oct, mostly rumble).
NOISE_COLORS = {"white": 0.0, "pink": 1.0, "brown": 2.0}

# The band where the music actually lives. SNR set over total power is misleading for
# coloured noise — brown noise at a nominal 0dB is ~+20dB *in this band* because almost all
# its energy sits below it. Reporting in-band SNR makes the colours comparable on an honest
# axis. (200Hz-8kHz spans the fundamentals and the harmonics that carry timbre.)
IN_BAND_HZ = (200, 8000)
