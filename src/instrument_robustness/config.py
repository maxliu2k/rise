"""Shared config for the instrument-robustness pipeline.

Code now lives in the `instrument_robustness` package; DATA lives separately under the data root
(default: <repo>/all-samples). The two are decoupled so the package can be installed/imported from
anywhere while still finding the audio + artifacts. Override the data location with the
RISE_DATA_ROOT environment variable (see .env.example).
"""
import os
from pathlib import Path

# config.py is at <repo>/src/instrument_robustness/config.py  ->  parents[2] == <repo>
_REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RISE_DATA_ROOT", _REPO / "all-samples")).resolve()

ROOT = DATA_ROOT                       # kept for back-compat: step scripts resolve paths against ROOT
PIPE = DATA_ROOT / "pipeline"          # pipeline ARTIFACTS: manifest_9*.csv, splits/windows.csv, stats, report
WORK = DATA_ROOT / "work"
RESAMPLED = WORK / "resampled"
TRIMMED = WORK / "trimmed"
WINDOWS = WORK / "windows"
FEATURES = DATA_ROOT / "features"

PHILHARMONIA_LABELS = [
    "violin", "viola", "cello",          # strings
    "flute", "clarinet", "bassoon",      # woodwinds
    "trumpet", "tuba", "trombone",       # brass
]

TINYSOL_LABELS = [
    "violin", "viola", "cello", "double bass",
    "flute", "clarinet", "oboe", "bassoon",
    "trumpet", "french horn", "trombone", "tuba",
]

INSTRUMENT_LABEL_ALIASES = {
    "double-bass": "double bass",
    "french-horn": "french horn",
}

INSTRUMENT_FAMILY = {
    "violin": "strings",
    "viola": "strings",
    "cello": "strings",
    "double bass": "strings",
    "flute": "woodwinds",
    "clarinet": "woodwinds",
    "oboe": "woodwinds",
    "bassoon": "woodwinds",
    "trumpet": "brass",
    "french horn": "brass",
    "trombone": "brass",
    "tuba": "brass",
}

# Keep the original nine AST label IDs stable and append newly supported classes.
AST_LABEL_ORDER = PHILHARMONIA_LABELS + [
    label for label in TINYSOL_LABELS if label not in PHILHARMONIA_LABELS
]


def normalize_instrument_label(label: str) -> str:
    normalized = str(label).strip().lower()
    return INSTRUMENT_LABEL_ALIASES.get(normalized, normalized)


def _target_labels_from_environment():
    value = os.environ.get("RISE_TARGET_LABELS")
    if not value:
        return list(PHILHARMONIA_LABELS)

    labels = [label.strip().lower() for label in value.split(",") if label.strip()]
    if not labels:
        raise ValueError("RISE_TARGET_LABELS must contain at least one label")
    if len(labels) != len(set(labels)):
        raise ValueError("RISE_TARGET_LABELS contains duplicate labels")
    return labels


# Preprocessing defaults to the original Philharmonia nine classes. Set the comma-separated
# RISE_TARGET_LABELS when rebuilding another dataset. AST training additionally discovers and
# validates its classes directly from pipeline/windows.csv.
TARGET_LABELS = _target_labels_from_environment()

SR = 22050            # common resample rate; Nyquist 11025 Hz sits below the lowest MP3 brick wall (~16 kHz)
TRIM_TOP_DB = 30      # silence-trim threshold
MIN_TRIM_S = 0.10     # if trimming leaves less than this, keep the untrimmed (resampled) audio and flag

# --- Step 3: split ---
SPLIT_FRACS = (0.70, 0.15, 0.15)   # train / val / test, stratified by label, split BY SOURCE FILE
SEED = 0

# --- Step 4: window ---
WINDOW_S = 3.0        # fixed window length (matches IEEE baseline)
HOP_S = 3.0           # NO overlap (hop == window). Chosen to avoid amplifying phrase-window imbalance
                      # and to avoid near-duplicate correlated windows. Short/only windows are zero-padded.
MIN_WINDOW_CONTENT_S = 0.5   # drop a trailing window with less real content than this, UNLESS it is a
                             # source's only window (every source must contribute >= 1 window)

# --- Step 5: loudness normalize ---
TARGET_RMS = 0.1      # per-window RMS target; peak-guarded to avoid clipping

# --- Steps 6-7: featurization ---
STATS_NPZ = PIPE / "norm_stats.npz"
STATS_JSON = PIPE / "norm_stats.json"

# log-mel params (CNN/CRNN). All windows are exactly 3.0 s (66150 samples) -> exactly 130 frames.
N_FFT = 2048
HOP = 512
N_MELS = 128
N_FRAMES = 130
N_MFCC = 20           # SVM MFCC coefficients

# AST / MERT / PANNs carry their own extractors + sample rates (NOT the 22050 set, NOT Step-6 stats)
AST_SR = 16000
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MERT_SR = 24000
MERT_MODEL = "m-a-p/MERT-v1-95M"

MANIFEST_IN = DATA_ROOT / "manifest.csv"
MANIFEST_9 = PIPE / "manifest_9.csv"
REPORT = PIPE / "pipeline_report.txt"
