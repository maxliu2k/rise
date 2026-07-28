"""Shared config for the instrument-robustness pipeline (12-class Philharmonia).

This module is the SINGLE SOURCE OF TRUTH. Change constants here, never inline in a step script.

Code lives in the `instrument_robustness` package; DATA lives separately under the data root
(default: <repo>/all-samples). The two are decoupled so the package can be installed/imported from
anywhere while still finding the audio + artifacts. Override the data location with the
RISE_DATA_ROOT environment variable (see .env.example).

The audio itself is fetched by `python -m instrument_robustness.prep_data`, which is the only
supported way to obtain the dataset -- it writes `manifest.csv`, the index every step reads.
"""
import os
from pathlib import Path

# config.py is at <repo>/src/instrument_robustness/config.py  ->  parents[2] == <repo>
_REPO = Path(__file__).resolve().parents[2]

# --- Canonical class list: 12 classes, 4 strings / 4 woodwinds / 4 brass. ALPHABETICAL --
# this ordering fixes the label indices, so reordering silently invalidates every saved checkpoint.
#
# This replaces the previous 9-class set. Oboe, double-bass and french-horn were absent from the
# older local Philharmonia copy; prep_data.py fetches all 12 from the Internet Archive mirror, so
# they are no longer missing. Any checkpoint trained under the 9-class set is INVALID here --
# label indices have shifted and must be retrained.
#
# BOTH datasets use these same 12 labels, so cross-dataset evaluation is index-compatible with no
# remapping. TinySOL folder names map onto them in build_tinysol_manifest.py (Violoncello->cello,
# Contrabass->double-bass, Horn->french-horn, Bass_Tuba->tuba, ...). Note the HYPHENS:
# "double-bass" / "french-horn" match the Philharmonia archive's filename field.
CANONICAL_LABELS = [
    "bassoon", "cello", "clarinet", "double-bass", "flute", "french-horn",
    "oboe", "trombone", "trumpet", "tuba", "viola", "violin",
]
# Kept as names for readability at call sites; both now refer to the SAME 12 labels above.
PHILHARMONIA_LABELS = CANONICAL_LABELS
TINYSOL_LABELS = CANONICAL_LABELS

# --- Per-dataset data roots. Each mirrors the DATA_ROOT layout (pipeline/ tracked in git;
#     work/, features/, checkpoints/ are git-ignored). Overridable via env; never hardcode paths. ---
PHILHARMONIA_ROOT = Path(os.environ.get("RISE_PHIL_ROOT", _REPO / "all-samples")).resolve()
TINYSOL_ROOT = Path(os.environ.get("RISE_TINYSOL_ROOT", _REPO / "tinysol")).resolve()

# --- Active data root for the pipeline steps. Set RISE_DATA_ROOT to select the dataset;
#     defaults to Philharmonia. For TinySOL: RISE_DATA_ROOT=$RISE_TINYSOL_ROOT (see .env.example). ---
DATA_ROOT = Path(os.environ.get("RISE_DATA_ROOT", PHILHARMONIA_ROOT)).resolve()

ROOT = DATA_ROOT                       # kept for back-compat: step scripts resolve paths against ROOT
PIPE = DATA_ROOT / "pipeline"          # pipeline ARTIFACTS: manifest_*.csv, splits/windows.csv, stats, report
WORK = DATA_ROOT / "work"
RESAMPLED = WORK / "resampled"
TRIMMED = WORK / "trimmed"
WINDOWS = WORK / "windows"
FEATURES = DATA_ROOT / "features"

_DEFAULT_LABELS = CANONICAL_LABELS
TARGET_LABELS = ([s.strip() for s in os.environ["RISE_TARGET_LABELS"].split(",") if s.strip()]
                 if os.environ.get("RISE_TARGET_LABELS") else _DEFAULT_LABELS)

# --- Acquisition (prep_data.py) ---
# The official philharmonia.co.uk/assets/... URLs predate their site redesign and no longer
# resolve. The Internet Archive mirror is the working source. CC-BY-SA 4.0.
ARCHIVE_BASE = "https://archive.org/download/philharmonicorchestrasamples"
DATA_RAW = DATA_ROOT / "raw"

# The archive's zip/dir names are NOT the instrument field: zips use spaces where filenames use
# hyphens, and `cor anglais.zip` contains `english-horn_*.mp3`.
ZIP_NAME = {"double-bass": "double bass", "french-horn": "french horn"}

# One articulation per class, so the model cannot separate classes on playing technique.
# `normal`/`arco-normal` dominates (84-89% of files), so this costs almost nothing.
STRICT_ARTICULATIONS = {
    "bassoon": {"normal"}, "cello": {"arco-normal"}, "clarinet": {"normal"},
    "double-bass": {"arco-normal"}, "flute": {"normal"}, "french-horn": {"normal"},
    "oboe": {"normal"}, "trombone": {"normal"}, "trumpet": {"normal"},
    "tuba": {"normal"}, "viola": {"arco-normal"}, "violin": {"arco-normal"},
}
MIN_STRICT_N = 200      # per-class floor before falling back to the sustained family
MAX_IMBALANCE = 1.5     # above this ratio, apply class weights

SR = 22050            # common resample rate; Nyquist 11025 Hz sits below the lowest MP3 brick wall (~16 kHz)
TRIM_TOP_DB = 30      # silence-trim threshold
MIN_TRIM_S = 0.10     # if trimming leaves less than this, keep the untrimmed (resampled) audio and flag

# --- Step 3: split ---
SPLIT_FRACS = (0.70, 0.15, 0.15)   # train / val / test, stratified by label, split BY PITCH GROUP
                                   # (all files sharing label+note move together -- see step3)
SEED = 0

# --- Step 4: window ---
WINDOW_S = 3.0        # fixed window length (matches IEEE baseline)
HOP_S = 3.0           # NO overlap (hop == window). Chosen to avoid amplifying phrase-window imbalance
                      # and to avoid near-duplicate correlated windows.
                      # Short/only windows are TILED (looped) to fill the window -- never zero-padded.
                      # Zero-padding is not a style choice: power_to_db(ref=np.max) clamps digital
                      # silence to the -80 dB floor, injected noise fills it, and the clip lands
                      # outside the training distribution. Measured effect is majority-class collapse
                      # at EVERY SNR, i.e. it destroys any noise-robustness result. Tiling keeps every
                      # sample real signal. See step4_window.py and FINDINGS S6 on cnn-ensemble.
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

MANIFEST_IN = DATA_ROOT / "manifest.csv"          # written by prep_data.py -- the canonical index
REPORT = PIPE / "pipeline_report.txt"

# Intermediate manifests. Previously these were named manifest_9*.csv and steps 1-3 hard-coded the
# filenames as string literals, so config did not actually own the paths it claimed to. Both are
# fixed here: the names no longer assert a class count that can drift, and every step reads its
# input path from this module.
MANIFEST_LABELED = PIPE / "manifest_labeled.csv"          # step 0 out
MANIFEST_RESAMPLED = PIPE / "manifest_resampled.csv"      # step 1 out
MANIFEST_TRIMMED = PIPE / "manifest_trimmed.csv"          # step 2 out
SPLITS_CSV = PIPE / "splits.csv"                          # step 3 out
WINDOWS_CSV = PIPE / "windows.csv"                        # step 4 out


# --------------------------------------------------------------------------- provenance

def config_fingerprint():
    """The settings that define what a cached array, manifest, or checkpoint MEANS.

    Postcondition: returns a JSON-serialisable dict. Only fields that change the meaning of the
    data belong here. Training hyperparameters (learning rate, epochs, dropout) deliberately do
    NOT -- two models trained at different learning rates on the same data are comparable; two
    trained on different CLASS SETS or window lengths are not.
    """
    return {
        "sr": SR,
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "trim_top_db": TRIM_TOP_DB,
        "min_window_content_s": MIN_WINDOW_CONTENT_S,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "n_frames": N_FRAMES,
        "labels": list(TARGET_LABELS),
    }


class StaleArtifactError(RuntimeError):
    """An artifact was built under a different config than the one now in effect."""


def assert_fingerprint(found, source, expected=None):
    """Crash unless `found` matches the current config fingerprint.

    Preconditions: `found` is the fingerprint dict read back from an artifact, or None if the
    artifact predates fingerprinting. `source` names the artifact, for the error message.
    Postcondition: returns None if the artifact is consistent with the current config.
    Raises: StaleArtifactError, naming every field that differs.

    This exists because the failure it guards is silent. A checkpoint or feature array built under
    a different label set or window length still loads, still runs, and still produces plausible
    numbers -- there is no crash and no warning, only a wrong result that gets believed. Never rely
    on noticing a file timestamp.
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
            "\n  Rebuild the pipeline, or check out the config that produced it.")
