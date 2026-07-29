"""Shared config for the instrument-robustness pipeline (12-class Philharmonia).

This module is the SINGLE SOURCE OF TRUTH. Change constants here, never inline in a step script.

Code lives in the `instrument_robustness` package; DATA lives separately under the data root
(default: <repo>/all-samples). The two are decoupled so the package can be installed/imported from
anywhere while still finding the audio + artifacts. Override the data location with the
RISE_DATA_ROOT environment variable (see .env.example).

The audio itself is fetched by `python -m instrument_robustness.prep_data`, which is the only
supported way to obtain the dataset -- it writes `manifest.csv`, the index every step reads.
"""
import hashlib
import json
import os
from pathlib import Path

# config.py is at <repo>/src/instrument_robustness/config.py  ->  parents[2] == <repo>
_REPO = Path(__file__).resolve().parents[2]
REPO_ROOT = _REPO
ARTIFACTS = REPO_ROOT / "artifacts"
DATA_ROOT = Path(os.environ.get("RISE_DATA_ROOT", _REPO / "all-samples")).resolve()

ROOT = DATA_ROOT                       # kept for back-compat: step scripts resolve paths against ROOT
PIPE = DATA_ROOT / "pipeline"          # pipeline ARTIFACTS: manifest_*.csv, splits/windows.csv, stats, report
WORK = DATA_ROOT / "work"
RESAMPLED = WORK / "resampled"
TRIMMED = WORK / "trimmed"
WINDOWS = WORK / "windows"
FEATURES = DATA_ROOT / "features"

# 12 classes: 4 strings, 4 woodwinds, 4 brass. Kept ALPHABETICAL -- this ordering fixes the label
# indices, so reordering silently invalidates every saved checkpoint.
#
# This replaces the previous 9-class set. Oboe, double-bass and french-horn were absent from the
# older local Philharmonia copy; prep_data.py fetches all 12 from the Internet Archive mirror, so
# they are no longer missing. Any checkpoint trained under the 9-class set is INVALID here --
# label indices have shifted and must be retrained.
TARGET_LABELS = [
    "bassoon", "cello", "clarinet", "double-bass", "flute", "french-horn",
    "oboe", "trombone", "trumpet", "tuba", "viola", "violin",
]

INSTRUMENT_FAMILY = {
    "bassoon": "woodwinds",
    "cello": "strings",
    "clarinet": "woodwinds",
    "double-bass": "strings",
    "flute": "woodwinds",
    "french-horn": "brass",
    "oboe": "woodwinds",
    "trombone": "brass",
    "trumpet": "brass",
    "tuba": "brass",
    "viola": "strings",
    "violin": "strings",
}

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

# How many windows a single source file may contribute. 1 = crop to the first WINDOW_S seconds.
#
# Set to 1 for three reasons, in order of importance:
#
#  1. ONLY THE FIRST WINDOW STARTS AT A NOTE ONSET. Step 2 trims to the attack, so window 0 begins
#     at the note's start. Window 1 begins at exactly WINDOW_S, wherever that falls -- mid-sustain
#     for a held note, mid-note for a phrase. Attack transient is a dominant instrument cue, so
#     later windows are a structurally different kind of example: same label, no attack.
#  2. THE MIX WAS CLASS-CORRELATED. Windowing whole files produced +127 windows for clarinet and
#     +119 for flute against +10 for cello and +13 for violin -- winds and brass have the longer
#     recordings, so they supplied nearly all the attack-less examples.
#  3. IT OVER-WEIGHTED A FEW RECORDINGS. 2.6% of files produced 10.6% of windows, one file
#     yielding 26. Those 26 are one performance counted 26 times: in training that recording gets
#     26x the weight of a single note, and in test it inflates the effective sample size, since
#     per-window scoring treats correlated windows as independent.
#
# Cost of cropping, stated honestly: 749 windows (8.9%) discarded, and class imbalance goes from
# 1.73:1 back to 1.97:1 -- the extra windows happened to favour the smaller classes.
#
# Raise this to keep more of each recording; the trailing-window rule above still applies, so a
# 3.01 s file yields one window rather than a 0.01 s fragment either way.
MAX_WINDOWS_PER_SOURCE = 1

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
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"

# manifest.csv has two legitimate producers: prep_data (Philharmonia, fetched from the
# Internet Archive) and build_tinysol_manifest (TinySOL, indexed from local WAVs). Both emit
# the same schema and feed the identical downstream steps, so every stage that verifies the
# manifest's provenance accepts either. Kept here so those call sites cannot drift apart.
MANIFEST_PRODUCER_STAGES = ("prep_data", "build_tinysol_manifest")

MANIFEST_IN = DATA_ROOT / "manifest.csv"          # written by prep_data.py -- the canonical index
MANIFEST_FINGERPRINT = DATA_ROOT / "manifest_fingerprint.json"
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
        "fingerprint_version": 2,
        "archive_base": ARCHIVE_BASE,
        "sr": SR,
        "waveform_subtype": "PCM_16",
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "short_window_policy": "tile",
        "trim_top_db": TRIM_TOP_DB,
        "min_trim_s": MIN_TRIM_S,
        "split_group_fields": ["label", "note"],
        "split_policy": "label_note_pitch_group",
        "split_fracs": list(SPLIT_FRACS),
        "split_seed": SEED,
        "min_window_content_s": MIN_WINDOW_CONTENT_S,
        # Reshapes the dataset (9116 windows unlimited vs 8378 cropped), so it MUST be here.
        # Without it, a pre-crop feature array and a post-crop one produce identical fingerprints
        # and the stale one loads clean -- the same hole the articulation policy had.
        "max_windows_per_source": MAX_WINDOWS_PER_SOURCE,
        "target_rms": TARGET_RMS,
        # Which articulations step0 keeps. Without this field an artifact built before the
        # articulation filter existed (10196 rows, all techniques, 3.10:1 imbalance) produces a
        # fingerprint IDENTICAL to one built after it (8378 rows, one technique per class,
        # 1.97:1) -- so a stale feature array would pass assert_fingerprint and train silently on
        # the technique shortcut. Sorted for a stable comparison across runs.
        "articulations": {k: sorted(v) for k, v in sorted(STRICT_ARTICULATIONS.items())},
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP,
        "n_frames": N_FRAMES,
        "n_mfcc": N_MFCC,
        "svm_standardization": "per_feature_train",
        "logmel_standardization": "per_mel_bin_train",
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
    if not isinstance(found, dict):
        raise StaleArtifactError(
            f"{source} contains an invalid config fingerprint; expected a JSON object."
        )
    diffs = [f"    {k}: artifact={found.get(k, '<missing>')!r} current={v!r}"
             for k, v in expected.items() if found.get(k) != v]
    if diffs:
        raise StaleArtifactError(
            f"{source} was built under a different config:\n" + "\n".join(diffs) +
            "\n  Rebuild the pipeline, or check out the config that produced it.")


def config_fingerprint_json(fingerprint=None):
    """Return a deterministic JSON representation suitable for NPZ metadata."""
    value = config_fingerprint() if fingerprint is None else fingerprint
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def assert_serialized_fingerprint(value, source):
    """Decode fingerprint JSON from an NPZ scalar and verify it against current config."""
    if value is None:
        found = None
    else:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            found = json.loads(value) if isinstance(value, str) else value
        except (TypeError, json.JSONDecodeError) as error:
            raise StaleArtifactError(
                f"{source} contains an unreadable config fingerprint"
            ) from error
    assert_fingerprint(found, source)
    return found


def artifact_fingerprint_path(artifact_path):
    """Return the default sidecar path for a generated CSV artifact."""
    path = Path(artifact_path)
    return path.with_name(f"{path.name}.fingerprint.json")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_fingerprint(
    artifact_path,
    stage,
    *,
    fingerprint_path=None,
    metadata=None,
):
    """Write provenance for a generated artifact without inflating its tabular schema."""
    sidecar = (
        Path(fingerprint_path)
        if fingerprint_path is not None
        else artifact_fingerprint_path(artifact_path)
    )
    payload = {
        "artifact": str(Path(artifact_path).name),
        "sha256": _sha256(artifact_path),
        "stage": stage,
        "fingerprint": config_fingerprint(),
    }
    if metadata:
        payload["metadata"] = metadata
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar


def assert_artifact_fingerprint(
    artifact_path,
    expected_stage,
    *,
    fingerprint_path=None,
):
    """Verify both the config and producing stage recorded beside an artifact."""
    sidecar = (
        Path(fingerprint_path)
        if fingerprint_path is not None
        else artifact_fingerprint_path(artifact_path)
    )
    if not sidecar.exists():
        raise StaleArtifactError(
            f"{artifact_path} has no provenance sidecar at {sidecar}. Rebuild the pipeline."
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StaleArtifactError(
            f"{artifact_path} has an unreadable provenance sidecar at {sidecar}"
        ) from error
    if not isinstance(payload, dict):
        raise StaleArtifactError(
            f"{artifact_path} has an invalid provenance sidecar at {sidecar}"
        )
    recorded_hash = payload.get("sha256")
    current_hash = _sha256(artifact_path)
    if recorded_hash != current_hash:
        raise StaleArtifactError(
            f"{artifact_path} does not match its provenance sidecar at {sidecar}. "
            "Rebuild the pipeline stage."
        )
    assert_fingerprint(payload.get("fingerprint"), str(artifact_path))
    # `expected_stage` may name one producer or several. An artifact can legitimately have more
    # than one: manifest.csv comes from prep_data for Philharmonia and from
    # build_tinysol_manifest for TinySOL, and both feed the identical downstream steps. Accepting
    # a collection keeps the check strict -- an unrecognised stage still fails -- without forcing
    # a second dataset to misreport which stage built it.
    allowed = (
        {expected_stage}
        if isinstance(expected_stage, str)
        else set(expected_stage)
    )
    if payload.get("stage") not in allowed:
        expected_description = (
            repr(expected_stage)
            if isinstance(expected_stage, str)
            else " or ".join(repr(stage) for stage in sorted(allowed))
        )
        raise StaleArtifactError(
            f"{artifact_path} was produced by stage {payload.get('stage')!r}; "
            f"expected {expected_description}. Run the missing pipeline step."
        )
    return payload["fingerprint"]
