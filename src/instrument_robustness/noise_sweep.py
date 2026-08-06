"""Generate the one shared noisy TEST set used by every model.

The clean train/validation data and all fitted models remain untouched. One deterministic noise
realization is drawn for each (dataset build, test window, noise type, replicate), then scaled to
every SNR. The manifest is written last and is the completion marker consumed by model evaluators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import warnings
from pathlib import Path
from typing import NamedTuple

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from instrument_robustness.config import (
    INSTRUMENT_BAND_HZ,
    MANIFEST_FINGERPRINT,
    MANIFEST_IN,
    MANIFEST_PRODUCER_STAGES,
    NOISE_ACTIVE_TOP_DB,
    NOISE_TYPES,
    N_REPLICATES,
    ROOT,
    SEGMENTAL_FRAME,
    SEGMENTAL_HOP,
    SIGNAL_ACTIVE_TOP_DB,
    SNRS,
    SR,
    TARGET_RMS,
    WINDOWS_CSV,
    WORK,
    artifact_fingerprint_path,
    assert_artifact_fingerprint,
    config_fingerprint,
)
from instrument_robustness.noise_metrics import (
    DIAGNOSTIC_COLUMNS,
    OCTAVE_CENTERS_HZ,
    mixture_diagnostics,
)

warnings.filterwarnings("ignore")

NOISY_DIR = WORK / "windows_noisy"
NOISE_MANIFEST_NAME = "noise_manifest.json"
NOISE_PROVENANCE_NAME = "noise_provenance.csv"
NOISE_MANIFEST_VERSION = 6
NOISE_ROOT = Path(
    os.environ.get("RISE_NOISE_ROOT", Path.home() / "Downloads/noise_sources")
)
ESC50_ROOT = NOISE_ROOT / "ESC-50-master"
ESC50_DIR = ESC50_ROOT / "audio"
ESC50_META = ESC50_ROOT / "meta" / "esc50.csv"
CLIP_LEN = int(round(3.0 * SR))
MAX_SNR_ERROR_DB = 0.1
MIN_CENTERED_NOISE_RMS = 1e-6

# How far the cosine similarity between two SNR levels' added components may fall below 1 before
# validate() calls it a re-drawn realization.
#
# The property guarded: for one (window, noise_type) the SAME realization must be reused at every
# SNR, scaled only by alpha. Otherwise SNR is not the only thing varying along the curve.
#
# The bound is set from measurement, not taste. `noisy` is stored float32, so recovering the added
# component as `noisy - clean` costs a rounding step, and the deviation is never exactly zero.
# Measured on the sealed 97b1cdd2 build, 5 validate windows x 3 noise types x 8 SNRs:
#
#   same realization, rounding only     6e-8 .. 6.62e-6      <- what we must tolerate
#   genuinely different realization     0.992 .. 0.996       <- what we must catch
#
# The two populations are five orders of magnitude apart. 1e-3 sits ~150x above the worst
# rounding and ~1000x below the cheapest real failure. The previous bound was 1e-6 -- BELOW the
# rounding floor -- so on 2026-08-03 it failed the whole sweep on a mechanical draw measuring
# 6.62e-6, which was correct arithmetic. A threshold inside the noise it is measuring reports
# noise.
MAX_REALIZATION_COSINE_DEVIATION = 1e-3

# Reject a completed corpus if more than this share of a mixture's noise power is residual DC.
# Every realization is explicitly centered before scaling, so crossing 1% indicates that the
# generated files no longer implement the recorded protocol.
MAX_DC_POWER_SHARE = 0.01

# ESC-50 target blocks: 0-19 animals, 20-29 human non-speech, 30-49 domestic/urban.
#
# `audience` (20-29) is the block this study actually cares about: breathing, brushing_teeth,
# clapping, coughing, crying_baby, drinking_sipping, footsteps, laughing, sneezing, snoring.
# That is the sound of a room full of people around a performance -- the realistic corruption
# for a concert recording.
#
# `natural` and `mechanical` are kept because the 2026-08-03 sweep was generated with them and
# its results must stay reproducible from this file. They are NOT in config.NOISE_TYPES: birds
# and chainsaws are not what degrades a concert or studio capture, so they answer a question
# nobody asked.
ESC50_TARGETS = {
    "audience": range(20, 30),
    "natural": range(0, 20),
    "mechanical": range(30, 50),
}

# --- DEMAND ------------------------------------------------------------------------------------
# 18 environments, each a 16-microphone array recording ONE scene for 300 s at 48 kHz.
DEMAND_ROOT = Path(
    os.environ.get("RISE_DEMAND_ROOT", NOISE_ROOT / "DEMAND")
)

# ONE CHANNEL, NOT SIXTEEN. ch01..ch16 are sixteen microphones of the same array capturing the
# same acoustic event at the same instant. They are near-duplicates that differ by array
# geometry, not sixteen independent samples of the environment. Indexing all sixteen would
# multiply the apparent corpus by 16 while adding almost no acoustic diversity -- the same
# mistake as splitting this dataset's windows randomly instead of by pitch group, and it would
# inflate the studio condition's effective sample size in exactly the way that is hardest to
# notice afterwards.
DEMAND_CHANNEL = "ch01"

# Extra source frames read either side of a window so the resampler's filter has support and the
# output is not short by a sample or two. Cheap insurance; the segment is trimmed to CLIP_LEN.
RESAMPLE_MARGIN_FRAMES = 4096

# Grouped the way DEMAND itself does: domestic, nature, office, public, street, transportation.
# The grouping is recorded per mixture as provenance; it does not select anything.
DEMAND_ENVIRONMENTS: dict[str, str] = {
    "DKITCHEN": "domestic", "DLIVING": "domestic", "DWASHING": "domestic",
    "NFIELD": "nature", "NPARK": "nature", "NRIVER": "nature",
    "OHALLWAY": "office", "OMEETING": "office", "OOFFICE": "office",
    "PCAFETER": "public", "PRESTO": "public", "PSTATION": "public",
    "SCAFE": "street", "SPSQUARE": "street", "STRAFFIC": "street",
    "TBUS": "transportation", "TCAR": "transportation", "TMETRO": "transportation",
}

# Which project noise type each DEMAND environment feeds. Kept as a dict of the same shape as
# ESC50_TARGETS so a reader can see both corpus mappings in one place.
#
# `studio` is DEMAND room tone -- the continuous background of a real recording space. It is in
# config.NOISE_TYPES as of 2026-08-03, after snr_pilot was run on it and on `audience` with both
# SVM and MERT; the grid was regridded to [50..-10] on that evidence. Anything added here in
# future needs the same treatment before it is enabled, because an unpiloted grid is how this
# project once ended up with one sitting entirely at or below chance.
DEMAND_TARGETS = {"studio": tuple(DEMAND_ENVIRONMENTS)}


def diagnostic_protocol() -> dict[str, object]:
    """Settings that define the SNR diagnostics stored with every mixture."""
    return {
        "instrument_band_hz": list(INSTRUMENT_BAND_HZ),
        "octave_centers_hz": list(OCTAVE_CENTERS_HZ),
        "segmental_frame": SEGMENTAL_FRAME,
        "segmental_hop": SEGMENTAL_HOP,
        "noise_active_top_db": NOISE_ACTIVE_TOP_DB,
        "signal_active_top_db": SIGNAL_ACTIVE_TOP_DB,
        "effective_snr_rates_hz": {"ast": 16000, "mert": 24000, "panns": 32000},
    }


def noise_preprocessing_protocol() -> dict[str, object]:
    """Transformations applied to every noise realization before SNR scaling."""
    return {
        "remove_segment_mean": True,
        "minimum_centered_rms": MIN_CENTERED_NOISE_RMS,
    }


def _assert_residual_dc_is_bounded(
    provenance: pd.DataFrame,
    *,
    provenance_name: str,
) -> None:
    """Reject noise whose residual DC violates the centered-noise protocol."""
    if "noise_dc_power_share" not in provenance.columns:
        return
    worst_dc = float(provenance["noise_dc_power_share"].astype(float).max())
    if worst_dc > MAX_DC_POWER_SHARE:
        raise ValueError(
            f"A mixture's noise is {100 * worst_dc:.3f}% residual DC; "
            f"maximum allowed is {100 * MAX_DC_POWER_SHARE:.3f}%. "
            f"Inspect noise_dc_offset in {provenance_name}."
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read JSON provenance at {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def dataset_build_identity(
    *,
    manifest_csv: str | Path = MANIFEST_IN,
    manifest_fingerprint: str | Path = MANIFEST_FINGERPRINT,
    windows_csv: str | Path = WINDOWS_CSV,
) -> dict[str, object]:
    """Identify the actual manifest/windows build, not only its configuration."""
    manifest_csv = Path(manifest_csv)
    manifest_fingerprint = Path(manifest_fingerprint)
    windows_csv = Path(windows_csv)
    assert_artifact_fingerprint(
        manifest_csv,
        MANIFEST_PRODUCER_STAGES,
        fingerprint_path=manifest_fingerprint,
    )
    assert_artifact_fingerprint(windows_csv, "step5_normalize")
    manifest_sidecar = _read_json(manifest_fingerprint)
    windows_sidecar = _read_json(artifact_fingerprint_path(windows_csv))
    identity = {
        "config_fingerprint": config_fingerprint(),
        "manifest_sha256": manifest_sidecar["sha256"],
        "windows_csv_sha256": windows_sidecar["sha256"],
    }
    audio_hash = (windows_sidecar.get("metadata") or {}).get("windows_audio_inventory_sha256")
    if audio_hash is not None:
        identity["windows_audio_inventory_sha256"] = audio_hash
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["dataset_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return identity


def dataset_fingerprint(identity: dict[str, object] | None = None) -> str:
    """Full digest of the actual canonical dataset build."""
    value = dataset_build_identity() if identity is None else identity
    return str(value["dataset_fingerprint"])


SEED_SCHEME = (
    "sha256(dataset_fingerprint|window_id|noise_type|replicate)[:4]; SNR excluded"
)


def window_seed(
    window_id: str,
    noise_type: str,
    fingerprint: str | None = None,
    replicate: int = 0,
) -> int:
    """Stable seed for one noise realization.

    SNR is intentionally ABSENT so a single realization is merely rescaled along the SNR curve --
    otherwise part of the drop between levels would be a different noise draw rather than a louder
    one. `replicate` IS present, because that is exactly the axis along which a different draw is
    wanted: replicate 1 must be an independent realization of the same condition, so that the spread
    across replicates can be separated from the difference between models.

    Preconditions: `replicate` is a non-negative integer.
    """
    # Gate on what draw_noise can produce, not on config.NOISE_TYPES. Seeding a type is a
    # prerequisite for PILOTING it, and a type must be piloted before it is configured -- gating
    # here on the configured set locked every new type out of its own pilot.
    if noise_type not in {"white", *ESC50_TARGETS, *DEMAND_TARGETS}:
        raise ValueError(f"Unknown noise type: {noise_type}")
    if int(replicate) != replicate or replicate < 0:
        raise ValueError(f"replicate must be a non-negative integer, got {replicate!r}")
    build = dataset_fingerprint() if fingerprint is None else fingerprint
    key = f"{build}|{window_id}|{noise_type}|{int(replicate)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


class Esc50Clip(NamedTuple):
    """One selected ESC-50 source clip and the corpus metadata describing it.

    `target`, `category` and `fold` are carried so per-mixture provenance can record WHICH of the
    20 ESC-50 classes inside a project category was actually drawn. Collapsing 20 classes into
    "natural" is a defensible grouping, but it is only auditable if the original label survives
    into the output -- and it cannot be recovered afterwards without regenerating the sweep.
    """

    path: Path
    target: int
    category: str
    fold: int


class DemandRecording(NamedTuple):
    """One DEMAND environment recording, and the metadata describing it.

    `environment` and `grouping` are carried for the same reason Esc50Clip carries `category`:
    collapsing 18 environments into "studio" is only auditable if the original environment
    survives into per-mixture provenance. `frames` is the source length, cached at index time so
    a draw can pick a crop offset without opening the file twice.
    """

    path: Path
    environment: str
    grouping: str
    channel: str
    frames: int
    samplerate: int


def load_demand_index(demand_root: Path | None = None) -> dict[str, list[DemandRecording]]:
    """Index one channel of every DEMAND environment.

    Preconditions: `demand_root` contains one directory per environment named in
    DEMAND_ENVIRONMENTS, each holding `<DEMAND_CHANNEL>.wav`.
    Postcondition: returns {noise_type: [DemandRecording, ...]} ordered by environment name, one
    entry per environment. The ordering is load-bearing exactly as it is for ESC-50 -- `draw_noise`
    indexes into this list with a seeded RNG, so reordering changes which environment a seed picks.
    Raises: FileNotFoundError if the root or any environment channel is missing; ValueError if a
    recording is too short to yield one window.
    """
    root = Path(demand_root) if demand_root is not None else DEMAND_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"DEMAND not found at {root}. Set RISE_DEMAND_ROOT, or drop `studio` from the "
            "noise types."
        )

    recordings: list[DemandRecording] = []
    for environment in sorted(DEMAND_ENVIRONMENTS):
        path = root / environment / f"{DEMAND_CHANNEL}.wav"
        if not path.is_file():
            raise FileNotFoundError(
                f"DEMAND environment {environment} has no {DEMAND_CHANNEL}.wav at {path}"
            )
        info = sf.info(str(path))
        if info.channels != 1:
            raise ValueError(f"{path} has {info.channels} channels; expected mono")
        # Enough source frames to yield CLIP_LEN after resampling to SR, plus the resampler margin.
        needed = _source_frames_for_window(info.samplerate)
        if info.frames < needed:
            raise ValueError(
                f"{path} has {info.frames} frames; need at least {needed} to cut one "
                f"{CLIP_LEN}-sample window at {SR} Hz"
            )
        recordings.append(
            DemandRecording(
                path=path,
                environment=environment,
                grouping=DEMAND_ENVIRONMENTS[environment],
                channel=DEMAND_CHANNEL,
                frames=int(info.frames),
                samplerate=int(info.samplerate),
            )
        )

    if len(recordings) != len(DEMAND_ENVIRONMENTS):
        raise ValueError(
            f"Indexed {len(recordings)} DEMAND environments; expected "
            f"{len(DEMAND_ENVIRONMENTS)}"
        )
    return {noise_type: list(recordings) for noise_type in DEMAND_TARGETS}


def _source_frames_for_window(source_sr: int) -> int:
    """Source frames needed to produce one CLIP_LEN window at SR, including resampler margin."""
    if source_sr == SR:
        return CLIP_LEN
    return int(np.ceil(CLIP_LEN * source_sr / SR)) + RESAMPLE_MARGIN_FRAMES


def _read_source_segment(
    path: Path,
    start_frame: int,
    source_sr: int,
) -> np.ndarray:
    """Read ONE window's worth of audio from `path` starting at `start_frame`, resampled to SR.

    DEMAND recordings are 300 s at 48 kHz -- 14.4 M frames. Decoding and resampling all of that
    to use 3 s of it, once per mixture, would dominate generation time. This reads only the span
    it needs.

    Postcondition: returns exactly CLIP_LEN float32 samples at SR.
    Raises: ValueError if the file yields fewer frames than requested or non-finite samples.
    """
    frames = _source_frames_for_window(source_sr)
    segment, read_sr = sf.read(
        str(path),
        start=int(start_frame),
        frames=frames,
        dtype="float32",
        always_2d=False,
    )
    if read_sr != source_sr:
        raise ValueError(f"{path} reported {source_sr} Hz at index time, {read_sr} Hz now")
    if segment.ndim == 2:
        segment = segment.mean(axis=1)
    if segment.shape[0] < frames:
        raise ValueError(
            f"{path}: read {segment.shape[0]} frames from offset {start_frame}; expected {frames}"
        )
    if not np.all(np.isfinite(segment)):
        raise ValueError(f"{path} contains non-finite samples at offset {start_frame}")
    if read_sr != SR:
        segment = librosa.resample(segment, orig_sr=read_sr, target_sr=SR)
    if segment.shape[0] < CLIP_LEN:
        raise ValueError(
            f"{path}: resampled segment is {segment.shape[0]} samples; expected >= {CLIP_LEN}"
        )
    return np.asarray(segment[:CLIP_LEN], dtype=np.float32)


def load_esc50_index() -> dict[str, list[Esc50Clip]]:
    """Select the ESC-50 clips for each project noise category, with their corpus metadata.

    Postcondition: returns {noise_type: [Esc50Clip, ...]} ordered by filename, 40 per ESC-50 class.
    The filename ordering is load-bearing: `draw_noise` indexes into this list with a seeded RNG,
    so any change to the order changes which clip a given seed selects.
    Raises: FileNotFoundError if audio or metadata is absent; ValueError on a missing column or an
    unexpected clip count.
    """
    if not ESC50_DIR.exists():
        raise FileNotFoundError(
            f"ESC-50 audio not found at {ESC50_DIR}. "
            "Set RISE_NOISE_ROOT or download ESC-50."
        )
    if not ESC50_META.exists():
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {ESC50_META}; both audio and meta/esc50.csv "
            "are required."
        )
    metadata = pd.read_csv(ESC50_META)
    required = {"filename", "target", "category", "fold"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"{ESC50_META} is missing columns: {sorted(missing_columns)}. "
            "`category` and `fold` are required so per-mixture provenance can record which "
            "original ESC-50 class was drawn; the standard meta/esc50.csv contains both."
        )
    index: dict[str, list[Esc50Clip]] = {}
    for noise_type, targets in ESC50_TARGETS.items():
        selected = metadata[metadata["target"].isin(list(targets))]
        clips = [
            Esc50Clip(
                path=ESC50_DIR / str(row.filename),
                target=int(row.target),
                category=str(row.category),
                fold=int(row.fold),
            )
            for row in selected.sort_values("filename").itertuples(index=False)
        ]
        missing_paths = [clip.path for clip in clips if not clip.path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} ESC-50 files are missing; first: {missing_paths[0]}"
            )
        # 40 clips per ESC-50 class, so the expected count follows the BLOCK WIDTH. This was
        # hardcoded to 800, which silently assumed every category spans 20 classes -- true for
        # natural and mechanical, false for audience (10 classes, 400 clips), which was then
        # reported as "ESC-50 unavailable" and skipped.
        expected = 40 * len(list(targets))
        if len(clips) != expected:
            raise ValueError(
                f"Expected {expected} ESC-50 {noise_type} clips "
                f"({len(list(targets))} classes x 40), found {len(clips)}"
            )
        index[noise_type] = clips
    return index


def esc50_category_map(index: dict[str, list[Esc50Clip]]) -> dict[str, dict[str, int]]:
    """Which original ESC-50 categories make up each project noise category, and how many clips.

    Recorded in the manifest so a paper can state the composition of "natural" and "mechanical"
    exactly, rather than citing a numeric target range and leaving the reader to look it up.
    """
    composition: dict[str, dict[str, int]] = {}
    for noise_type, clips in index.items():
        counts: dict[str, int] = {}
        for clip in clips:
            key = f"{clip.target}:{clip.category}"
            counts[key] = counts.get(key, 0) + 1
        composition[noise_type] = dict(sorted(counts.items(), key=lambda kv: int(kv[0].split(":")[0])))
    return composition


def esc50_corpus_provenance(index: dict[str, list[Esc50Clip]]) -> dict[str, object]:
    """Record the corpus metadata and extracted-file inventory used by the sweep.

    Returns a `used: False` record when no ESC-50 category is in the configured grid. A white-noise
    only sweep is a legitimate configuration -- Gaussian noise needs no corpus -- and it should not
    require an ESC-50 download just to write a manifest.
    """
    if not any(index.values()):
        return {"used": False, "reason": "no ESC-50 noise type in the configured grid"}
    inventory = hashlib.sha256()
    all_paths = sorted({clip.path for clips in index.values() for clip in clips})
    for path in all_paths:
        relative = path.relative_to(ESC50_ROOT).as_posix()
        inventory.update(f"{relative}\0{sha256_file(path)}\n".encode())
    archive_candidates = [
        NOISE_ROOT / "esc50.zip",
        NOISE_ROOT / "ESC-50-master.zip",
    ]
    archive = next((path for path in archive_candidates if path.is_file()), None)
    return {
        "used": True,
        "metadata_path": str(ESC50_META.resolve()),
        "metadata_sha256": sha256_file(ESC50_META),
        "selected_corpus_sha256": inventory.hexdigest(),
        "selected_file_count": len(all_paths),
        "archive_path": None if archive is None else str(archive.resolve()),
        "archive_sha256": None if archive is None else sha256_file(archive),
        "target_ranges": {
            noise_type: [int(min(targets)), int(max(targets))]
            for noise_type, targets in ESC50_TARGETS.items()
        },
        "category_composition": esc50_category_map(index),
    }


def _read_source_noise(path: Path) -> tuple[np.ndarray, int]:
    noise, source_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if noise.ndim == 2:
        noise = noise.mean(axis=1)
    elif noise.ndim != 1:
        raise ValueError(f"Unexpected ESC-50 waveform shape {noise.shape} at {path}")
    if not np.all(np.isfinite(noise)):
        raise ValueError(f"ESC-50 file contains non-finite samples: {path}")
    if source_sr != SR:
        noise = librosa.resample(noise, orig_sr=source_sr, target_sr=SR)
    return np.asarray(noise, dtype=np.float32), int(source_sr)


def draw_noise(
    noise_type: str,
    rng: np.random.Generator,
    esc_index: dict[str, list[Esc50Clip]],
    demand_index: dict[str, list[DemandRecording]] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Draw one centered, non-silent realization and return its source provenance.

    Preconditions: `demand_index` is required for any noise type in DEMAND_TARGETS and ignored
    otherwise, so a caller that never asks for `studio` needs no DEMAND download.
    Postcondition: the returned provenance dict always carries `noise_target`, `noise_category`,
    `noise_fold`, `noise_environment` and `noise_channel`. Fields that do not apply to the corpus
    behind a given type are None rather than absent -- generated white noise has no clip, and
    DEMAND has no ESC-50 target -- which keeps the provenance CSV one schema across every type.
    """
    if noise_type == "white":
        noise = rng.standard_normal(CLIP_LEN).astype(np.float32)
        noise = (noise.astype(np.float64) - np.mean(noise, dtype=np.float64)).astype(
            np.float32
        )
        return noise, {
            "noise_source": "generated_gaussian",
            "noise_source_sr": SR,
            "crop_start_resampled_sample": 0,
            "noise_target": None,
            "noise_category": None,
            "noise_fold": None,
            "noise_environment": None,
            "noise_channel": None,
        }

    if noise_type in DEMAND_TARGETS:
        if demand_index is None:
            raise ValueError(
                f"{noise_type} needs a DEMAND index; call load_demand_index() and pass it"
            )
        recordings = demand_index[noise_type]
        for _attempt in range(20):
            recording = recordings[int(rng.integers(len(recordings)))]
            # Crop offset is drawn in SOURCE frames, because that is what sf.read seeks in.
            last_start = recording.frames - _source_frames_for_window(recording.samplerate)
            start = int(rng.integers(0, max(last_start, 0) + 1))
            segment = _read_source_segment(recording.path, start, recording.samplerate)
            segment = (
                segment.astype(np.float64) - np.mean(segment, dtype=np.float64)
            ).astype(np.float32)
            if (
                float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
                >= MIN_CENTERED_NOISE_RMS
            ):
                return segment, {
                    "noise_source": f"{recording.environment}/{recording.channel}.wav",
                    "noise_source_path": str(recording.path.resolve()),
                    "noise_source_sr": recording.samplerate,
                    "crop_start_resampled_sample": start,
                    "noise_target": None,
                    "noise_category": recording.grouping,
                    "noise_fold": None,
                    "noise_environment": recording.environment,
                    "noise_channel": recording.channel,
                }
        raise ValueError(
            f"Unable to draw a non-silent centered {noise_type} DEMAND segment after 20 attempts"
        )

    if noise_type not in ESC50_TARGETS:
        raise ValueError(f"Unknown noise type: {noise_type}")

    clips = esc_index[noise_type]
    for _attempt in range(20):
        clip = clips[int(rng.integers(len(clips)))]
        noise, source_sr = _read_source_noise(clip.path)
        if noise.size == 0:
            continue
        if noise.size < CLIP_LEN:
            noise = np.tile(noise, int(np.ceil(CLIP_LEN / noise.size)))
        start = int(rng.integers(0, max(noise.size - CLIP_LEN, 0) + 1))
        segment = np.asarray(noise[start : start + CLIP_LEN], dtype=np.float32)
        segment = (
            segment.astype(np.float64) - np.mean(segment, dtype=np.float64)
        ).astype(np.float32)
        if (
            float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
            >= MIN_CENTERED_NOISE_RMS
        ):
            return segment, {
                "noise_source": clip.path.relative_to(ESC50_ROOT).as_posix(),
                "noise_source_path": str(clip.path.resolve()),
                "noise_source_sr": source_sr,
                "crop_start_resampled_sample": start,
                "noise_target": clip.target,
                "noise_category": clip.category,
                "noise_fold": clip.fold,
                "noise_environment": None,
                "noise_channel": None,
            }
    raise ValueError(
        f"Unable to draw a non-silent centered {noise_type} ESC-50 segment "
        "after 20 attempts"
    )


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: int | float,
) -> tuple[np.ndarray, float, float]:
    p_signal = float(np.mean(clean.astype(np.float64) ** 2))
    p_noise = float(np.mean(noise.astype(np.float64) ** 2))
    if not np.isfinite(p_signal) or p_signal <= 0:
        raise ValueError("Clean signal has zero or invalid power")
    if not np.isfinite(p_noise) or p_noise <= 0:
        raise ValueError("Noise signal has zero or invalid power")
    alpha = float(
        np.sqrt(p_signal / (p_noise * (10 ** (float(snr_db) / 10.0))))
    )
    return (clean + alpha * noise).astype(np.float32), alpha, p_signal


def measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    added = noisy.astype(np.float64) - clean.astype(np.float64)
    return float(
        10
        * np.log10(
            np.mean(clean.astype(np.float64) ** 2) / np.mean(added**2)
        )
    )


def read_audio_window(path: str | Path) -> np.ndarray:
    """Read one canonical/noisy window without silently padding, resampling, or clipping."""
    path = Path(path)
    waveform, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if sample_rate != SR:
        raise ValueError(f"{path} has sample rate {sample_rate}; expected {SR}")
    if waveform.ndim != 1:
        raise ValueError(f"{path} is not mono; shape={waveform.shape}")
    if waveform.shape != (CLIP_LEN,):
        raise ValueError(
            f"{path} has {len(waveform)} samples; expected exactly {CLIP_LEN}"
        )
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"{path} contains non-finite samples")
    return waveform


def load_clean(relative_path: str, *, data_root: str | Path = ROOT) -> np.ndarray:
    return read_audio_window(Path(data_root) / relative_path)


def window_id_of(relative_path: str) -> str:
    return Path(relative_path).stem


def test_windows(
    *,
    windows_csv: str | Path = WINDOWS_CSV,
) -> pd.DataFrame:
    windows_csv = Path(windows_csv)
    assert_artifact_fingerprint(windows_csv, "step5_normalize")
    frame = pd.read_csv(windows_csv)
    required = {"window_path", "source_path", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{windows_csv} is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{windows_csv} contains no test windows")
    window_ids = frame["window_path"].map(window_id_of)
    duplicates = sorted(window_ids[window_ids.duplicated(keep=False)].unique())
    if duplicates:
        raise ValueError(
            "Window stems are not unique and would overwrite noisy files: "
            + ", ".join(duplicates[:5])
        )
    return frame


def out_path(
    noise_type: str,
    snr: int,
    window_id: str,
    *,
    replicate: int = 0,
    noisy_dir: str | Path = NOISY_DIR,
) -> Path:
    """Where one mixture lives.

    The replicate directory is always present, even when N_REPLICATES is 1. A layout that changes
    shape depending on a config value needs two code paths on every reader, and the second one is
    the one that never gets tested.
    """
    return (
        Path(noisy_dir)
        / noise_type
        / f"snr{snr}"
        / f"r{int(replicate)}"
        / f"{window_id}.wav"
    )


def _write_wav_atomic(path: Path, waveform: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        sf.write(
            str(temporary),
            waveform,
            SR,
            format="WAV",
            subtype="FLOAT",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_generation_target_is_empty(
    noisy_dir: Path,
    *,
    noise_types: tuple[str, ...] | None = None,
) -> None:
    manifest = noisy_dir / NOISE_MANIFEST_NAME
    if manifest.exists():
        raise FileExistsError(
            f"{manifest} already records a complete sweep; refusing to overwrite it"
        )
    for noise_type in (NOISE_TYPES if noise_types is None else noise_types):
        directory = noisy_dir / noise_type
        if directory.exists() and next(directory.rglob("*.wav"), None) is not None:
            raise FileExistsError(
                f"{directory} contains an incomplete or unrecorded sweep. "
                "Move it aside before generating a new canonical set."
            )


def validate(n_samples: int = 5) -> None:
    windows = test_windows()
    # Only load a corpus the grid actually asks for: a white-only sweep must not require an
    # ESC-50 or DEMAND download.
    esc_index = (
        load_esc50_index()
        if any(noise_type in ESC50_TARGETS for noise_type in NOISE_TYPES)
        else {}
    )
    demand_index = (
        load_demand_index()
        if any(noise_type in DEMAND_TARGETS for noise_type in NOISE_TYPES)
        else {}
    )
    identity = dataset_build_identity()
    fingerprint = dataset_fingerprint(identity)
    print(f"data root: {ROOT}")
    print(f"dataset fingerprint: {fingerprint}")
    print(f"test windows: {len(windows)}")

    rows: list[dict[str, object]] = []
    picked = np.random.default_rng(0).choice(
        len(windows),
        size=min(n_samples, len(windows)),
        replace=False,
    )
    for index in picked:
        relative_path = str(windows.iloc[index]["window_path"])
        window_id = window_id_of(relative_path)
        clean = load_clean(relative_path)
        clean_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
        if abs(clean_rms - TARGET_RMS) >= 0.05:
            raise ValueError(
                f"{window_id}: RMS {clean_rms:.4f} is far from TARGET_RMS {TARGET_RMS}"
            )
        for noise_type in NOISE_TYPES:
            seed = window_seed(window_id, noise_type, fingerprint)
            noise, _ = draw_noise(
                noise_type,
                np.random.default_rng(seed),
                esc_index,
                demand_index=demand_index,
            )
            added_components = []
            for snr in SNRS:
                noisy, _, _ = mix_at_snr(clean, noise, snr)
                achieved = measured_snr(clean, noisy)
                added_components.append(noisy - clean)
                rows.append(
                    {
                        "noise": noise_type,
                        "target_db": snr,
                        "measured_db": achieved,
                        "error_db": achieved - snr,
                        "peak": float(np.abs(noisy).max()),
                    }
                )
            base = added_components[0] / (
                np.linalg.norm(added_components[0]) + 1e-12
            )
            similarities = [
                float(
                    abs(
                        np.dot(
                            component
                            / (np.linalg.norm(component) + 1e-12),
                            base,
                        )
                    )
                )
                for component in added_components
            ]
            worst_deviation = 1 - min(similarities)
            if worst_deviation > MAX_REALIZATION_COSINE_DEVIATION:
                raise AssertionError(
                    f"{noise_type} did not reuse one realization across SNRs: "
                    f"worst cosine deviation {worst_deviation:.3e} exceeds "
                    f"{MAX_REALIZATION_COSINE_DEVIATION:.0e} "
                    f"(window {window_id}; a re-drawn realization measures ~1.0, "
                    f"float32 rounding measures ~1e-7)"
                )

    results = pd.DataFrame(rows)
    worst_error = float(results["error_db"].abs().max())
    print(
        results.groupby(["noise", "target_db"])
        .agg(
            measured_db=("measured_db", "mean"),
            max_abs_error_db=("error_db", lambda values: values.abs().max()),
            max_peak=("peak", "max"),
        )
        .round(6)
        .to_string()
    )
    if worst_error >= MAX_SNR_ERROR_DB:
        raise AssertionError(
            f"Worst measured SNR error {worst_error:.6f} dB exceeds "
            f"{MAX_SNR_ERROR_DB} dB"
        )
    sample_relative = str(windows.iloc[int(picked[0])]["window_path"])
    sample_window_id = window_id_of(sample_relative)
    sample_clean = load_clean(sample_relative)
    listen_dir = Path(NOISY_DIR) / "_validation_samples"
    listen_dir.mkdir(parents=True, exist_ok=True)
    _write_wav_atomic(listen_dir / f"{sample_window_id}__clean.wav", sample_clean)
    for noise_type in NOISE_TYPES:
        seed = window_seed(sample_window_id, noise_type, fingerprint)
        noise, _ = draw_noise(
            noise_type,
            np.random.default_rng(seed),
            esc_index,
            demand_index=demand_index,
        )
        sample_noisy, _, _ = mix_at_snr(sample_clean, noise, 0)
        _write_wav_atomic(
            listen_dir / f"{sample_window_id}__{noise_type}_snr0.wav",
            sample_noisy,
        )
    print(f"validation passed; worst SNR error={worst_error:.6f} dB")
    print(f"listenable 0 dB samples: {listen_dir}")


def generate(
    *,
    data_root: str | Path | None = None,
    windows_csv: str | Path | None = None,
    manifest_csv: str | Path | None = None,
    manifest_fingerprint: str | Path | None = None,
    noisy_dir: str | Path | None = None,
    only_noise_types: tuple[str, ...] | None = None,
    only_replicates: tuple[int, ...] | None = None,
    write_completion: bool = True,
) -> Path:
    """Materialize the canonical noisy test set.

    Every path is an optional argument resolved AT CALL TIME rather than a module-level default
    bound at import. That is deliberate: with import-time defaults this function could only ever run
    against the real data root, so the generation loop -- where the seeds are drawn, the mixtures are
    written and every diagnostic is computed -- had no way to be exercised by a test. The provenance
    it writes cannot be reconstructed after the fact, so it is the last place that should be
    untested.

    Defaults are unchanged for production callers, which pass nothing.
    """
    root = Path(ROOT if data_root is None else data_root)
    windows_csv = WINDOWS_CSV if windows_csv is None else Path(windows_csv)
    manifest_csv = MANIFEST_IN if manifest_csv is None else Path(manifest_csv)
    manifest_fingerprint = (
        MANIFEST_FINGERPRINT
        if manifest_fingerprint is None
        else Path(manifest_fingerprint)
    )
    # SUBSET GENERATION, for streaming a sweep that will not fit on disk all at once. The chunk
    # is a (noise_type, replicate) pair and never a single SNR: the loop below draws noise ONCE
    # per (window, noise_type, replicate) and rescales that one draw across every SNR, so
    # splitting on SNR would force a redraw and make bit-identity depend on RNG replay instead of
    # on running the same code path.
    selected_types = tuple(NOISE_TYPES) if only_noise_types is None else tuple(only_noise_types)
    selected_replicates = (
        tuple(range(N_REPLICATES)) if only_replicates is None else tuple(only_replicates)
    )
    unknown_types = sorted(set(selected_types) - set(NOISE_TYPES))
    if unknown_types:
        raise ValueError(f"only_noise_types not in config.NOISE_TYPES: {unknown_types}")
    unknown_replicates = sorted(set(selected_replicates) - set(range(N_REPLICATES)))
    if unknown_replicates:
        raise ValueError(f"only_replicates outside range(N_REPLICATES): {unknown_replicates}")
    is_partial = (
        selected_types != tuple(NOISE_TYPES)
        or selected_replicates != tuple(range(N_REPLICATES))
    )
    # A partial sweep must not be able to leave behind something a consumer reads as complete.
    # validate_noise_manifest checks n_files and provenance_rows against the FULL grid, so a
    # partial manifest would fail there -- but it would also overwrite the canonical provenance
    # on its way to failing. Make the invalid combination unrepresentable instead.
    if is_partial and write_completion:
        raise ValueError(
            "refusing to write a completion manifest for a partial sweep "
            f"(types={selected_types}, replicates={selected_replicates}); "
            "pass write_completion=False"
        )

    windows = test_windows(windows_csv=windows_csv)
    # Only load the corpus if the grid actually asks for it: Gaussian noise needs none, so a
    # white-only sweep must not demand an ESC-50 download.
    esc_index: dict[str, list[Esc50Clip]] = (
        load_esc50_index()
        if any(noise_type in ESC50_TARGETS for noise_type in selected_types)
        else {}
    )
    demand_index: dict[str, list[DemandRecording]] = (
        load_demand_index()
        if any(noise_type in DEMAND_TARGETS for noise_type in selected_types)
        else {}
    )
    identity = dataset_build_identity(
        manifest_csv=manifest_csv,
        manifest_fingerprint=manifest_fingerprint,
        windows_csv=windows_csv,
    )
    fingerprint = dataset_fingerprint(identity)
    noisy_dir = Path(NOISY_DIR if noisy_dir is None else noisy_dir)
    _ensure_generation_target_is_empty(noisy_dir, noise_types=selected_types)

    for noise_type in selected_types:
        for snr in SNRS:
            for replicate in selected_replicates:
                (noisy_dir / noise_type / f"snr{snr}" / f"r{replicate}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

    source_hashes: dict[Path, str] = {}
    provenance_rows: list[dict[str, object]] = []
    per_window = len(selected_types) * len(selected_replicates) * len(SNRS)
    total = len(windows) * per_window
    made = 0
    for row in windows.itertuples(index=False):
        relative_path = str(row.window_path)
        window_id = window_id_of(relative_path)
        clean_path = root / relative_path
        clean = read_audio_window(clean_path)
        clean_sha256 = sha256_file(clean_path)
        for noise_type in selected_types:
            for replicate in selected_replicates:
                # One independent draw per replicate; that draw is then rescaled across every SNR,
                # so the SNR axis and the realization axis stay separable.
                seed = window_seed(window_id, noise_type, fingerprint, replicate)
                noise, source = draw_noise(
                    noise_type,
                    np.random.default_rng(seed),
                    esc_index,
                    demand_index=demand_index,
                )
                source_path_value = source.get("noise_source_path")
                source_sha256 = None
                if source_path_value is not None:
                    source_path = Path(str(source_path_value))
                    if source_path not in source_hashes:
                        source_hashes[source_path] = sha256_file(source_path)
                    source_sha256 = source_hashes[source_path]
                for snr in SNRS:
                    noisy, alpha, signal_power = mix_at_snr(clean, noise, snr)
                    output_path = out_path(
                        noise_type,
                        snr,
                        window_id,
                        replicate=replicate,
                        noisy_dir=noisy_dir,
                    )
                    _write_wav_atomic(output_path, noisy)
                    reloaded = read_audio_window(output_path)
                    achieved = measured_snr(clean, reloaded)
                    if abs(achieved - snr) >= MAX_SNR_ERROR_DB:
                        raise AssertionError(
                            f"{output_path} achieved {achieved:.6f} dB; expected {snr}"
                        )
                    # Diagnostics come from the RELOADED file, so they describe what the models will
                    # actually read rather than the in-memory mixture.
                    diagnostics = mixture_diagnostics(clean, reloaded - clean)
                    provenance_rows.append(
                        {
                            "window_id": window_id,
                            "window_path": relative_path,
                            "clean_sha256": clean_sha256,
                            "noise_type": noise_type,
                            "snr_db": snr,
                            "replicate": replicate,
                            "seed": seed,
                            "noise_source": source["noise_source"],
                            "noise_source_sha256": source_sha256,
                            "noise_source_sr": source["noise_source_sr"],
                            "crop_start_resampled_sample": source[
                                "crop_start_resampled_sample"
                            ],
                            "noise_target": source["noise_target"],
                            "noise_category": source["noise_category"],
                            "noise_fold": source["noise_fold"],
                            "alpha": alpha,
                            "signal_power": signal_power,
                            "unscaled_noise_power": float(
                                np.mean(noise.astype(np.float64) ** 2)
                            ),
                            "realized_snr_db": achieved,
                            "peak": float(np.abs(reloaded).max()),
                            **diagnostics,
                            "output_path": str(output_path.relative_to(root)),
                            "output_sha256": sha256_file(output_path),
                        }
                    )
                    made += 1
        if made % 3000 < per_window:
            print(f"{made}/{total}", flush=True)

    provenance = pd.DataFrame(provenance_rows)
    # A chunk's provenance is written under its OWN name. Writing it to NOISE_PROVENANCE_NAME
    # would leave a file that looks canonical while covering one sixth of the grid, and the next
    # chunk would silently overwrite it.
    provenance_name = (
        NOISE_PROVENANCE_NAME
        if not is_partial
        else "noise_provenance_"
        + "_".join(selected_types)
        + "_r"
        + "".join(str(r) for r in selected_replicates)
        + ".csv"
    )
    provenance_path = noisy_dir / provenance_name
    temporary_provenance = provenance_path.with_name(
        f".{provenance_path.name}.tmp"
    )
    try:
        provenance.to_csv(temporary_provenance, index=False)
        temporary_provenance.replace(provenance_path)
    finally:
        temporary_provenance.unlink(missing_ok=True)
    _assert_residual_dc_is_bounded(
        provenance,
        provenance_name=provenance_name,
    )

    if not write_completion:
        print(
            f"generated {made} files under {noisy_dir} "
            f"(partial: types={selected_types}, replicates={selected_replicates})",
            flush=True,
        )
        print(f"wrote {provenance_path}", flush=True)
        return provenance_path

    manifest = {
        "manifest_version": NOISE_MANIFEST_VERSION,
        "state": "complete",
        "dataset": identity,
        "snrs": SNRS,
        "noise_types": NOISE_TYPES,
        "n_replicates": N_REPLICATES,
        "n_test_windows": int(len(windows)),
        "n_files": made,
        "waveform_format": {
            "sample_rate": SR,
            "samples": CLIP_LEN,
            "channels": 1,
            "subtype": "FLOAT",
            "post_mix_normalization": False,
        },
        "seed_scheme": SEED_SCHEME,
        "one_realization_scaled_to_all_snrs": True,
        "noise_preprocessing": noise_preprocessing_protocol(),
        "diagnostics": {
            **diagnostic_protocol(),
            "note": (
                "snr_db is the requested whole-window whole-spectrum SNR; these columns record "
                "where in frequency, when in time, and at which model rate that SNR actually lands"
            ),
        },
        "esc50": esc50_corpus_provenance(esc_index),
        "provenance_file": NOISE_PROVENANCE_NAME,
        "provenance_sha256": sha256_file(provenance_path),
        "provenance_rows": int(len(provenance)),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "librosa": librosa.__version__,
            "soundfile": sf.__version__,
        },
    }
    manifest_path = noisy_dir / NOISE_MANIFEST_NAME
    _write_text_atomic(manifest_path, json.dumps(manifest, indent=2) + "\n")
    print(f"generated {made} files under {noisy_dir}")
    print(f"wrote {manifest_path}")
    return provenance_path


def validate_noise_manifest(
    *,
    noisy_dir: str | Path = NOISY_DIR,
    data_root: str | Path = ROOT,
    windows_csv: str | Path = WINDOWS_CSV,
    manifest_csv: str | Path = MANIFEST_IN,
    manifest_fingerprint: str | Path = MANIFEST_FINGERPRINT,
    verify_audio_hashes: bool = False,
) -> dict[str, object]:
    """Fail closed unless a complete sweep matches the current canonical dataset."""
    noisy_dir = Path(noisy_dir)
    manifest_path = noisy_dir / NOISE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing completed noise manifest: {manifest_path}. "
            "Run noise_sweep --generate first."
        )
    manifest = _read_json(manifest_path)
    expected_identity = dataset_build_identity(
        manifest_csv=manifest_csv,
        manifest_fingerprint=manifest_fingerprint,
        windows_csv=windows_csv,
    )
    checks = {
        "manifest_version": NOISE_MANIFEST_VERSION,
        "state": "complete",
        "snrs": SNRS,
        "noise_types": NOISE_TYPES,
        "n_replicates": N_REPLICATES,
        "dataset": expected_identity,
        "one_realization_scaled_to_all_snrs": True,
        "noise_preprocessing": noise_preprocessing_protocol(),
        "waveform_format": {
            "sample_rate": SR,
            "samples": CLIP_LEN,
            "channels": 1,
            "subtype": "FLOAT",
            "post_mix_normalization": False,
        },
        "seed_scheme": SEED_SCHEME,
    }
    mismatches = [
        key for key, expected in checks.items() if manifest.get(key) != expected
    ]
    diagnostics = manifest.get("diagnostics")
    if not isinstance(diagnostics, dict) or any(
        diagnostics.get(key) != expected
        for key, expected in diagnostic_protocol().items()
    ):
        mismatches.append("diagnostics")
    if mismatches:
        raise ValueError(
            f"{manifest_path} does not match the current protocol/build: {mismatches}"
        )

    windows = test_windows(windows_csv=windows_csv)
    expected_rows = len(windows) * len(NOISE_TYPES) * N_REPLICATES * len(SNRS)
    if manifest.get("n_test_windows") != len(windows):
        raise ValueError("Noise manifest test-window count is stale")
    if manifest.get("n_files") != expected_rows:
        raise ValueError("Noise manifest file count is incomplete")
    if manifest.get("provenance_rows") != expected_rows:
        raise ValueError("Noise manifest provenance-row count is incomplete")

    provenance_path = noisy_dir / str(manifest.get("provenance_file", ""))
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing noise provenance: {provenance_path}")
    if sha256_file(provenance_path) != manifest.get("provenance_sha256"):
        raise ValueError("Noise provenance hash does not match the manifest")
    provenance = pd.read_csv(provenance_path)
    required = {
        "window_id",
        "noise_type",
        "snr_db",
        "seed",
        "noise_source",
        "noise_source_sha256",
        "crop_start_resampled_sample",
        "noise_target",
        "noise_category",
        "noise_fold",
        "replicate",
        "unscaled_noise_power",
        "realized_snr_db",
        "output_path",
        "output_sha256",
    } | set(DIAGNOSTIC_COLUMNS)
    missing = required - set(provenance.columns)
    if missing:
        raise ValueError(
            f"Noise provenance is missing columns: {sorted(missing)}"
        )
    if len(provenance) != expected_rows:
        raise ValueError(
            f"Noise provenance has {len(provenance)} rows; expected {expected_rows}"
        )
    if provenance["output_path"].duplicated().any():
        raise ValueError("Noise provenance contains duplicate output paths")
    if (
        provenance["realized_snr_db"].astype(float)
        - provenance["snr_db"].astype(float)
    ).abs().max() >= MAX_SNR_ERROR_DB:
        raise ValueError("Noise provenance contains an out-of-tolerance SNR")

    expected_ids = set(windows["window_path"].map(window_id_of))
    if set(provenance["window_id"]) != expected_ids:
        raise ValueError("Noise provenance window IDs differ from windows.csv")
    expected_conditions = {
        (window_id, noise_type, snr, replicate)
        for window_id in expected_ids
        for noise_type in NOISE_TYPES
        for snr in SNRS
        for replicate in range(N_REPLICATES)
    }
    observed_conditions = set(
        zip(
            provenance["window_id"],
            provenance["noise_type"],
            provenance["snr_db"],
            provenance["replicate"],
        )
    )
    if observed_conditions != expected_conditions:
        raise ValueError("Noise provenance does not contain the exact condition grid")
    grouped = provenance.groupby(
        ["window_id", "noise_type", "replicate"], dropna=False
    )
    for column in (
        "seed",
        "noise_source",
        "noise_source_sha256",
        "crop_start_resampled_sample",
        "noise_target",
        "noise_category",
        "noise_fold",
        "unscaled_noise_power",
        "noise_active_fraction",
        "signal_active_fraction",
        "snr_signal_active_frames",
    ):
        if grouped[column].nunique(dropna=False).max() != 1:
            raise ValueError(
                f"Noise realization provenance changes across SNR in column {column}"
            )

    # DC offset inflates measured power, so a DC-heavy noise clip is quieter than its SNR label
    # claims. Version 6 centers every realization before scaling; residual DC above this threshold
    # therefore means the materialized corpus violates its recorded preprocessing protocol.
    _assert_residual_dc_is_bounded(
        provenance,
        provenance_name=provenance_path.name,
    )

    for row in provenance.itertuples(index=False):
        path = Path(data_root) / str(row.output_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated noisy window: {path}")
        if verify_audio_hashes and sha256_file(path) != row.output_sha256:
            raise ValueError(f"Generated noisy-window hash mismatch: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--check-generated",
        action="store_true",
        help="validate the completed manifest and every recorded output path",
    )
    parser.add_argument(
        "--verify-audio-hashes",
        action="store_true",
        help="with --check-generated, hash every generated WAV",
    )
    args = parser.parse_args()
    selected = sum((args.validate, args.generate, args.check_generated))
    if selected != 1:
        parser.error("pass exactly one of --validate, --generate, --check-generated")
    if args.validate:
        validate()
    elif args.generate:
        generate()
    else:
        validate_noise_manifest(verify_audio_hashes=args.verify_audio_hashes)
        print("completed noise manifest is valid")


if __name__ == "__main__":
    main()
