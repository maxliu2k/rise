"""Generate the one shared noisy TEST set used by every model.

The clean train/validation data and all fitted models remain untouched. One deterministic noise
realization is drawn for each (dataset build, test window, noise type), then scaled to every SNR.
The manifest is written last and is the completion marker consumed by model evaluators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import warnings
from pathlib import Path

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
    MANIFEST_FINGERPRINT,
    MANIFEST_IN,
    ROOT,
    SR,
    TARGET_RMS,
    WINDOWS_CSV,
    WORK,
    artifact_fingerprint_path,
    assert_artifact_fingerprint,
    config_fingerprint,
)

warnings.filterwarnings("ignore")

SNRS = [20, 10, 5, 0, -5]
NOISE_TYPES = ["white", "natural", "mechanical"]
NOISY_DIR = WORK / "windows_noisy"
NOISE_MANIFEST_NAME = "noise_manifest.json"
NOISE_PROVENANCE_NAME = "noise_provenance.csv"
NOISE_MANIFEST_VERSION = 2
NOISE_ROOT = Path(
    os.environ.get("RISE_NOISE_ROOT", Path.home() / "Downloads/noise_sources")
)
ESC50_ROOT = NOISE_ROOT / "ESC-50-master"
ESC50_DIR = ESC50_ROOT / "audio"
ESC50_META = ESC50_ROOT / "meta" / "esc50.csv"
CLIP_LEN = int(round(3.0 * SR))
MAX_SNR_ERROR_DB = 0.1

# ESC-50 target blocks: 0-19 animals/natural, 20-29 human non-speech (excluded),
# 30-49 domestic/urban.
ESC50_TARGETS = {"natural": range(0, 20), "mechanical": range(30, 50)}


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
        "prep_data",
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
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["dataset_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return identity


def dataset_fingerprint(identity: dict[str, object] | None = None) -> str:
    """Full digest of the actual canonical dataset build."""
    value = dataset_build_identity() if identity is None else identity
    return str(value["dataset_fingerprint"])


def window_seed(
    window_id: str,
    noise_type: str,
    fingerprint: str | None = None,
) -> int:
    """Stable seed; SNR is intentionally absent so one realization spans the curve."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type: {noise_type}")
    build = dataset_fingerprint() if fingerprint is None else fingerprint
    key = f"{build}|{window_id}|{noise_type}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def load_esc50_index() -> dict[str, list[Path]]:
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
    required = {"filename", "target"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"{ESC50_META} is missing columns: {sorted(missing_columns)}"
        )
    index: dict[str, list[Path]] = {}
    for noise_type, targets in ESC50_TARGETS.items():
        selected = metadata[metadata["target"].isin(list(targets))]
        paths = [ESC50_DIR / filename for filename in sorted(selected["filename"])]
        missing_paths = [path for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} ESC-50 files are missing; first: {missing_paths[0]}"
            )
        if len(paths) != 800:
            raise ValueError(
                f"Expected 800 ESC-50 {noise_type} clips, found {len(paths)}"
            )
        index[noise_type] = paths
    return index


def esc50_corpus_provenance(index: dict[str, list[Path]]) -> dict[str, object]:
    """Record the corpus metadata and extracted-file inventory used by the sweep."""
    inventory = hashlib.sha256()
    all_paths = sorted({path for paths in index.values() for path in paths})
    for path in all_paths:
        relative = path.relative_to(ESC50_ROOT).as_posix()
        inventory.update(f"{relative}\0{sha256_file(path)}\n".encode())
    archive_candidates = [
        NOISE_ROOT / "esc50.zip",
        NOISE_ROOT / "ESC-50-master.zip",
    ]
    archive = next((path for path in archive_candidates if path.is_file()), None)
    return {
        "metadata_path": str(ESC50_META.resolve()),
        "metadata_sha256": sha256_file(ESC50_META),
        "selected_corpus_sha256": inventory.hexdigest(),
        "selected_file_count": len(all_paths),
        "archive_path": None if archive is None else str(archive.resolve()),
        "archive_sha256": None if archive is None else sha256_file(archive),
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
    esc_index: dict[str, list[Path]],
) -> tuple[np.ndarray, dict[str, object]]:
    """Draw one non-silent realization and return its complete source provenance."""
    if noise_type == "white":
        noise = rng.standard_normal(CLIP_LEN).astype(np.float32)
        return noise, {
            "noise_source": "generated_gaussian",
            "noise_source_sr": SR,
            "crop_start_resampled_sample": 0,
        }
    if noise_type not in ESC50_TARGETS:
        raise ValueError(f"Unknown noise type: {noise_type}")

    paths = esc_index[noise_type]
    for _attempt in range(20):
        path = paths[int(rng.integers(len(paths)))]
        noise, source_sr = _read_source_noise(path)
        if noise.size == 0:
            continue
        if noise.size < CLIP_LEN:
            noise = np.tile(noise, int(np.ceil(CLIP_LEN / noise.size)))
        start = int(rng.integers(0, max(noise.size - CLIP_LEN, 0) + 1))
        segment = np.asarray(noise[start : start + CLIP_LEN], dtype=np.float32)
        if float(np.sqrt(np.mean(segment.astype(np.float64) ** 2))) >= 1e-6:
            return segment, {
                "noise_source": path.relative_to(ESC50_ROOT).as_posix(),
                "noise_source_path": str(path.resolve()),
                "noise_source_sr": source_sr,
                "crop_start_resampled_sample": start,
            }
    raise ValueError(
        f"Unable to draw a non-silent {noise_type} ESC-50 segment after 20 attempts"
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
    noisy_dir: str | Path = NOISY_DIR,
) -> Path:
    return Path(noisy_dir) / noise_type / f"snr{snr}" / f"{window_id}.wav"


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


def _ensure_generation_target_is_empty(noisy_dir: Path) -> None:
    manifest = noisy_dir / NOISE_MANIFEST_NAME
    if manifest.exists():
        raise FileExistsError(
            f"{manifest} already records a complete sweep; refusing to overwrite it"
        )
    for noise_type in NOISE_TYPES:
        directory = noisy_dir / noise_type
        if directory.exists() and next(directory.rglob("*.wav"), None) is not None:
            raise FileExistsError(
                f"{directory} contains an incomplete or unrecorded sweep. "
                "Move it aside before generating a new canonical set."
            )


def validate(n_samples: int = 5) -> None:
    windows = test_windows()
    esc_index = load_esc50_index()
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
            if min(similarities) <= 1 - 1e-6:
                raise AssertionError(
                    f"{noise_type} did not reuse one realization across SNRs"
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
        )
        sample_noisy, _, _ = mix_at_snr(sample_clean, noise, 0)
        _write_wav_atomic(
            listen_dir / f"{sample_window_id}__{noise_type}_snr0.wav",
            sample_noisy,
        )
    print(f"validation passed; worst SNR error={worst_error:.6f} dB")
    print(f"listenable 0 dB samples: {listen_dir}")


def generate() -> None:
    windows = test_windows()
    esc_index = load_esc50_index()
    identity = dataset_build_identity()
    fingerprint = dataset_fingerprint(identity)
    noisy_dir = Path(NOISY_DIR)
    _ensure_generation_target_is_empty(noisy_dir)

    for noise_type in NOISE_TYPES:
        for snr in SNRS:
            (noisy_dir / noise_type / f"snr{snr}").mkdir(
                parents=True,
                exist_ok=True,
            )

    source_hashes: dict[Path, str] = {}
    provenance_rows: list[dict[str, object]] = []
    total = len(windows) * len(NOISE_TYPES) * len(SNRS)
    made = 0
    for row in windows.itertuples(index=False):
        relative_path = str(row.window_path)
        window_id = window_id_of(relative_path)
        clean_path = Path(ROOT) / relative_path
        clean = read_audio_window(clean_path)
        clean_sha256 = sha256_file(clean_path)
        for noise_type in NOISE_TYPES:
            seed = window_seed(window_id, noise_type, fingerprint)
            noise, source = draw_noise(
                noise_type,
                np.random.default_rng(seed),
                esc_index,
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
                    noisy_dir=noisy_dir,
                )
                _write_wav_atomic(output_path, noisy)
                reloaded = read_audio_window(output_path)
                achieved = measured_snr(clean, reloaded)
                if abs(achieved - snr) >= MAX_SNR_ERROR_DB:
                    raise AssertionError(
                        f"{output_path} achieved {achieved:.6f} dB; expected {snr}"
                    )
                provenance_rows.append(
                    {
                        "window_id": window_id,
                        "window_path": relative_path,
                        "clean_sha256": clean_sha256,
                        "noise_type": noise_type,
                        "snr_db": snr,
                        "seed": seed,
                        "noise_source": source["noise_source"],
                        "noise_source_sha256": source_sha256,
                        "noise_source_sr": source["noise_source_sr"],
                        "crop_start_resampled_sample": source[
                            "crop_start_resampled_sample"
                        ],
                        "alpha": alpha,
                        "signal_power": signal_power,
                        "unscaled_noise_power": float(
                            np.mean(noise.astype(np.float64) ** 2)
                        ),
                        "realized_snr_db": achieved,
                        "peak": float(np.abs(reloaded).max()),
                        "output_path": str(output_path.relative_to(ROOT)),
                        "output_sha256": sha256_file(output_path),
                    }
                )
                made += 1
        if made % 3000 < len(NOISE_TYPES) * len(SNRS):
            print(f"{made}/{total}", flush=True)

    provenance = pd.DataFrame(provenance_rows)
    provenance_path = noisy_dir / NOISE_PROVENANCE_NAME
    temporary_provenance = provenance_path.with_name(
        f".{provenance_path.name}.tmp"
    )
    try:
        provenance.to_csv(temporary_provenance, index=False)
        temporary_provenance.replace(provenance_path)
    finally:
        temporary_provenance.unlink(missing_ok=True)

    manifest = {
        "manifest_version": NOISE_MANIFEST_VERSION,
        "state": "complete",
        "dataset": identity,
        "snrs": SNRS,
        "noise_types": NOISE_TYPES,
        "n_test_windows": int(len(windows)),
        "n_files": made,
        "waveform_format": {
            "sample_rate": SR,
            "samples": CLIP_LEN,
            "channels": 1,
            "subtype": "FLOAT",
            "post_mix_normalization": False,
        },
        "seed_scheme": (
            "sha256(dataset_fingerprint|window_id|noise_type)[:4]; SNR excluded"
        ),
        "one_realization_scaled_to_all_snrs": True,
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
        "dataset": expected_identity,
        "one_realization_scaled_to_all_snrs": True,
        "waveform_format": {
            "sample_rate": SR,
            "samples": CLIP_LEN,
            "channels": 1,
            "subtype": "FLOAT",
            "post_mix_normalization": False,
        },
        "seed_scheme": (
            "sha256(dataset_fingerprint|window_id|noise_type)[:4]; SNR excluded"
        ),
    }
    mismatches = [
        key for key, expected in checks.items() if manifest.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            f"{manifest_path} does not match the current protocol/build: {mismatches}"
        )

    windows = test_windows(windows_csv=windows_csv)
    expected_rows = len(windows) * len(NOISE_TYPES) * len(SNRS)
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
        "unscaled_noise_power",
        "realized_snr_db",
        "output_path",
        "output_sha256",
    }
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
        (window_id, noise_type, snr)
        for window_id in expected_ids
        for noise_type in NOISE_TYPES
        for snr in SNRS
    }
    observed_conditions = set(
        zip(
            provenance["window_id"],
            provenance["noise_type"],
            provenance["snr_db"],
        )
    )
    if observed_conditions != expected_conditions:
        raise ValueError("Noise provenance does not contain the exact condition grid")
    grouped = provenance.groupby(["window_id", "noise_type"], dropna=False)
    for column in (
        "seed",
        "noise_source",
        "noise_source_sha256",
        "crop_start_resampled_sample",
        "unscaled_noise_power",
    ):
        if grouped[column].nunique(dropna=False).max() != 1:
            raise ValueError(
                f"Noise realization provenance changes across SNR in column {column}"
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
