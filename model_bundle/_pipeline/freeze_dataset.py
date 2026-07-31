"""Seal a completed pipeline build with hashes of its metadata and normalized audio."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from instrument_robustness.audio_inventory import record_window_audio_inventory, verify_window_audio
from instrument_robustness.config import DATASET_FREEZE, ROOT, TARGET_LABELS, WINDOWS_CSV
from instrument_robustness.noise_sweep import dataset_build_identity


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conflicting_audio_labels(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append((row.window_path, row.label, _sha256(ROOT / row.window_path)))
    hashed = pd.DataFrame(rows, columns=["window_path", "label", "sha256"])
    bad = []
    for digest, group in hashed.groupby("sha256"):
        labels = sorted(group["label"].unique())
        if len(labels) > 1:
            bad.append({"sha256": digest, "labels": labels, "paths": sorted(group["window_path"])})
    return bad


def build_freeze() -> dict:
    record_window_audio_inventory(windows_csv=WINDOWS_CSV, data_root=ROOT)
    verified = verify_window_audio(windows_csv=WINDOWS_CSV, data_root=ROOT, required=True)
    frame = pd.read_csv(WINDOWS_CSV)
    conflicts = conflicting_audio_labels(frame)
    if conflicts:
        raise ValueError(f"Found {len(conflicts)} exact-audio group(s) with conflicting labels")
    labels = sorted(frame["label"].unique())
    if labels != sorted(TARGET_LABELS):
        raise ValueError(f"Dataset labels differ from TARGET_LABELS: {labels}")
    identity = dataset_build_identity()
    payload = {
        "state": "frozen",
        **identity,
        "window_audio_file_count": verified["file_count"],
        "window_counts_by_split": {
            key: int(value) for key, value in frame.groupby("split").size().sort_index().items()
        },
        "window_counts_by_label": {
            key: int(value) for key, value in frame.groupby("label").size().sort_index().items()
        },
        "cross_label_exact_audio_groups": 0,
    }
    DATASET_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    DATASET_FREEZE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    if DATASET_FREEZE.exists():
        raise SystemExit(f"Dataset is already sealed: {DATASET_FREEZE}")
    payload = build_freeze()
    print(f"sealed {payload['window_audio_file_count']} windows")
    print(f"dataset fingerprint: {payload['dataset_fingerprint']}")
    print(f"wrote {DATASET_FREEZE}")


if __name__ == "__main__":
    main()
