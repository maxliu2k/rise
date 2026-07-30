"""Hash the actual window WAV bytes, so a silently edited waveform cannot pass unnoticed.

AUDIT ITEM 16. The clean pipeline's fingerprints hash the CSVs, not the audio. `windows.csv` and its
sidecar pin the *metadata* — which windows exist, their splits, their RMS — but nothing pins the
sample values. A window rewritten in place, half-copied, truncated by a failed sync, or regenerated
under a different librosa keeps the same path, the same row and the same CSV hash, so every loader
accepts it and every downstream number silently changes.

Only the noise generator hashes clean audio today (`clean_sha256`, per test window). That protects
the noisy set and leaves the clean feature arrays, the SVM, and every clean result unprotected.

    inventory hash = sha256 over  "<relative path>\\0<file sha256>\\n"  for every window, path-sorted

Two properties make this the right shape. Sorting by path makes it independent of filesystem order,
and including the path means a *renamed* file changes the digest even if its bytes did not. The same
construction is already used for the ESC-50 corpus in `noise_sweep.esc50_corpus_provenance`.

Deliberately a separate, opt-in step rather than a hook inside Step 4 or 5: hashing ~1.1 GB takes
seconds, not milliseconds, and forcing it into every stage's happy path would slow the loop people
run most. Run it once after a build, then verify cheaply whenever provenance is in question.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from instrument_robustness.config import (
    ROOT,
    WINDOWS_CSV,
    artifact_fingerprint_path,
)

AUDIO_INVENTORY_KEY = "windows_audio_inventory_sha256"
AUDIO_COUNT_KEY = "windows_audio_file_count"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_audio_inventory(
    *,
    windows_csv: str | Path = WINDOWS_CSV,
    data_root: str | Path = ROOT,
) -> dict[str, object]:
    """Hash every window listed in `windows.csv`.

    Postcondition: `{"sha256", "file_count", "missing"}`. `missing` lists any path in the CSV with no
    file on disk; the digest covers only the files that exist, so a missing file is reported rather
    than silently changing the digest into something that looks like a content change.
    """
    windows_csv = Path(windows_csv)
    root = Path(data_root)
    frame = pd.read_csv(windows_csv)
    if "window_path" not in frame.columns:
        raise ValueError(f"{windows_csv} has no window_path column")

    inventory = hashlib.sha256()
    missing: list[str] = []
    counted = 0
    for relative in sorted(frame["window_path"].astype(str)):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        inventory.update(f"{relative}\0{_sha256_file(path)}\n".encode())
        counted += 1
    return {
        "sha256": inventory.hexdigest(),
        "file_count": counted,
        "missing": missing,
    }


def record_window_audio_inventory(
    *,
    windows_csv: str | Path = WINDOWS_CSV,
    data_root: str | Path = ROOT,
) -> dict[str, object]:
    """Add the inventory digest to `windows.csv`'s existing sidecar, in place.

    Postcondition: the sidecar gains `metadata.windows_audio_inventory_sha256` and
    `metadata.windows_audio_file_count`. The CSV itself is untouched, so its own recorded hash stays
    valid and nothing needs regenerating — this can be run against an existing build.
    Raises: FileNotFoundError if the sidecar is absent; ValueError if any window file is missing.
    """
    sidecar = artifact_fingerprint_path(Path(windows_csv))
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"No provenance sidecar at {sidecar}; run the pipeline before recording audio hashes."
        )
    result = window_audio_inventory(windows_csv=windows_csv, data_root=data_root)
    if result["missing"]:
        raise ValueError(
            f"{len(result['missing'])} window file(s) listed in {windows_csv} are absent; "
            f"first: {result['missing'][0]}. Refusing to record an inventory of a partial build."
        )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    metadata[AUDIO_INVENTORY_KEY] = result["sha256"]
    metadata[AUDIO_COUNT_KEY] = result["file_count"]
    payload["metadata"] = metadata
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"sidecar": str(sidecar), **result}


def verify_window_audio(
    *,
    windows_csv: str | Path = WINDOWS_CSV,
    data_root: str | Path = ROOT,
    required: bool = False,
) -> dict[str, object]:
    """Recompute the inventory and compare it against what the sidecar recorded.

    Postcondition: `{"status", "expected", "actual", "file_count", "missing"}` where `status` is one
    of `"match"`, `"mismatch"`, or `"not_recorded"`.
    Raises: ValueError on a mismatch, or on `not_recorded` when `required=True`.

    `required=False` by default so an existing build that predates this check reports
    `"not_recorded"` rather than failing — the digest is additive provenance, and refusing to load
    every artifact built before it existed would be worse than the gap it closes.
    """
    sidecar = artifact_fingerprint_path(Path(windows_csv))
    payload = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    expected = (payload.get("metadata") or {}).get(AUDIO_INVENTORY_KEY)
    result = window_audio_inventory(windows_csv=windows_csv, data_root=data_root)

    if expected is None:
        outcome = {
            "status": "not_recorded",
            "expected": None,
            "actual": result["sha256"],
            "file_count": result["file_count"],
            "missing": result["missing"],
        }
        if required:
            raise ValueError(
                f"{sidecar} records no {AUDIO_INVENTORY_KEY}. Run "
                "`python -m instrument_robustness.audio_inventory --record` first."
            )
        return outcome

    if expected != result["sha256"] or result["missing"]:
        detail = (
            f"{len(result['missing'])} missing file(s), first {result['missing'][0]}"
            if result["missing"]
            else f"expected {expected[:12]}, got {result['sha256'][:12]}"
        )
        raise ValueError(
            f"Window audio does not match the recorded inventory ({detail}). The CSV and its hash "
            "are unchanged, so this is a change to the AUDIO ITSELF -- every clean feature array and "
            "model result derived from it is suspect."
        )
    return {
        "status": "match",
        "expected": expected,
        "actual": result["sha256"],
        "file_count": result["file_count"],
        "missing": [],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Hash or verify the clean window audio (audit item 16)."
    )
    parser.add_argument("--windows-csv", type=Path, default=WINDOWS_CSV)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--record",
        action="store_true",
        help="compute the inventory and store it in the existing windows.csv sidecar",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="recompute and compare against the recorded inventory (read-only)",
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help="with --verify, fail when no inventory has been recorded",
    )
    args = parser.parse_args()

    if args.record:
        result = record_window_audio_inventory(
            windows_csv=args.windows_csv, data_root=args.data_root
        )
        print(f"hashed {result['file_count']} window files")
        print(f"inventory sha256: {result['sha256']}")
        print(f"recorded in {result['sidecar']}")
    else:
        result = verify_window_audio(
            windows_csv=args.windows_csv,
            data_root=args.data_root,
            required=args.required,
        )
        print(f"status: {result['status']}  ({result['file_count']} files)")
        if result["status"] == "not_recorded":
            print("no inventory recorded yet; run with --record")
        else:
            print(f"inventory sha256: {result['actual']}")


if __name__ == "__main__":
    main()
