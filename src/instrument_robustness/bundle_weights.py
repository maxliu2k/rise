"""Collect every trained model's WEIGHTS into one folder, and prove they have not drifted.

    python -m instrument_robustness.bundle_weights           # (re)build models/
    python -m instrument_robustness.bundle_weights --check   # fail if any copy != its source

Companion to bundle_models.py, which collects source code. This collects the trained artifacts:
the thing you would hand someone who wants to run the six models rather than read them.

WHY THE COPIES CARRY HASHES. A checkpoint is worse to duplicate than source, not better: it is a
large opaque binary, git keeps every version forever, and a copy that silently disagrees with its
original cannot be spotted by reading it. So every copy records the sha256 of the file it came
from, and --check recomputes both sides. Edit nothing here; retrain, then re-run this.

FLAT, WITH SELF-DESCRIBING NAMES. One folder, no subdirectories, and every filename carries its
model. That is not cosmetic: `model_s42.pt` exists under both artifacts/cnn/ and artifacts/crnn/
and would collide outright once the folders are gone, and `model.safetensors` stops meaning
anything the moment it leaves artifacts/ast/. Checkpoints get emailed and dropped into scratch
directories; the name should survive that.

AST IS LFS-TRACKED AND 329 MB. `.gitattributes` matches LFS by exact path, so the renamed
models/ast_finetuned.safetensors needs its own pattern -- without it git stores a 329 MB blob in
every clone forever instead of a pointer. This script refuses to run if that pattern is missing,
because the failure is invisible until someone clones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from instrument_robustness.config import ARTIFACTS, REPO_ROOT

DEST = REPO_ROOT / "models"

# (model, filename under artifacts/<model>/, name in models/)
#
# FLAT, and every name carries its model. Two reasons the originals could not simply be copied
# across: `model_s42.pt` exists under BOTH cnn/ and crnn/ and would collide the moment the
# subfolders go away, and `model.safetensors` says nothing about being AST once it is moved
# anywhere else. A checkpoint gets emailed, dropped in a scratch dir, attached to an issue --
# it should still say what it is.
#
# The suffix names the ROLE, and the two roles are not interchangeable:
#   _selected  fit on TRAIN, the configuration validation chose
#   _final     refit on TRAIN+VAL, the model the one permitted test evaluation used
# For the seed ensembles the seed is the role: every seed is equal, none is "best".
WEIGHTS: list[tuple[str, str, str]] = [
    ("ast",   "model.safetensors",              "ast_finetuned.safetensors"),
    *[("cnn",  f"model_s{s}.pt", f"cnn_seed{s}.pt")  for s in (42, 43, 44, 45, 46)],
    *[("crnn", f"model_s{s}.pt", f"crnn_seed{s}.pt") for s in (42, 43, 44, 45, 46)],
    ("mert",  "best_probe.pt",                  "mert_probe_selected.pt"),
    ("mert",  "final_probe.pt",                 "mert_probe_final.pt"),
    ("panns", "panns_probe_philharmonia.pt",    "panns_probe_philharmonia.pt"),
    ("panns", "panns_probe_tinysol.pt",         "panns_probe_tinysol.pt"),
    ("svm",   "best_model.joblib",              "svm_selected.joblib"),
    ("svm",   "final_model.joblib",             "svm_final.joblib"),
]

# Paths that must be declared LFS before they are written, or git commits a raw blob.
LFS_REQUIRED = ("models/ast_finetuned.safetensors",)

README = """\
GENERATED -- DO NOT EDIT. Trained weights for all six models, one folder per model.

Rebuild after retraining:   python -m instrument_robustness.bundle_weights
Verify nothing has drifted: python -m instrument_robustness.bundle_weights --check

MANIFEST.json records the sha256 of each source artifact. --check recomputes both the source and
the copy, so an edited copy, a retrained source, or a missing file all fail rather than pass
quietly.

These are COPIES. artifacts/<model>/ remains where finalize_* and noise_eval_* read from, and
where the metrics, confusion matrices and status files live. Nothing here is loaded by the code.

ast/model.safetensors is Git LFS (329 MB). Clone with git-lfs installed or you get a pointer file
that will fail to load as a model.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    """True if this is a 130-byte LFS stub rather than the real artifact.

    Copying a pointer produces a file that looks like a model, is named like a model, and fails
    at load time with something unrelated-sounding. Worth one open() to rule out.
    """
    with path.open("rb") as handle:
        return handle.read(40).startswith(b"version https://git-lfs.github.com/spec")


def assert_lfs_declared() -> None:
    for rel in LFS_REQUIRED:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "check-attr", "filter", "--", rel],
                             capture_output=True, text=True)
        if "filter: lfs" not in out.stdout:
            raise SystemExit(
                f"ERROR: {rel} is not declared as LFS in .gitattributes.\n"
                f"  Committing it would put a large binary in every clone, permanently.\n"
                f"  Add:  {rel} filter=lfs diff=lfs merge=lfs -text")


def plan() -> list[tuple[str, str, str]]:
    return list(WEIGHTS)


def build() -> int:
    assert_lfs_declared()

    missing = [f"{m}/{src}" for m, src, _ in plan() if not (ARTIFACTS / m / src).exists()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} artifact(s) missing: {missing}\n"
                         f"  Train the model or fix the map in {Path(__file__).name}; do not "
                         f"skip, or the folder silently omits a model.")
    pointers = [f"{m}/{src}" for m, src, _ in plan() if is_lfs_pointer(ARTIFACTS / m / src)]
    if pointers:
        raise SystemExit(f"ERROR: these are LFS POINTERS, not real files: {pointers}\n"
                         f"  Run `git lfs pull` first. Copying a pointer produces a broken model "
                         f"that fails only at load time.")

    if DEST.exists():
        shutil.rmtree(DEST)
    entries = {}
    DEST.mkdir(parents=True, exist_ok=True)
    for model, src_name, dest_name in plan():
        source = ARTIFACTS / model / src_name
        shutil.copy2(source, DEST / dest_name)
        entries[dest_name] = {
            "model": model,
            "source": f"artifacts/{model}/{src_name}",
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        }

    (DEST / "README.txt").write_text(README, encoding="utf-8")
    (DEST / "MANIFEST.json").write_text(json.dumps({
        "generated_by": "instrument_robustness.bundle_weights",
        "models": sorted({m for m, _, _ in WEIGHTS}),
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries.values()),
        "files": dict(sorted(entries.items())),
    }, indent=2) + "\n", encoding="utf-8")

    for name, rec in sorted(entries.items()):
        print(f"  {name:<34} {rec['bytes'] / 1048576:8.2f} MB   <- {rec['source']}")
    print(f"\nwrote {DEST}  ({len(entries)} files, "
          f"{sum(e['bytes'] for e in entries.values()) / 1048576:.1f} MB)")
    return 0


def check() -> int:
    manifest_path = DEST / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"FAIL: no bundle at {DEST}. Run without --check to build it.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems = []
    for rel, rec in manifest["files"].items():
        source, copy = REPO_ROOT / rec["source"], DEST / rel
        if not source.exists():
            problems.append(f"{rel}: source {rec['source']} no longer exists")
        elif not copy.exists():
            problems.append(f"{rel}: copy missing")
        elif sha256(source) != rec["sha256"]:
            problems.append(f"{rel}: SOURCE CHANGED (retrained?) since the bundle was built")
        elif sha256(copy) != rec["sha256"]:
            problems.append(f"{rel}: copy differs from its source")
    for _, _, dest_name in plan():
        if dest_name not in manifest["files"]:
            problems.append(f"{dest_name}: in the map but not the manifest (rebuild)")

    if problems:
        print(f"FAIL: {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nRebuild: python -m instrument_robustness.bundle_weights", file=sys.stderr)
        return 1
    print(f"weights OK: {len(manifest['files'])} files match their sources "
          f"({manifest['total_bytes'] / 1048576:.1f} MB)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the weights match artifacts/; write nothing")
    raise SystemExit(check() if parser.parse_args().check else build())


if __name__ == "__main__":
    main()
