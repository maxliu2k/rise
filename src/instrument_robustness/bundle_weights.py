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

THE TWO 300 MB CHECKPOINTS ARE POINTERS, NOT FILES. AST (329 MB) and the PANNs fine-tune
(312 MB) live in EXTERNAL_WEIGHTS rather than being copied here. Git LFS was the obvious
alternative and is the wrong tool: LFS storage is not reclaimable, because deleting the file
leaves the object referenced by history. Two superseded AST checkpoints already cost the
repository owner ~658 MB permanently. A pointer also carries `dataset_fingerprint`, which is what
lets build() refuse a checkpoint from a different dataset build -- LFS would hand it over
silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from instrument_robustness.config import ARTIFACTS, DATASET_FREEZE, REPO_ROOT

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
    # AST is NOT here any more. At 329 MB it was the single largest thing in the repo, and Git LFS
    # storage is not reclaimable: deleting the file leaves the object referenced by history, so the
    # two superseded AST checkpoints already cost ~658 MB of the owner's LFS quota permanently.
    # Adding a third per retrain is not sustainable when the dataset fingerprint changes. It is an
    # EXTERNAL_WEIGHTS pointer instead -- which also gains the dataset_fingerprint check that LFS
    # cannot do.
    *[("cnn",  f"model_s{s}.pt", f"cnn_seed{s}.pt")  for s in (42, 43, 44, 45, 46)],
    *[("crnn", f"model_s{s}.pt", f"crnn_seed{s}.pt") for s in (42, 43, 44, 45, 46)],
    ("mert",  "best_probe.pt",                  "mert_probe_selected.pt"),
    ("mert",  "final_probe.pt",                 "mert_probe_final.pt"),
    ("svm",   "best_model.joblib",              "svm_selected.joblib"),
    ("svm",   "final_model.joblib",             "svm_final.joblib"),
]

# The reported PANNs result used this full fine-tune, not the much smaller probe formerly copied
# into models/. It is deliberately distributed as a release asset to avoid committing another
# 312 MB checkpoint. A pointer is honest; substituting the probe is not.
HISTORICAL_EXTERNAL_WEIGHTS = {
    "ast_finetuned_philharmonia_8378.safetensors": {
        "model": "ast",
        "role": "reported Philharmonia clean model (historical 8,378-source build; validation-balanced-accuracy selection and unsealed test; not valid for the corrected frozen build)",
        "sha256": "25789685e1cb0a4df0d64e5c84df84f49eff72c2831cc8e89fb02bd7676763e7",
        "bytes": 344820808,
        "windows_csv_sha256": "cfb725e320b703364dc3f5f3b8d98782c93c594630d89e6a5e147ac33b63e8ab",
        "git_commit": "baa5970",
        "git_path": "artifacts/new-ast-results-20260730-022036/model.safetensors",
    },
    "panns_finetune_philharmonia_8378.pt": {
        "model": "panns",
        "role": "reported Philharmonia clean and noise model (historical 8,378-source build; not valid for the corrected frozen build)",
        "sha256": "00cc195e1cbea756fc0afcb1ab823d639e31668c1a859f67941c29fda40741e3",
        "release_tag": "v1.0-panns-12class",
        "dataset_fingerprint": "89f126e290d0a9674e4e0a2b6344dcced32fa42ebe4e872006918e044f723073",
        "download_url": "https://github.com/maxliu2k/rise/releases/download/v1.0-panns-12class/panns_finetune_philharmonia.pt",
    },
}

EXTERNAL_WEIGHTS = {
    "ast_finetuned.safetensors": {
        "model": "ast",
        "role": "reported clean model for the corrected frozen build; test macro-F1 0.9908 on 1255 examples",
        "sha256": "a37bd70d51356ae32b24fd09b57e22572847c94c920a0db1ac7c3b369f1cc6b1",
        "bytes": 344820808,
        "dataset_fingerprint": "97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf",
        "scc_path": "/projectnb/rise-grid/models/97b1cdd2/ast_finetuned.safetensors",
        "download_url": None,   # pending a release upload; see scc_path meanwhile
    },
    "panns_finetune.pt": {
        "model": "panns",
        "role": "reported clean model for the corrected frozen build (--mode finetune); test macro-F1 0.9868 on 1255 examples",
        "sha256": "5b102c8aaa91071391ff257f2ce978b624910019780c6fc91a3e51159d702143",
        "bytes": 327544421,
        "dataset_fingerprint": "97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf",
        "scc_path": "/projectnb/rise-grid/models/97b1cdd2/panns_finetune.pt",
        "download_url": None,   # pending a release upload; see scc_path meanwhile
    },
}

# Paths that must be declared LFS before they are written, or git commits a raw blob.
# Empty on purpose: every weight now copied into models/ is a few MB, and the two 300 MB
# checkpoints are EXTERNAL_WEIGHTS pointers rather than LFS objects. See the WEIGHTS comment.
LFS_REQUIRED: tuple[str, ...] = ()

README = """\
GENERATED -- DO NOT EDIT. Trained weights for all six models, flat in one folder.

Rebuild after retraining:   python -m instrument_robustness.bundle_weights
Verify nothing has drifted: python -m instrument_robustness.bundle_weights --check

  cnn_seed{42..46}.pt           CNN ensemble, 5 seeds
  crnn_seed{42..46}.pt          CRNN ensemble, 5 seeds
  svm_selected.joblib           fit on TRAIN, config chosen on validation
  svm_final.joblib              refit on TRAIN+VAL, used for the test evaluation
  mert_probe_selected.pt        fit on TRAIN, chosen on validation
  mert_probe_final.pt           refit on TRAIN+VAL, used for the test evaluation
  (AST and PANNs fine-tune are NOT here -- see external_files in MANIFEST.json for their
   sha256, byte count, dataset_fingerprint and SCC path)

_selected and _final are NOT interchangeable. _selected is what validation chose; _final saw the
validation split during fitting, so scoring it on validation is meaningless. For the seed
ensembles the seed IS the role -- all five are equal and none is "best".

MANIFEST.json records the sha256, byte count and originating artifacts/ path of every file.
--check recomputes both sides, so an edited copy, a retrained source, or a missing file fails
rather than passing quietly.

These are COPIES. artifacts/<model>/ remains where finalize_* and noise_eval_* read from, and
where the metrics, confusion matrices and status files live. Nothing here is loaded by the code.

The two 300 MB checkpoints (AST, PANNs fine-tune) are NOT here. They are EXTERNAL_WEIGHTS
entries in bundle_weights.py, each carrying sha256, byte count and dataset_fingerprint, and are
readable on SCC at /projectnb/rise-grid/models/<fingerprint>/. Verify against SHA256SUMS there.

The PANNs fine-tune is NOT copied into this folder. MANIFEST.json records its exact filename,
SHA-256, release URL and scientific role. Download it explicitly and verify the hash. The included
probe was removed because it cannot reproduce the reported PANNs result.
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

    if not DATASET_FREEZE.is_file():
        raise SystemExit(f"ERROR: no frozen dataset record at {DATASET_FREEZE}")
    frozen = json.loads(DATASET_FREEZE.read_text(encoding="utf-8"))["dataset_fingerprint"]
    stale_external = [
        name for name, record in EXTERNAL_WEIGHTS.items()
        if record.get("dataset_fingerprint") != frozen
    ]
    if stale_external:
        raise SystemExit(
            "ERROR: external weight pointer(s) belong to a different dataset build: "
            f"{stale_external}. Publish the corrected checkpoint and update its exact pointer; "
            "do not substitute a historical model."
        )

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
        "models": sorted({m for m, _, _ in WEIGHTS} | {r["model"] for r in EXTERNAL_WEIGHTS.values()}),
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries.values()),
        "files": dict(sorted(entries.items())),
        "external_files": EXTERNAL_WEIGHTS,
        "historical_external_files": HISTORICAL_EXTERNAL_WEIGHTS,
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
    if manifest.get("external_files") != EXTERNAL_WEIGHTS:
        problems.append("external checkpoint pointers differ from the declared release identities")
    if manifest.get("historical_external_files") != HISTORICAL_EXTERNAL_WEIGHTS:
        problems.append("historical checkpoint pointers differ from their recorded identities")

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
