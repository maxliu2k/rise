"""Assemble one browsable folder holding all six models' code, and prove it has not drifted.

    python -m instrument_robustness.bundle_models           # (re)build model_bundle/
    python -m instrument_robustness.bundle_models --check   # fail if any copy != its source

WHY COPIES ARE DANGEROUS AND WHY THESE ARE NOT.

The package layout is flat and per-model by naming convention (train_cnn.py, finalize_cnn.py,
noise_eval_cnn.py, ...), which is fine for running things and poor for answering "show me
everything the CRNN is". A folder-per-model answers that. Physically MOVING the files would
rewrite every import in the package, so this copies instead.

Copies of source code are normally a bad idea: two editable copies drift, and nothing says which
one is real. That failure already happened in this repo -- train_cnn.py defined its own
MAX_IMBALANCE at the same value config.py already held, so the two were silently independent and
changing config would have given the CNN and MERT different class weighting.

So this bundle is GENERATED, NEVER EDITED, and every copy records the sha256 of the file it came
from. `--check` recomputes those hashes and exits non-zero on the first mismatch, which makes
drift a build failure rather than a thing someone notices later. Edit the source under
src/instrument_robustness/ and re-run this; do not edit anything under model_bundle/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from instrument_robustness.config import REPO_ROOT

SRC = REPO_ROOT / "src" / "instrument_robustness"
BUNDLE = REPO_ROOT / "model_bundle"

# Every model's own files. Shared code is deliberately NOT duplicated per model -- see SHARED.
MODELS: dict[str, list[str]] = {
    "svm":   ["svm_model.py", "train_svm.py", "finalize_svm.py", "noise_eval_svm.py"],
    "cnn":   ["cnn_model.py", "cnn_data.py", "train_cnn.py", "finalize_cnn.py",
              "noise_eval_cnn.py"],
    "crnn":  ["crnn_model.py", "crnn_data.py", "train_crnn.py", "finalize_crnn.py",
              "noise_eval_crnn.py"],
    "ast":   ["ast_data.py", "train_ast.py", "noise_eval_ast.py"],
    "mert":  ["mert_data.py", "mert_probe.py", "extract_mert.py", "train_mert.py",
              "finalize_mert.py", "noise_eval_mert.py"],
    "panns": ["train_panns.py", "eval_panns_probe.py", "noise_eval_panns.py"],
}

# Used by more than one model. Copied ONCE, not into each model folder: duplicating featurelib per
# model would suggest each has its own, and "one featurelib for clean and noisy alike" is the
# property the noise comparison depends on.
SHARED: dict[str, list[str]] = {
    "_shared": ["config.py", "featurelib.py", "logmel_input.py", "pretrained_extractors.py"],
    "_pipeline": ["prep_data.py", "run_pipeline.py", "step0_filter.py", "step1_resample.py",
                  "step2_trim.py", "step3_split.py", "step4_window.py", "step5_normalize.py",
                  "step6_stats.py", "step7_featurize.py", "audio_inventory.py"],
    "_noise": ["noise_sweep.py", "noise_eval_common.py", "noise_metrics.py", "noise_stats.py",
               "robustness_curve.py", "snr_pilot.py", "ensemble_scores.py"],
}

README = """\
GENERATED -- DO NOT EDIT ANYTHING IN THIS FOLDER.

Every file here is a copy of one under src/instrument_robustness/. Edit the original and re-run:

    python -m instrument_robustness.bundle_models

MANIFEST.json records the sha256 of each source file at bundle time. To prove nothing has drifted:

    python -m instrument_robustness.bundle_models --check

That exits non-zero if any copy disagrees with its source, so a stale bundle fails rather than
quietly misleading whoever reads it.

The code here is NOT importable as a package and is not on sys.path. It exists so a reader can see
everything belonging to one model in one place. Run models from the real package.

  <model>/     the six models: svm, cnn, crnn, ast, mert, panns
  _shared/     config, featurelib and the pretrained extractors, used by more than one model
  _pipeline/   prep_data, run_pipeline and steps 0-7, which build the features every model reads
  _noise/      the shared noise sweep, metrics and evaluation contract
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """The commit the bundle was built from, or 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def plan() -> list[tuple[str, str]]:
    """(folder, filename) for every file the bundle contains, models first."""
    return [(folder, name)
            for group in (MODELS, SHARED)
            for folder, names in group.items()
            for name in names]


def build() -> int:
    missing = [n for _, n in plan() if not (SRC / n).exists()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} source file(s) do not exist: {missing}\n"
                         f"  The bundle map in {__file__} is stale -- fix it rather than "
                         f"skipping files, or the bundle silently omits part of a model.")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)                 # rebuild from scratch: a rename would orphan a copy
    entries = {}
    for folder, name in plan():
        dest = BUNDLE / folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / name, dest)
        entries[f"{folder}/{name}"] = {"source": f"src/instrument_robustness/{name}",
                                       "sha256": sha256(SRC / name)}

    (BUNDLE / "README.txt").write_text(README, encoding="utf-8")
    (BUNDLE / "MANIFEST.json").write_text(json.dumps({
        "generated_by": "instrument_robustness.bundle_models",
        "built_from_commit": git_commit(),
        "models": sorted(MODELS),
        "n_files": len(entries),
        "files": dict(sorted(entries.items())),
    }, indent=2) + "\n", encoding="utf-8")

    for folder in sorted(MODELS):
        print(f"  {folder:<8} {len(MODELS[folder])} files")
    for folder in sorted(SHARED):
        print(f"  {folder:<8} {len(SHARED[folder])} files")
    print(f"\nwrote {BUNDLE} ({len(entries)} files) from commit {git_commit()[:8]}")
    return 0


def check() -> int:
    """Exit non-zero unless every bundled copy matches the source it was made from."""
    manifest_path = BUNDLE / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"FAIL: no bundle at {BUNDLE}. Run without --check to build it.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems = []
    for rel, rec in manifest["files"].items():
        source = REPO_ROOT / rec["source"]
        copy = BUNDLE / rel
        if not source.exists():
            problems.append(f"{rel}: source {rec['source']} no longer exists")
        elif not copy.exists():
            problems.append(f"{rel}: bundled copy missing")
        elif sha256(source) != rec["sha256"]:
            problems.append(f"{rel}: SOURCE CHANGED since the bundle was built")
        elif sha256(copy) != rec["sha256"]:
            problems.append(f"{rel}: bundled copy was EDITED -- edit the source instead")

    # A file added to the package but never added to MODELS/SHARED is invisible here, so check the
    # map too: a model quietly missing a file is exactly what this bundle would otherwise hide.
    for folder, name in plan():
        if f"{folder}/{name}" not in manifest["files"]:
            problems.append(f"{folder}/{name}: in the bundle map but not the manifest (rebuild)")

    if problems:
        print(f"FAIL: bundle is stale, {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nRebuild: python -m instrument_robustness.bundle_models", file=sys.stderr)
        return 1
    print(f"bundle OK: {len(manifest['files'])} files match their sources "
          f"(built from {manifest['built_from_commit'][:8]})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the bundle matches its sources; do not write anything")
    raise SystemExit(check() if parser.parse_args().check else build())


if __name__ == "__main__":
    main()
