"""Generate docs/REPO_MAP.md from the repository's actual state.

    python -m instrument_robustness.gen_repo_map            # print
    python -m instrument_robustness.gen_repo_map --write     # write docs/REPO_MAP.md
    python -m instrument_robustness.gen_repo_map --check     # exit 1 if the file is out of date

The map stays honest because it is REGENERATED from the tree, not hand-maintained: every source
module and top-level entry is listed by scanning the filesystem, and anything without a curated
description is shown as "(undocumented)" rather than omitted -- so a new file is visibly missing a
description instead of silently absent. Branches are read from git.

Install as a pre-commit hook so it refreshes on every commit:

    git config core.hooksPath .githooks
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from instrument_robustness.config import REPO_ROOT

# --- Curated descriptions. Keys are repo-relative paths (top level) or bare module basenames. ---
TOP_LEVEL = {
    "src/": "The instrument_robustness package: pipeline, models, noise, and tooling.",
    "all-samples/": "Philharmonia DATA ROOT (RISE_DATA_ROOT default). pipeline/ is tracked; work/, features/, checkpoints/ are gitignored.",
    "tinysol/": "TinySOL small tracked artifact mirror (pipeline CSVs only). Real TinySOL audio lives under RISE_TINYSOL_ROOT.",
    "artifacts/": "Committed model checkpoints, clean results, and noise results. One subdir per model, plus <model>/noise/.",
    "model_bundle/": "All six models' training/eval code copied into one folder for hand-off; drift from src/ is a build failure (bundle_models.py).",
    "models/": "All six trained weight files in one flat folder with self-describing names (assembled by bundle_weights.py).",
    "scc/": "BU SCC (SGE) qsub job scripts and their README.",
    "docs/": "Design and protocol docs, including the frozen noise and failure-analysis plans.",
    "tests/": "Unit tests: preprocessing, models, noise, robustness curves, and failure analysis.",
    "legacy/": "Retired code and 9-class-era artifacts, kept for provenance. Not on the active path.",
    "configs/": "Configuration inputs consumed by the pipeline.",
    "data/": "Small tracked data inputs.",
    "pyproject.toml": "Package + dependency definition; extras like [pretrained] pull torch/transformers/panns.",
    "download_data.py": "RETIRED Google-Drive fetch. prep_data.py is the only supported dataset acquisition.",
    "README.md": "Project overview and how to run the pipeline.",
    "internal/": "Working notes, planning docs, and internal review/audit records. Not part of the "
                 "published method; kept for provenance.",
}

# Module descriptions, grouped. A module not listed here still appears under its group as
# "(undocumented)". Order within a group is meaningful only for the pipeline.
GROUPS = {
    "Pipeline (run in this order via run_pipeline.py)": {
        "prep_data": "Acquire the 12 Philharmonia archives and write manifest.csv (the only supported acquisition).",
        "step0_filter": "Keep the 12 target labels and one plain articulation per class; drop defects.",
        "step1_resample": "Resample to 22050 Hz mono, killing the MP3-bitrate confound.",
        "step2_trim": "Trim leading/trailing silence at 30 dB; keep interior silence.",
        "step3_split": "Split 70/15/15 by (label, note) pitch group -- no group spans splits.",
        "step4_window": "One 3.0 s window per source (MAX_WINDOWS_PER_SOURCE=1); short notes tiled, not padded.",
        "step5_normalize": "Loudness-normalize each window to 0.1 RMS with a peak guard.",
        "step6_stats": "Compute TRAIN-ONLY feature/log-mel normalization statistics.",
        "step7_featurize": "Materialize SVM/CNN feature arrays under the trained stats.",
        "run_pipeline": "One-command runner for the nine stages; sets thread limits, stops at first failure.",
    },
    "Configuration & shared libraries": {
        "config.py": "Single source of truth: paths, labels, window/split params, the SNR grid, config_fingerprint().",
        "featurelib": "Window loading and the 88 handcrafted SVM features.",
        "pretrained_extractors": "Input adapters for PANNs (32 kHz) / AST / MERT.",
        "logmel_input": "Log-mel front end for CNN/CRNN.",
        "audio_inventory": "Dataset inventory / cross-checks.",
    },
    "Model definitions & data adapters": {
        "svm_model": "SVM architecture / pipeline definition.",
        "cnn_model": "From-scratch CNN.",
        "crnn_model": "From-scratch CRNN.",
        "mert_probe": "MERT linear probe head.",
        "cnn_data": "CNN input pipeline.",
        "crnn_data": "CRNN input pipeline.",
        "mert_data": "MERT embedding extraction inputs.",
        "ast_data": "AST label resolution and window validation.",
    },
    "Training & finalization": {
        "train_svm": "Train/select the SVM.",
        "train_cnn": "Train the CNN ensemble (5 seeds), resumable.",
        "train_crnn": "Train the CRNN ensemble (5 seeds).",
        "train_mert": "Train the MERT probe.",
        "train_ast": "Fine-tune AST.",
        "train_panns": "PANNs CNN14 probe or fine-tune.",
        "finalize_svm": "Refit SVM on train only and evaluate test once.",
        "finalize_cnn": "Finalize CNN and evaluate test once.",
        "finalize_crnn": "Finalize CRNN and evaluate test once.",
        "finalize_mert": "Refit MERT on train only, evaluate test once, write a guard record.",
        "extract_mert": "Extract MERT embeddings to .npz.",
        "eval_panns_probe": "Evaluate the PANNs linear probe.",
    },
    "Noise sweep & robustness": {
        "noise_sweep": "Generate + fingerprint the shared noisy audio (--validate/--generate/--check-generated).",
        "noise_eval_common": "Shared fail-closed evaluation contract: clean-parity gate, cluster columns, output schema.",
        "noise_eval_svm": "SVM adapter for the noise sweep.",
        "noise_eval_cnn": "CNN adapter for the noise sweep.",
        "noise_eval_crnn": "CRNN adapter for the noise sweep.",
        "noise_eval_mert": "MERT adapter for the noise sweep.",
        "noise_eval_ast": "AST adapter for the noise sweep.",
        "noise_eval_panns": "PANNs adapter for the noise sweep.",
        "noise_metrics": "Per-mixture band/octave/active/effective-SNR diagnostics.",
        "noise_stats": "Paired cluster bootstrap + exact cluster sign test.",
        "snr_pilot": "Validation-only SNR grid selection; writes no audio. NOT a benchmark.",
        "robustness_curve": "Retention-vs-SNR curve and normalized AUC.",
        "failure_analysis": "Post-rerun instrument recall-loss and acoustic-distance/confusion analyses.",
    },
    "Cross-dataset, ensemble & tooling": {
        "build_tinysol_manifest": "Emit a Philharmonia-schema manifest from TinySOL audio (tinysol-12class branch).",
        "cross_dataset_eval": "Score each model on the OTHER dataset's audio (tinysol-12class branch).",
        "ensemble_scores": "Combine per-seed model scores.",
        "bundle_models": "Assemble model_bundle/; fails on drift from src/.",
        "bundle_weights": "Assemble models/ (the flat trained-weights folder).",
        "summarize_results": "Consolidate every model's clean+noise results, verified against config_fingerprint().",
        "gen_repo_map": "Generate this map from the tree.",
    },
}

# Branches worth naming; everything else from git is listed as "other".
BRANCH_NOTES = {
    "main": "Canonical Philharmonia branch. Presentation + release target.",
    "tinysol-12class": "TinySOL work: adds build_tinysol_manifest + cross_dataset_eval; kept merged up with main.",
}


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _module_files():
    pkg = REPO_ROOT / "src" / "instrument_robustness"
    return sorted(p.stem for p in pkg.glob("*.py") if p.name != "__init__.py")


def render():
    lines = ["# Repository map", ""]
    lines.append("_Auto-generated by `python -m instrument_robustness.gen_repo_map`. "
                 "Do not edit by hand; edit the registry in that module instead._")
    lines.append("")

    # Branches
    lines += ["## Branches", ""]
    local = [b.strip("* ").strip() for b in _git("branch", "--format=%(refname:short)").splitlines() if b.strip()]
    for b in local:
        note = BRANCH_NOTES.get(b, "working / historical branch")
        lines.append(f"- **{b}** — {note}")
    lines.append("")

    # Top-level layout
    lines += ["## Top-level layout", ""]
    present = sorted(p.name + ("/" if p.is_dir() else "") for p in REPO_ROOT.iterdir()
                     if not p.name.startswith(".") and p.name not in {"__pycache__"})
    for name in present:
        desc = TOP_LEVEL.get(name, "")
        lines.append(f"- `{name}`{' — ' + desc if desc else ' — (undocumented)'}")
    lines.append("")

    # Source modules, grouped, with orphan detection
    described = {m for group in GROUPS.values() for m in group}
    actual = set(_module_files())
    lines += ["## Source modules (`src/instrument_robustness/`)", ""]
    for title, mods in GROUPS.items():
        lines.append(f"### {title}")
        for mod, desc in mods.items():
            here = "" if mod.replace(".py", "") in actual or mod in actual else "  _(not on this branch)_"
            lines.append(f"- `{mod}` — {desc}{here}")
        lines.append("")
    undocumented = sorted(actual - {m.replace(".py", "") for m in described})
    if undocumented:
        lines += ["### Undocumented modules (add to gen_repo_map registry)", ""]
        lines += [f"- `{m}` — (undocumented)" for m in undocumented]
        lines.append("")

    # Artifacts snapshot
    art = REPO_ROOT / "artifacts"
    if art.is_dir():
        lines += ["## Artifacts present", ""]
        for sub in sorted(p for p in art.iterdir() if p.is_dir()):
            noise = (sub / "noise")
            tag = "  (+ noise sweep)" if noise.is_dir() and any(noise.iterdir()) else ""
            lines.append(f"- `artifacts/{sub.name}/`{tag}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/REPO_MAP.md is stale")
    args = ap.parse_args()
    text = render()
    out = REPO_ROOT / "docs" / "REPO_MAP.md"
    if args.check:
        current = out.read_text() if out.is_file() else ""
        if current != text:
            print("docs/REPO_MAP.md is out of date; run: python -m instrument_robustness.gen_repo_map --write")
            sys.exit(1)
        print("docs/REPO_MAP.md is up to date")
        return
    if args.write:
        out.write_text(text)
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
