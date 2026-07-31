"""Evaluate the frozen clean-trained CNN seed ensemble on the shared noise sweep.

    python -m instrument_robustness.noise_eval_cnn

THE ONE THING THIS MODULE MUST NOT DO is read `features/cnn/{split}.npz`. Those arrays are the
CLEAN log-mel, standardized and frozen at Step 7. Scoring them would report the clean number under
a noisy label. The noisy waveform has to travel the whole representation path again:

    noisy WAV -> featurelib.logmel -> Step-6 per-mel-bin train stats -> (N,1,128,130)

`featurelib.logmel` is the same function Step 6 and Step 7 call, which is what guarantees the noisy
features are produced by identical code to the clean ones. The statistics are LOADED from
norm_stats.npz, never recomputed: they are part of the trained model's contract, and refitting them
on noisy audio would silently rescale the very distortion being measured.

ENSEMBLE SCORING. train_cnn selects a combiner on validation and finalize_cnn records it. The
shared runner wants an (N, n_classes) score array and takes its argmax, so:
  * soft_vote  -> return the mean per-seed probability. argmax(mean) IS soft_vote.
  * hard_vote  -> return a finite lexicographic encoding of (vote count, summed probability),
                  whose argmax reproduces hard_vote exactly. Its own -inf construction cannot be
                  returned, because run_noise_evaluation rejects non-finite scores.
Both are asserted against the real combiner at runtime rather than trusted.

`noise_eval_crnn` imports `run_cnn_noise_evaluation` from here with a different architecture, so a
bug fixed here is fixed for both.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from instrument_robustness.cnn_model import MediumCNN, hard_vote, predict_probs, soft_vote
from instrument_robustness.config import ARTIFACTS, STATS_NPZ
from instrument_robustness.ensemble_scores import combiner_scores
from instrument_robustness.logmel_input import (
    cnn_batch_from_waveforms,
    load_logmel_statistics,
)
from instrument_robustness.noise_eval_common import (
    load_official_summary,
    run_noise_evaluation,
)
from instrument_robustness.noise_sweep import read_audio_window, sha256_file

COMBINERS = {"soft_vote": soft_vote, "hard_vote": hard_vote}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ensemble(output_dir: Path, model_cls, seeds: list[int], device: str) -> list:
    """Load one fingerprint-checked checkpoint per seed, under the recorded architecture."""
    from instrument_robustness.finalize_cnn import load_checkpoint

    models = []
    for seed in seeds:
        path = Path(output_dir) / f"model_s{seed}.pt"
        model = load_checkpoint(path, model_cls.__name__, device)
        model.eval()
        models.append(model)
    if not models:
        raise ValueError(f"No checkpoints found under {output_dir}")
    return models


def run_cnn_noise_evaluation(
    model_cls,
    *,
    model_name: str,
    file_prefix: str,
    clean_dir: Path,
    stats_path: Path,
    batch_size: int,
    device: str,
    output_dir: Path | None,
    overwrite: bool,
):
    """Score `model_cls`'s clean-trained seed ensemble across every noise condition."""
    clean_dir = Path(clean_dir)
    summary_path = clean_dir / "test_summary.json"
    summary = load_official_summary(summary_path)

    architecture = summary.get("architecture")
    if architecture != model_cls.__name__:
        raise ValueError(
            f"{summary_path} records architecture {architecture!r}, "
            f"expected {model_cls.__name__!r}"
        )
    combiner_name = summary["combiner"]
    seeds = list(summary["seeds"])
    official_macro_f1 = float(summary["test_metrics"]["macro_f1"])
    official_examples = int(summary["test_examples"])

    mean, std = load_logmel_statistics(stats_path)
    models = load_ensemble(clean_dir, model_cls, seeds, device)
    print(
        f"{model_name}: seeds {seeds}, combiner {combiner_name}, device {device}, "
        f"clean macro-F1 {official_macro_f1:.6f} over {official_examples} examples"
    )

    # One checkpoint hash per seed, ordered, so the recorded model identity covers the whole
    # ensemble rather than an arbitrary member of it.
    model_sha256 = ",".join(
        sha256_file(clean_dir / f"model_s{seed}.pt") for seed in seeds
    )

    @torch.inference_mode()
    def predict_scores(paths: list[Path]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            features = cnn_batch_from_waveforms(
                [read_audio_window(path) for path in chunk], mean, std
            )
            per_seed = np.stack(
                [predict_probs(model, features, device, batch_size) for model in models]
            )
            batches.append(combiner_scores(per_seed, combiner_name))
        return np.concatenate(batches, axis=0)

    return run_noise_evaluation(
        model_name=model_name,
        file_prefix=file_prefix,
        predict_scores=predict_scores,
        official_macro_f1=official_macro_f1,
        official_examples=official_examples,
        model_sha256=model_sha256,
        score_type="score",
        output_dir=output_dir,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the clean-trained CNN ensemble on the shared noise sweep."
    )
    parser.add_argument("--clean-dir", type=Path, default=ARTIFACTS / "cnn")
    parser.add_argument("--stats", type=Path, default=STATS_NPZ)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    run_cnn_noise_evaluation(
        MediumCNN,
        model_name="cnn",
        file_prefix="cnn_test_",
        clean_dir=args.clean_dir,
        stats_path=args.stats,
        batch_size=args.batch_size,
        device=args.device or get_device(),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
