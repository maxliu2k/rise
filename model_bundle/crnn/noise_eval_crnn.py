"""Evaluate the frozen clean-trained CRNN seed ensemble on the shared noise sweep.

    python -m instrument_robustness.noise_eval_crnn

Same data, same representation path, same protocol as noise_eval_cnn; only the architecture and
the directories differ. The machinery lives in `noise_eval_cnn.run_cnn_noise_evaluation` rather
than being copied, so a bug fixed there is fixed for both.

Note the input layout: MediumCRNN takes (B, 1, n_mels, n_frames) exactly as MediumCNN does and
collapses frequency internally, so this reuses the CNN feature path unchanged. It does NOT use
`crnn_data.load_crnn`'s (N, time, features) layout -- neither train_crnn nor finalize_crnn does
either.

WHEN REPORTING ANY NUMBER FROM THIS MODEL, carry the tiling caveat from crnn_model with it. An
order-sensitive readout can reach the loop period that a GAP-CNN cannot, and under noise that
shortcut may degrade differently from timbre -- which is precisely what this sweep measures. The
`period_error_probe` check found no evidence of use on clean audio, but it was never run on noisy
audio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from instrument_robustness.config import ARTIFACTS, STATS_NPZ
from instrument_robustness.crnn_model import MediumCRNN
from instrument_robustness.noise_eval_cnn import get_device, run_cnn_noise_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the clean-trained CRNN ensemble on the shared noise sweep."
    )
    parser.add_argument("--clean-dir", type=Path, default=ARTIFACTS / "crnn")
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
        MediumCRNN,
        model_name="crnn",
        file_prefix="crnn_test_",
        clean_dir=args.clean_dir,
        stats_path=args.stats,
        batch_size=args.batch_size,
        device=args.device or get_device(),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
