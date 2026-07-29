"""Train the CRNN ensemble. VALIDATION ONLY — the test split is never read here.

    python -m instrument_robustness.train_crnn

Same data, same trainer, same protocol as train_cnn; only the architecture and the output
directory differ. The machinery lives in train_cnn.run_training rather than being copied, so a
bug fixed there is fixed for both.

BEFORE REPORTING ANY NUMBER FROM THIS MODEL, read the tiling note in crnn_model. 97.3% of clips
are tiled and the loop period encodes source note length, which correlates with instrument as a
recording artifact. A GAP-CNN cannot read that period; this can. The check has been run once and
found no evidence of use, but it is underpowered and specific to the data it was run on —
re-run `period_error_probe` whenever this is retrained.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from instrument_robustness.config import ARTIFACTS
from instrument_robustness.crnn_model import MediumCRNN
from instrument_robustness.train_cnn import DEFAULT_SEEDS, run_training

DEFAULT_OUTPUT_DIR = ARTIFACTS / "crnn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the CRNN ensemble (validation only).")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_training(MediumCRNN, args.output_dir, args.seeds, args.device)
    print("next: python -m instrument_robustness.finalize_crnn")


if __name__ == "__main__":
    main()
