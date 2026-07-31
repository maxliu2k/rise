"""The single permitted test evaluation for the CRNN ensemble.

    python -m instrument_robustness.finalize_crnn

Same protocol and same gates as finalize_cnn; only the architecture and output directory differ.
The machinery lives in finalize_cnn.run_finalize rather than being copied.

Whatever number this produces, report it beside the tiling check (`period_error_probe`) and the
two structural confounds recorded in crnn_model: this model has 2.6x MediumCNN's parameters and
twice its temporal resolution at the readout, so a CRNN advantage is not attributable to the
recurrence alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from instrument_robustness.config import ARTIFACTS
from instrument_robustness.crnn_model import MediumCRNN
from instrument_robustness.finalize_cnn import run_finalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One permitted CRNN test evaluation.")
    p.add_argument("--output-dir", type=Path, default=ARTIFACTS / "crnn")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_finalize(MediumCRNN, args.output_dir, args.device)


if __name__ == "__main__":
    main()
