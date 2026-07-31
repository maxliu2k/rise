"""Evaluate the frozen clean-trained AST fine-tune on the shared noise sweep.

    python -m instrument_robustness.noise_eval_ast --model-dir artifacts/ast

Noise is already in the waveform when this runs. AST's own `ASTFeatureExtractor` then does exactly
what it did during training -- resample 22.05 kHz -> 16 kHz, mel-filter, pad/truncate to
(1024, 128), apply its own normalization -- because `pretrained_extractors.ast_input` is the same
function `ASTWindowDataset` calls. The Step-6 statistics are deliberately NOT involved; AST carries
its own.

    noisy WAV -> read_audio_window -> ast_input (16 kHz, extractor) -> (1024,128) -> logits

REQUIRES a contract-shaped `test_summary.json` in the model directory, written by train_ast. The
older `metrics.json` alone is not enough: it records `labels` rather than `label_order` and nests
its counts differently, so `load_official_summary` cannot verify the label order or read the
example count from it, and the clean-parity gate would have nothing to compare against.

ONE SCOPE NOTE. AST's classifier was selected on validation BALANCED ACCURACY while the sweep's
primary metric is macro-F1. The parity gate compares macro-F1 recomputed here against the macro-F1
recorded at training time, so the gate is self-consistent -- but when AST is placed beside SVM and
MERT, the selection asymmetry belongs in the caption.
"""
from __future__ import annotations

import argparse
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

from instrument_robustness.config import ARTIFACTS, TARGET_LABELS
from instrument_robustness.noise_eval_common import (
    load_official_summary,
    run_noise_evaluation,
)
from instrument_robustness.noise_sweep import read_audio_window, sha256_file
from instrument_robustness.pretrained_extractors import ast_input, build_ast_extractor


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_weights(model_dir: Path) -> Path:
    """Locate the saved AST weights, so the parity gate can pin the checkpoint identity."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No AST weights found in {model_dir} (looked for model.safetensors and "
        "pytorch_model.bin). The checkpoint is tracked with Git LFS -- run `git lfs pull` if the "
        "file is present but is a pointer stub."
    )


def load_model(model_dir: Path, device: str):
    """Load the fine-tuned AST classifier and verify its label mapping.

    Raises: ValueError if the checkpoint's id2label disagrees with TARGET_LABELS, which would make
    every predicted index mean something different from what the evaluator assumes.
    """
    from transformers import ASTForAudioClassification

    model = ASTForAudioClassification.from_pretrained(str(model_dir))
    id2label = getattr(model.config, "id2label", None) or {}
    recorded = [id2label[key] for key in sorted(id2label, key=lambda k: int(k))]
    if recorded != list(TARGET_LABELS):
        raise ValueError(
            f"{model_dir} was trained on a different label order:\n"
            f"  checkpoint: {recorded}\n  current:    {list(TARGET_LABELS)}"
        )
    return model.eval().to(device)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    summary_path = args.clean_summary or (model_dir / "test_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {summary_path}. train_ast writes it alongside metrics.json; re-run the AST "
            "training/evaluation on the current data build to produce it."
        )
    weights = find_weights(model_dir)
    summary = load_official_summary(summary_path, expected_model_path=weights)

    device = args.device or get_device()
    model = load_model(model_dir, device)
    extractor = build_ast_extractor()
    print(
        f"ast: device {device}, clean macro-F1 "
        f"{float(summary['test_metrics']['macro_f1']):.6f} over "
        f"{int(summary['test_examples'])} examples"
    )

    @torch.inference_mode()
    def predict_scores(paths: list[Path]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start : start + args.batch_size]
            # ast_input returns (1, 1024, 128) per window; drop that leading axis before stacking,
            # exactly as ASTWindowDataset does, so the batch is (B, 1024, 128).
            inputs = torch.cat(
                [ast_input(read_audio_window(path), extractor) for path in chunk],
                dim=0,
            ).to(device)
            logits = model(input_values=inputs).logits
            batches.append(logits.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    run_noise_evaluation(
        model_name="ast",
        file_prefix="ast_test_",
        predict_scores=predict_scores,
        official_macro_f1=float(summary["test_metrics"]["macro_f1"]),
        official_examples=int(summary["test_examples"]),
        model_sha256=sha256_file(weights),
        score_type="score",
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the clean-trained AST fine-tune on the shared noise sweep."
    )
    parser.add_argument("--model-dir", type=Path, default=ARTIFACTS / "ast")
    parser.add_argument("--clean-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    return args


if __name__ == "__main__":
    main()
