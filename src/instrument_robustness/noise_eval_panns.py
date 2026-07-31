"""Evaluate the frozen clean-trained PANNs fine-tune on the shared noise sweep."""
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
from torch import nn

from instrument_robustness.config import (
    ARTIFACTS,
    TARGET_LABELS,
    assert_fingerprint,
)
from instrument_robustness.noise_eval_common import load_official_summary, run_noise_evaluation
from instrument_robustness.noise_sweep import read_audio_window, sha256_file
from instrument_robustness.pretrained_extractors import PANNS_SR, panns_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ARTIFACTS / "panns")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    return args


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PannsClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from panns_inference.models import Cnn14

        self.backbone = Cnn14(
            sample_rate=PANNS_SR,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        self.head = nn.Linear(2048, len(TARGET_LABELS))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(waveforms)["embedding"])


def load_model(path: Path, device: str) -> PannsClassifier:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is an unversioned PANNs checkpoint; a fingerprinted checkpoint is required"
        )
    if checkpoint.get("label_order") != TARGET_LABELS:
        raise ValueError(f"Unexpected PANNs label order in {path}")
    assert_fingerprint(checkpoint.get("config_fingerprint"), str(path))
    from instrument_robustness.train_panns import CKPT, PannsClassifier as TrainingClassifier, build_backbone
    expected_base = checkpoint.get("base_checkpoint_sha256")
    if expected_base is not None and sha256_file(CKPT) != expected_base:
        raise ValueError("Installed PANNs CNN14 base checkpoint differs from the selected model")
    model = TrainingClassifier(build_backbone(), freeze=checkpoint.get("mode") == "probe")
    if checkpoint.get("mode") == "probe":
        model.head.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    return model.eval().to(device)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_dir) / "selected_model.pt"
    result_path = Path(args.model_dir) / "test_summary.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing PANNs model: {model_path}")
    official = load_official_summary(result_path, expected_model_path=model_path)
    official_macro_f1 = float(official["test_metrics"]["macro_f1"])
    official_examples = int(official["test_examples"])
    device = get_device()
    model = load_model(model_path, device)

    @torch.inference_mode()
    def predict_scores(paths: list[Path]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), args.batch_size):
            waveforms = [
                panns_input(read_audio_window(path))
                for path in paths[start : start + args.batch_size]
            ]
            inputs = torch.from_numpy(np.stack(waveforms)).float().to(device)
            batches.append(model(inputs).float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    run_noise_evaluation(
        model_name="panns",
        file_prefix="panns_ft_test_",
        predict_scores=predict_scores,
        official_macro_f1=official_macro_f1,
        official_examples=official_examples,
        model_sha256=sha256_file(model_path),
        score_type="score",
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
