"""Evaluate the frozen final MERT probe on the shared noise sweep."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from instrument_robustness.config import ARTIFACTS
from instrument_robustness.extract_mert import (
    choose_device,
    extract_mert_batch,
)
from instrument_robustness.noise_eval_common import (
    load_official_summary,
    run_noise_evaluation,
)
from instrument_robustness.noise_sweep import read_audio_window, sha256_file
from instrument_robustness.pretrained_extractors import (
    build_mert_model,
    build_mert_processor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ARTIFACTS / "mert" / "final_probe.pt",
    )
    parser.add_argument(
        "--clean-summary",
        type=Path,
        default=ARTIFACTS / "mert" / "test_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    summary = load_official_summary(
        args.clean_summary,
        expected_model_path=args.model,
    )
    if summary.get("backbone_frozen") is not True:
        raise ValueError("Official MERT result is not a frozen-backbone probe")
    embedding_schema = summary.get("embedding_schema", {})
    model_id = embedding_schema.get("model_id")
    revision = embedding_schema.get("model_revision")
    if not model_id or not revision:
        raise ValueError("Official MERT summary lacks an immutable embedding schema")

    try:
        import torch
        from instrument_robustness.mert_probe import load_mert_probe
    except ImportError as error:
        raise RuntimeError(
            "MERT noise evaluation requires: pip install -e '.[mert]'"
        ) from error

    device = choose_device(args.device, torch)
    probe, checkpoint = load_mert_probe(args.model, device=device)
    if checkpoint.get("embedding_schema") != embedding_schema:
        raise ValueError("MERT probe and clean summary use different embeddings")
    processor = build_mert_processor(model_id, revision)
    backbone = build_mert_model(model_id, revision)
    resolved_revision = getattr(backbone.config, "_commit_hash", None) or revision
    if resolved_revision != revision:
        raise ValueError(
            f"MERT resolved revision {resolved_revision!r}; expected {revision!r}"
        )
    backbone.requires_grad_(False)
    backbone.eval().to(device)

    def predict_scores(paths: list[Path]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(paths), args.batch_size):
            waveforms = [
                read_audio_window(path)
                for path in paths[start : start + args.batch_size]
            ]
            embeddings = extract_mert_batch(
                waveforms,
                processor=processor,
                model=backbone,
                target_device=device,
                torch=torch,
            )
            with torch.inference_mode():
                logits = probe(
                    torch.from_numpy(embeddings).float().to(device)
                )
            batches.append(logits.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    run_noise_evaluation(
        model_name="mert",
        file_prefix="mert_test_",
        predict_scores=predict_scores,
        official_macro_f1=float(summary["test_metrics"]["macro_f1"]),
        official_examples=int(summary["test_examples"]),
        model_sha256=sha256_file(args.model),
        score_type="score",
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
