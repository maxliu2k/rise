from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from instrument_robustness.config import MERT_MODEL, PIPE, ROOT, TARGET_LABELS
from instrument_robustness.featurelib import load_window
from instrument_robustness.mert_data import (
    MERT_FEATURE_DIR,
    MERT_HIDDEN_SIZE,
    MERT_NUM_LAYERS,
    load_mert_examples,
)
from instrument_robustness.pretrained_extractors import (
    build_mert_model,
    build_mert_processor,
    mert_batch_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen MERT-v1-95M train/validation representations. "
            "This command never reads the test split."
        )
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--windows-csv", type=Path, default=PIPE / "windows.csv")
    parser.add_argument("--output-dir", type=Path, default=MERT_FEATURE_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-id", default=MERT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    return args


def choose_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def main() -> None:
    args = parse_args()

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "MERT extraction requires the pretrained dependencies: "
            "pip install -e '.[mert]'"
        ) from error

    device = choose_device(args.device, torch)
    processor = build_mert_processor(args.model_id, args.revision)
    model = build_mert_model(args.model_id, args.revision)
    model.requires_grad_(False)
    model.eval().to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = getattr(model.config, "_commit_hash", None)

    for split in args.splits:
        examples = load_mert_examples(
            split,
            windows_csv=args.windows_csv,
            data_root=args.data_root,
        )
        batches: list[np.ndarray] = []
        started = perf_counter()

        for start in range(0, len(examples), args.batch_size):
            batch_examples = examples[start : start + args.batch_size]
            waveforms = [load_window(example.window_path) for example in batch_examples]
            processed = mert_batch_input(waveforms, processor)
            input_values = processed["input_values"].to(device)

            with torch.inference_mode():
                output = model(input_values=input_values, output_hidden_states=True)
                hidden_states = output.hidden_states
                if len(hidden_states) != MERT_NUM_LAYERS:
                    raise ValueError(
                        f"Expected {MERT_NUM_LAYERS} hidden states, "
                        f"received {len(hidden_states)}"
                    )
                pooled = torch.stack(
                    [hidden.mean(dim=1) for hidden in hidden_states],
                    dim=1,
                )
                if pooled.shape[2] != MERT_HIDDEN_SIZE:
                    raise ValueError(
                        f"Expected hidden size {MERT_HIDDEN_SIZE}, "
                        f"received {pooled.shape[2]}"
                    )
                batches.append(pooled.float().cpu().numpy())

            completed = min(start + len(batch_examples), len(examples))
            print(f"[{split}] {completed}/{len(examples)}", flush=True)

        X = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        y = np.asarray([example.target for example in examples], dtype=np.int64)
        output_path = output_dir / f"{split}.npz"
        np.savez(
            output_path,
            X=X,
            y=y,
            window_path=np.asarray(
                [example.window_relative_path for example in examples]
            ),
            source_path=np.asarray([example.source_path for example in examples]),
            label_names=np.asarray(TARGET_LABELS),
            model_id=np.asarray(args.model_id),
            model_revision=np.asarray(resolved_revision or args.revision or "main"),
            pooling=np.asarray("mean_over_time_per_hidden_layer"),
        )
        elapsed = perf_counter() - started
        print(f"Saved {output_path}: X={X.shape}, seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
