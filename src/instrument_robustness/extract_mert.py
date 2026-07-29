from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from instrument_robustness.config import (
    MERT_MODEL,
    MERT_REVISION,
    PIPE,
    ROOT,
    TARGET_LABELS,
    config_fingerprint_json,
)
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
    parser.add_argument("--revision", default=MERT_REVISION)
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


def extract_mert_splits(
    *,
    splits,
    data_root: Path,
    windows_csv: Path,
    output_dir: Path,
    batch_size: int,
    model_id: str,
    revision: str | None,
    device: str,
    allow_test: bool = False,
) -> dict[str, Path]:
    """Extract frozen MERT embeddings, keeping test access internal to finalization."""
    requested_splits = tuple(splits)
    invalid = set(requested_splits) - {"train", "val", "test"}
    if invalid:
        raise ValueError(f"Unknown MERT splits: {sorted(invalid)}")
    if "test" in requested_splits and not allow_test:
        raise ValueError(
            "Test extraction is sealed. Run instrument_robustness.finalize_mert "
            "after validation selection is frozen."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    output_dir = Path(output_dir)
    existing = [
        output_dir / f"{split}.npz"
        for split in requested_splits
        if (output_dir / f"{split}.npz").exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite MERT embeddings: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "MERT extraction requires the pretrained dependencies: "
            "pip install -e '.[mert]'"
        ) from error

    target_device = choose_device(device, torch)
    processor = build_mert_processor(model_id, revision)
    model = build_mert_model(model_id, revision)
    model.requires_grad_(False)
    model.eval().to(target_device)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    if not resolved_revision:
        raise ValueError(
            "MERT did not expose an immutable checkpoint revision. "
            "Pass --revision with a Hugging Face commit hash."
        )

    written = {}
    for split in requested_splits:
        output_path = output_dir / f"{split}.npz"
        examples = load_mert_examples(
            split,
            windows_csv=windows_csv,
            data_root=data_root,
        )
        batches: list[np.ndarray] = []
        started = perf_counter()

        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            waveforms = [load_window(example.window_path) for example in batch_examples]
            processed = mert_batch_input(waveforms, processor)
            model_inputs = {
                name: value.to(target_device)
                for name, value in processed.items()
            }

            with torch.inference_mode():
                output = model(**model_inputs, output_hidden_states=True)
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
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            with temporary_path.open("wb") as file:
                np.savez(
                    file,
                    X=X,
                    y=y,
                    window_path=np.asarray(
                        [example.window_relative_path for example in examples]
                    ),
                    source_path=np.asarray(
                        [example.source_path for example in examples]
                    ),
                    label_names=np.asarray(TARGET_LABELS),
                    model_id=np.asarray(model_id),
                    model_revision=np.asarray(resolved_revision),
                    pooling=np.asarray("mean_over_time_per_hidden_layer"),
                    config_fingerprint=np.asarray(config_fingerprint_json()),
                )
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        elapsed = perf_counter() - started
        print(f"Saved {output_path}: X={X.shape}, seconds={elapsed:.1f}")
        written[split] = output_path
    return written


def main() -> None:
    args = parse_args()
    extract_mert_splits(
        splits=args.splits,
        data_root=args.data_root,
        windows_csv=args.windows_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
    )


if __name__ == "__main__":
    main()
