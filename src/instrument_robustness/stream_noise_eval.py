"""Score a model across the whole noise sweep without ever holding the corpus on disk.

    python -m instrument_robustness.stream_noise_eval --model mert_ft \
        --checkpoint /project/rise-grid/$USER/mert_ft_s42/best_finetune.pt \
        --scratch /project/rise-grid/$USER/noise_scratch

WHY. The sweep is 60,240 files (1,255 test windows x 3 noise types x 8 SNRs x 2 replicates) at
258 KB each -- about 15 GB. /projectnb/rise-grid has ~1.2 GB free, and the corpus the six
reported models were scored against was deleted after that run. Without streaming, a seventh
model simply cannot be added to the comparison.

HOW. The corpus is materialized one (noise_type, replicate) chunk at a time -- 2.5 GB, all eight
SNRs of one realization -- scored, then deleted. Chunking on (type, replicate) rather than on SNR
is load-bearing: generate() draws noise ONCE per (window, type, replicate) and rescales that
single draw across every SNR, so splitting on SNR would force a redraw. tests/test_noise.py
asserts a chunked grid is sample-identical to a monolithic one.

WHAT THIS GIVES UP, STATED PLAINLY. A streamed sweep has no completed noise manifest, so the
`n_files == 60240` check cannot run. What replaces it: the dataset build identity is verified
directly, every condition's 1,255 files must exist when that condition is scored, and every
generated file has its achieved SNR checked inside generate(). What remains unverified is that
the regenerated audio matches the audio the other six models saw -- that is what
`--verify-against` exists for, and it should be run once before any result from this script is
compared against theirs.

PRECONDITIONS
  * The dataset fingerprint matches the build the other models were scored on.
  * --scratch has room for one chunk (~2.5 GB) and is NOT the canonical windows_noisy directory.

POSTCONDITIONS
  * output_dir holds one predictions CSV and one metrics JSON per condition, in exactly the
    format run_noise_evaluation writes for every other model, plus noise_sweep_summary.csv.
  * The scratch directory is empty on success.

RAISES
  * FileNotFoundError if a condition's windows are missing when it is scored.
  * ValueError on clean-condition parity failure against the model's official clean result.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from instrument_robustness.config import ARTIFACTS, MERT_MODEL, MERT_REVISION
from instrument_robustness.noise_eval_common import (
    NoiseCondition,
    load_official_summary,
    run_noise_evaluation,
)
from instrument_robustness.noise_sweep import generate, read_audio_window, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mert_ft")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ChunkStreamer:
    """Materialize one (noise_type, replicate) chunk on demand; delete it when the chunk is done.

    Precondition: `scratch` is writable and is not the canonical noisy directory.
    Postcondition: after the last condition of a chunk, that chunk's WAVs are gone.

    The clean condition needs nothing generated -- it reads the ordinary Step-5 windows -- so it
    is a no-op here rather than a special case at the call site.
    """

    def __init__(self, scratch: Path) -> None:
        self.scratch = scratch
        self.current: tuple[str, int] | None = None
        self.generated_chunks = 0

    def ensure(self, condition: NoiseCondition) -> None:
        if condition.noise_type == "clean":
            return
        key = (condition.noise_type, int(condition.replicate or 0))
        if key == self.current:
            return
        self._drop()
        noise_type, replicate = key
        print(f"\n=== generating chunk {noise_type} r{replicate} ===", flush=True)
        generate(
            noisy_dir=self.scratch,
            only_noise_types=(noise_type,),
            only_replicates=(replicate,),
            write_completion=False,
        )
        self.current = key
        self.generated_chunks += 1

    def _drop(self) -> None:
        if self.current is None:
            return
        noise_type, _ = self.current
        directory = self.scratch / noise_type
        if directory.exists():
            shutil.rmtree(directory)
        self.current = None

    def finish(self) -> None:
        self._drop()


def mert_ft_scorer(checkpoint: Path, batch_size: int):
    """Return a predict_scores callable for a fine-tuned MERT checkpoint."""
    import torch

    from instrument_robustness.mert_ft_model import load_mert_finetune
    from instrument_robustness.pretrained_extractors import (
        build_mert_processor,
        mert_batch_input,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_mert_finetune(checkpoint, device=device)
    processor = build_mert_processor(MERT_MODEL, MERT_REVISION)

    def predict_scores(paths: list[Path]) -> np.ndarray:
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                waveforms = [
                    read_audio_window(path) for path in paths[start : start + batch_size]
                ]
                inputs = mert_batch_input(waveforms, processor)["input_values"].float()
                batches.append(model(inputs.to(device)).float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    return predict_scores


def main() -> int:
    args = parse_args()
    summary = load_official_summary(args.clean_summary, expected_model_path=args.checkpoint)
    if summary.get("backbone_frozen") is not False:
        raise ValueError(
            f"{args.clean_summary} is not a fine-tuned result "
            f"(backbone_frozen={summary.get('backbone_frozen')!r})"
        )

    args.scratch.mkdir(parents=True, exist_ok=True)
    streamer = ChunkStreamer(args.scratch)
    try:
        run_noise_evaluation(
            model_name=args.model,
            file_prefix=f"{args.model}_test_",
            predict_scores=mert_ft_scorer(args.checkpoint, args.batch_size),
            official_macro_f1=float(summary["test_metrics"]["macro_f1"]),
            official_examples=int(summary["test_examples"]),
            model_sha256=sha256_file(args.checkpoint),
            score_type="score",
            output_dir=args.output_dir or (ARTIFACTS / args.model / "noise"),
            overwrite=args.overwrite,
            noisy_dir=args.scratch,
            ensure_condition=streamer.ensure,
            release_condition=lambda condition: None,
            require_complete_manifest=False,
        )
    finally:
        streamer.finish()
    print(f"\ngenerated and released {streamer.generated_chunks} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
