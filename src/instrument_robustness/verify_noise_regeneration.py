"""Does a regenerated noise chunk reproduce the audio the reported models were scored on?

    python -m instrument_robustness.verify_noise_regeneration \
        --scratch /project/rise-grid/$USER/noise_scratch

Regenerates ONE (noise_type, replicate) chunk, scores it with the FROZEN MERT probe -- whose
per-window predictions for that chunk are committed under artifacts/mert/noise/ -- and compares
label for label. Exit 0 only if every condition matches exactly.

WHY THIS AND NOT A FILE HASH. The obvious check would be to compare regenerated WAVs against the
`output_sha256` column of the original provenance. That cannot work: libsndfile stamps the PEAK
chunk of a float WAV with the write time, so two files with identical samples differ at one byte
(offset 60) -- see tests/test_noise.py. The hash moves even when the audio does not, and the
original provenance is gone with the corpus anyway. Model predictions are the only comparison
that survives, and they are already committed.

WHAT A MISMATCH WOULD MEAN. Not that the existing results are wrong -- those six models were all
scored against one corpus in one run and are internally paired regardless. It would mean the
corpus cannot be rebuilt, so a seventh model cannot be ADDED to that comparison without
regenerating and re-running all seven. Candidate causes, in rough order of likelihood:
librosa.resample's default res_type changing between versions (noise_sweep does not pin it), the
shared ESC-50/DEMAND corpora under /projectnb/rise-grid/noise-sources having changed, or a
config change since the sweep.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_robustness.config import (
    ARTIFACTS,
    MERT_MODEL,
    MERT_REVISION,
    SNRS,
    TARGET_LABELS,
)
from instrument_robustness.noise_eval_common import load_test_frame
from instrument_robustness.noise_sweep import (
    dataset_build_identity,
    generate,
    out_path,
    read_audio_window,
)

SEALED_FINGERPRINT = "97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--noise-type", default="audience")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--probe", type=Path, default=ARTIFACTS / "mert" / "final_probe.pt")
    parser.add_argument(
        "--committed-dir", type=Path, default=ARTIFACTS / "mert" / "noise"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--keep", action="store_true", help="do not delete the chunk")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    identity = dataset_build_identity()
    fingerprint = identity["dataset_fingerprint"]
    print(f"dataset fingerprint: {fingerprint}")
    if fingerprint != SEALED_FINGERPRINT:
        print(
            f"FAIL: build drifted from the sealed corpus ({SEALED_FINGERPRINT[:8]}...). "
            "Regenerated audio cannot match by construction.",
            file=sys.stderr,
        )
        return 1
    print("fingerprint matches the sealed build\n")

    import torch

    from instrument_robustness.extract_mert import choose_device, extract_mert_batch
    from instrument_robustness.mert_probe import load_mert_probe
    from instrument_robustness.pretrained_extractors import (
        build_mert_model,
        build_mert_processor,
    )

    device = choose_device("auto", torch)
    probe, _ = load_mert_probe(args.probe, device=device)
    processor = build_mert_processor(MERT_MODEL, MERT_REVISION)
    backbone = build_mert_model(MERT_MODEL, MERT_REVISION)
    backbone.requires_grad_(False).eval().to(device)

    frame = load_test_frame()
    args.scratch.mkdir(parents=True, exist_ok=True)
    print(f"regenerating chunk {args.noise_type} r{args.replicate} ...", flush=True)
    generate(
        noisy_dir=args.scratch,
        only_noise_types=(args.noise_type,),
        only_replicates=(args.replicate,),
        write_completion=False,
    )

    failures = 0
    try:
        for snr in SNRS:
            tag = f"{args.noise_type}_{snr}_r{args.replicate}"
            committed_path = args.committed_dir / f"mert_test_{tag}.csv"
            if not committed_path.is_file():
                print(f"{tag:<22} SKIP (no committed predictions at {committed_path})")
                continue
            committed = pd.read_csv(committed_path)

            paths = [
                out_path(
                    args.noise_type,
                    snr,
                    window_id,
                    replicate=args.replicate,
                    noisy_dir=args.scratch,
                )
                for window_id in frame["window_id"]
            ]
            predicted: list[np.ndarray] = []
            with torch.inference_mode():
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
                    logits = probe(torch.from_numpy(embeddings).float().to(device))
                    predicted.append(logits.argmax(dim=1).cpu().numpy())
            labels = [TARGET_LABELS[i] for i in np.concatenate(predicted)]

            if len(labels) != len(committed):
                print(f"{tag:<22} FAIL  {len(labels)} predictions vs {len(committed)} committed")
                failures += 1
                continue
            agree = int((np.asarray(labels) == committed["predicted_label"].to_numpy()).sum())
            status = "MATCH" if agree == len(labels) else "FAIL "
            print(f"{tag:<22} {status} {agree}/{len(labels)} labels identical", flush=True)
            if agree != len(labels):
                failures += 1
    finally:
        if not args.keep:
            shutil.rmtree(args.scratch / args.noise_type, ignore_errors=True)

    print()
    if failures:
        print(f"REGENERATION IS NOT FAITHFUL: {failures} condition(s) differ.", file=sys.stderr)
        print(
            "Existing six-model results are unaffected -- they were scored against one corpus in "
            "one run. But a seventh model cannot be added to that comparison from a rebuild.",
            file=sys.stderr,
        )
        return 1
    print("REGENERATION VERIFIED: every committed prediction reproduced exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
