"""Evaluate ONE fine-tuned MERT checkpoint on the sealed test split, exactly once.

    python -m instrument_robustness.finalize_mert_ft \
        --checkpoint /project/rise-grid/$USER/mert_ft_s42/best_finetune.pt \
        --output-dir artifacts/mert_ft/mert_ft_s42

Writes `test_summary.json` and `test_confusion_matrix.csv` next to the checkpoint's records.

WHY THIS IS A SEPARATE SCRIPT FROM TRAINING. train_mert_ft reads train and val only. The test
split is opened here and nowhere else, so "how many times has test been looked at" is answerable
by reading this file's call sites rather than by trusting memory. finalize_mert mirrors this for
the frozen probe.

SELECTION IS ALREADY FROZEN. The configuration (backbone 3e-5, head 5e-3, patience 10) was
chosen on validation across ten runs, all committed under artifacts/mert_ft/. This script does
not choose anything; it scores a checkpoint that was already selected.

PRECONDITIONS
  * The checkpoint was written by train_mert_ft (backbone_frozen is False, fingerprint matches).
  * windows.csv has a test split whose files exist.

POSTCONDITIONS
  * test_summary.json carries every field noise_eval_mert_ft asserts: config_fingerprint,
    label_order, embedding_schema, backbone_frozen=False, test_metrics, test_examples, and the
    checkpoint's sha256 under output_files.model.

RAISES
  * ValueError if the checkpoint is a frozen probe, or its fingerprint/label order disagrees
    with the current config (via load_mert_finetune).
  * FileExistsError if test_summary.json already exists and --overwrite was not passed, so a
    second look at test cannot silently overwrite the first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from instrument_robustness.config import (
    MERT_MODEL,
    MERT_REVISION,
    MERT_SR,
    TARGET_LABELS,
    config_fingerprint,
)
from instrument_robustness.mert_data import load_mert_examples


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.output_dir / "test_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. Test has been scored for this checkpoint once "
            f"already; pass --overwrite only if you intend to replace that record."
        )

    import torch
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    from instrument_robustness.mert_ft_model import load_mert_finetune
    from instrument_robustness.pretrained_extractors import (
        build_mert_processor,
        mert_batch_input,
    )
    from instrument_robustness.featurelib import load_window

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_mert_finetune(args.checkpoint, device=device)
    print(f"loaded {args.checkpoint} (backbone_frozen={checkpoint.get('backbone_frozen')})")

    test = load_mert_examples("test")
    print(f"test windows: {len(test)}", flush=True)
    processor = build_mert_processor(MERT_MODEL, MERT_REVISION)

    started = perf_counter()
    predicted: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(test), args.batch_size):
            block = test[start : start + args.batch_size]
            waveforms = [load_window(example.window_path) for example in block]
            inputs = mert_batch_input(waveforms, processor)["input_values"].float()
            predicted.append(model(inputs.to(device)).argmax(dim=1).cpu().numpy())
    prediction = np.concatenate(predicted)
    truth = np.array([example.target for example in test], dtype=np.int64)
    if len(prediction) != len(truth):
        raise ValueError(f"{len(prediction)} predictions for {len(truth)} test windows")

    matrix = confusion_matrix(truth, prediction, labels=list(range(len(TARGET_LABELS))))
    metrics = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            truth, prediction, labels=list(range(len(TARGET_LABELS))),
            target_names=TARGET_LABELS, output_dict=True, zero_division=0,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=TARGET_LABELS, columns=TARGET_LABELS).to_csv(
        args.output_dir / "test_confusion_matrix.csv"
    )
    # Per-window predictions, so the failure analysis and confusion grid can read this model
    # the same way they read the other six.
    pd.DataFrame({
        "window_path": [e.window_relative_path for e in test],
        "true_label": [TARGET_LABELS[t] for t in truth],
        "predicted_label": [TARGET_LABELS[p] for p in prediction],
    }).to_csv(args.output_dir / "mert_ft_test_clean.csv", index=False)

    summary = {
        "model": "fine-tuned MERT-v1-95M, layer-weighted linear head",
        "backbone_frozen": False,
        "config_fingerprint": config_fingerprint(),
        "label_order": TARGET_LABELS,
        "embedding_schema": {
            "model_id": MERT_MODEL,
            "model_revision": MERT_REVISION,
            "pooling": "time-mean over 13 hidden states, learned layer mixture",
        },
        "sample_rate": MERT_SR,
        "test_examples": int(len(truth)),
        "test_metrics": metrics,
        "checkpoint": str(args.checkpoint),
        "output_files": {
            "model": {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint)},
        },
        "elapsed_seconds": round(perf_counter() - started, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"test accuracy  {metrics['accuracy']:.4f}")
    print(f"test macro-F1  {metrics['macro_f1']:.4f}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
