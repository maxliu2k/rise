"""Fine-tune MERT-v1-95M on the train split, selecting on validation. Never reads test.

    python -m instrument_robustness.train_mert_ft --output-dir /project/rise-grid/$USER/mert_ft

Writes `best_finetune.pt` and `validation_summary.json` to --output-dir. Test evaluation is a
separate one-time finalizer, exactly as for the frozen probe.

WHY. MERT is the only pretrained model in this study evaluated as a frozen probe; AST and PANNs
are fine-tuned. The comparison is confounded until MERT is adapted the same way. See
`mert_ft_model.MERTFineTune`.

PRECONDITIONS
  * Step 5 has produced windows.csv with train and val splits (asserted by load_mert_examples).
  * The MERT fine-tuning path has been shown to backprop in this environment
    (scc/mert_ft_smoke.py returns GO).

POSTCONDITIONS
  * best_finetune.pt holds the state dict of the epoch with the highest validation macro-F1,
    plus the config fingerprint and label order.
  * validation_summary.json records every epoch's validation metrics, so a plateau is visible
    rather than inferred.

RAISES
  * FileNotFoundError if any window file named by windows.csv is absent.
  * ValueError on a label outside TARGET_LABELS, or a fingerprint mismatch.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from instrument_robustness.config import (
    MERT_MODEL,
    MERT_REVISION,
    MERT_SR,
    TARGET_LABELS,
    config_fingerprint,
)
from instrument_robustness.featurelib import load_window
from instrument_robustness.mert_data import MERTExample, load_mert_examples

REPO_ROOT = Path(__file__).resolve().parents[2]
# Default OUTSIDE the repo and on /project rather than /projectnb: the checkpoint is ~380 MB,
# /projectnb/rise-grid sits at 97% of quota, and /project is the backed-up space.
DEFAULT_OUTPUT_DIR = Path("/project/rise-grid") / "mert_ft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="debug only: use at most N examples per split (0 = all)",
    )
    return parser.parse_args()


def seed_everything(seed: int, torch) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def processed_inputs(examples: list[MERTExample], processor, *, chunk: int = 256):
    """Decode every window and run MERT's feature extractor once, up front.

    Preconditions: every example's window_path exists (load_mert_examples asserts this).
    Postcondition: returns (inputs, targets) as float32 (N, samples) and int64 (N,).

    Decoding is done ONCE rather than per epoch. At 3 s and 24 kHz this is ~290 MB per 1,000
    windows, which fits comfortably, and it keeps the 22050 -> 24000 resample identical to the
    evaluation path by reusing the same `mert_batch_input`.
    """
    import torch
    from instrument_robustness.pretrained_extractors import mert_batch_input

    pieces: list[torch.Tensor] = []
    for start in range(0, len(examples), chunk):
        block = examples[start : start + chunk]
        waveforms = [load_window(example.window_path) for example in block]
        pieces.append(mert_batch_input(waveforms, processor)["input_values"].float())
    inputs = torch.cat(pieces, dim=0)
    targets = torch.tensor([example.target for example in examples], dtype=torch.long)
    if len(inputs) != len(targets):
        raise ValueError(f"{len(inputs)} inputs but {len(targets)} targets")
    return inputs, targets


def evaluate(model, inputs, targets, *, batch_size: int, device, torch) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    predicted: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            predicted.append(model(batch).argmax(dim=1).cpu().numpy())
    prediction = np.concatenate(predicted)
    truth = targets.numpy()
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def main() -> int:
    args = parse_args()
    import torch
    from torch import nn

    from instrument_robustness.mert_ft_model import MERTFineTune
    from instrument_robustness.pretrained_extractors import build_mert_processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "Refusing to fine-tune a 95M backbone on CPU -- submit this through "
            "scc/mert_finetune.qsub, which requests a GPU."
        )
    seed_everything(args.seed, torch)

    train = load_mert_examples("train")
    val = load_mert_examples("val")
    if args.limit:
        train, val = train[: args.limit], val[: args.limit]
    print(f"train {len(train)} windows | val {len(val)} windows", flush=True)

    processor = build_mert_processor(MERT_MODEL, MERT_REVISION)
    started = perf_counter()
    train_x, train_y = processed_inputs(train, processor)
    val_x, val_y = processed_inputs(val, processor)
    print(f"decoded+processed in {perf_counter() - started:.1f}s "
          f"| train {tuple(train_x.shape)} val {tuple(val_x.shape)}", flush=True)

    model = MERTFineTune(len(TARGET_LABELS)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(backbone_lr=args.backbone_lr, head_lr=args.head_lr),
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)

    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": 0}
    best_state = None
    stale = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        running, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[index].to(device)), train_y[index].to(device))
            loss.backward()
            # SSL backbones diverge without this; wav2vec2/HuBERT recipes all clip.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            batches += 1

        metrics = evaluate(model, val_x, val_y, batch_size=args.batch_size,
                           device=device, torch=torch)
        metrics.update(epoch=epoch, train_loss=running / max(batches, 1))
        history.append(metrics)
        print(f"epoch {epoch:>2}  loss {metrics['train_loss']:.4f}  "
              f"val macro-F1 {metrics['macro_f1']:.4f}  acc {metrics['accuracy']:.4f}",
              flush=True)

        if metrics["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": metrics["macro_f1"], "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: {stale} epochs without improvement", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_finetune.pt"
    torch.save(
        {
            "state_dict": best_state,
            "label_order": TARGET_LABELS,
            "num_classes": len(TARGET_LABELS),
            "config_fingerprint": config_fingerprint(),
            "model_id": MERT_MODEL,
            "model_revision": MERT_REVISION,
            "backbone_frozen": False,
        },
        checkpoint,
    )
    summary = {
        "model": "fine-tuned MERT-v1-95M, layer-weighted linear head",
        "backbone_frozen": False,
        "config_fingerprint": config_fingerprint(),
        "label_order": TARGET_LABELS,
        "sample_rate": MERT_SR,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["macro_f1"],
        "history": history,
        "hyperparameters": {
            "backbone_lr": args.backbone_lr, "head_lr": args.head_lr,
            "batch_size": args.batch_size, "max_epochs": args.max_epochs,
            "patience": args.patience, "seed": args.seed,
            "grad_clip": 1.0, "weight_decay": 0.01, "layerdrop": 0.0,
        },
        "checkpoint": str(checkpoint),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    (args.output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbest epoch {best['epoch']} | val macro-F1 {best['macro_f1']:.4f}")
    print(f"wrote {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
