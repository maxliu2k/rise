"""Go/no-go probe: can MERT-v1-95M be fine-tuned in this environment at all?

    python scc/mert_ft_smoke.py [--batches 4,8,16,32]

Touches NO project data and writes NO artifacts. It answers two questions that reading the
code cannot, and that must both be YES before any fine-tuning work is worth starting:

  1. Does MERT's `trust_remote_code` implementation BACKPROP cleanly under the pinned
     transformers version? A forward-only path can work while backward raises -- the custom
     modelling code is pinned to a 2023 revision and the frozen probe never exercised backward
     through the backbone even once.
  2. What batch size fits on the GPU with gradients live? The probe trained on cached
     (N, 13, 768) arrays, so peak memory during real fine-tuning is entirely unmeasured.

Exit 0 = both answered, at least one batch size fits. Exit 1 = fine-tuning is blocked here.
"""
from __future__ import annotations

import argparse
import sys

import torch
from torch import nn

sys.path.insert(0, "src")

from instrument_robustness.config import (  # noqa: E402
    MERT_MODEL,
    MERT_REVISION,
    MERT_SR,
    TARGET_LABELS,
    WINDOW_S,
)
from instrument_robustness.mert_data import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS  # noqa: E402
from instrument_robustness.pretrained_extractors import build_mert_model  # noqa: E402


class LiveProbe(nn.Module):
    """MERTProbe's head, but fed from a LIVE backbone pass instead of cached arrays.

    Precondition: `hidden_states` is the tuple transformers returns for
    `output_hidden_states=True`, length MERT_NUM_LAYERS, each (batch, time, MERT_HIDDEN_SIZE).
    Postcondition: returns (batch, len(TARGET_LABELS)) logits.

    This is deliberately identical in shape to the shipped probe head so that a successful
    backward here means the real fine-tune's head is also differentiable end to end.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(MERT_NUM_LAYERS))
        self.classifier = nn.Linear(MERT_HIDDEN_SIZE, num_classes)

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=1)          # (B, 13, T, 768)
        pooled = stacked.mean(dim=2)                          # (B, 13, 768) -- time-mean
        weights = torch.softmax(self.layer_logits, dim=0)
        return self.classifier(torch.sum(pooled * weights[None, :, None], dim=1))


def try_batch(backbone, head, batch: int, device: torch.device) -> tuple[bool, str]:
    """One forward+backward at this batch size. Returns (fitted, message). Never raises OOM."""
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        backbone.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)

        samples = int(MERT_SR * WINDOW_S)
        audio = torch.randn(batch, samples, device=device)
        targets = torch.randint(0, len(TARGET_LABELS), (batch,), device=device)

        out = backbone(audio, output_hidden_states=True)
        states = out.hidden_states
        if len(states) != MERT_NUM_LAYERS:
            return False, f"expected {MERT_NUM_LAYERS} hidden states, got {len(states)}"

        loss = nn.functional.cross_entropy(head(states), targets)
        loss.backward()

        # The whole point: gradients must actually REACH the backbone. A head-only gradient
        # would look like success and silently reproduce the frozen probe.
        named = list(backbone.named_parameters())
        with_grad = [(n, p) for n, p in named if p.grad is not None]
        if not with_grad:
            return False, "NO backbone parameter received a gradient -- backward did not reach it"
        bad = [n for n, p in with_grad if not torch.isfinite(p.grad).all()]
        if bad:
            return False, f"non-finite gradient in {len(bad)} backbone tensors, e.g. {bad[0]}"
        if head.layer_logits.grad is None:
            return False, "layer_logits received no gradient -- layer mixture would not train"

        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        return True, (f"loss {loss.item():.4f} | {len(with_grad)}/{len(named)} backbone tensors "
                      f"have finite grads | peak {peak:.2f} GiB")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, "CUDA OOM"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="4,8,16,32")
    sizes = [int(v) for v in parser.parse_args().batches.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | device {device}")
    import transformers
    print(f"transformers {transformers.__version__}")
    print(f"model {MERT_MODEL} @ {MERT_REVISION[:12]}")
    print(f"input {WINDOW_S}s @ {MERT_SR} Hz\n")

    backbone = build_mert_model(MERT_MODEL, MERT_REVISION)
    # NOT frozen -- this is the entire difference from extract_mert.py.
    backbone.requires_grad_(True).train().to(device)
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"backbone trainable parameters: {trainable / 1e6:.1f} M")

    head = LiveProbe(len(TARGET_LABELS)).to(device)

    fitted = []
    for batch in sizes:
        ok, message = try_batch(backbone, head, batch, device)
        print(f"  batch {batch:>3}: {'PASS' if ok else 'FAIL'}  {message}")
        if ok:
            fitted.append(batch)
        elif message != "CUDA OOM":
            print(f"\nBLOCKED: {message}")
            return 1

    print()
    if not fitted:
        print("BLOCKED: no batch size fits on this GPU.")
        return 1
    print(f"GO: MERT backprops cleanly. Largest fitting batch = {max(fitted)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
