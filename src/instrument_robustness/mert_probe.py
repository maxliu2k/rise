from __future__ import annotations

import torch
from torch import nn

from instrument_robustness.config import TARGET_LABELS, assert_fingerprint
from instrument_robustness.mert_data import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS


class MERTProbe(nn.Module):
    """Learn a layer mixture and linear classifier over frozen MERT embeddings."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(MERT_NUM_LAYERS))
        self.classifier = nn.Linear(MERT_HIDDEN_SIZE, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1:] != (
            MERT_NUM_LAYERS,
            MERT_HIDDEN_SIZE,
        ):
            raise ValueError(
                "Expected embeddings shaped "
                f"(batch, {MERT_NUM_LAYERS}, {MERT_HIDDEN_SIZE}), "
                f"received {tuple(embeddings.shape)}"
            )
        weights = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.sum(embeddings * weights[None, :, None], dim=1)
        return self.classifier(mixed)

    def layer_weights(self) -> list[float]:
        return torch.softmax(self.layer_logits.detach().cpu(), dim=0).tolist()


def load_mert_probe(path, *, device="cpu"):
    """Load a saved validation or final probe and verify its dataset identity."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("label_order") != TARGET_LABELS:
        raise ValueError(f"Unexpected MERT label order in {path}")
    if checkpoint.get("num_classes") != len(TARGET_LABELS):
        raise ValueError(f"Unexpected MERT class count in {path}")
    assert_fingerprint(checkpoint.get("config_fingerprint"), str(path))
    model = MERTProbe(len(TARGET_LABELS))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
