from __future__ import annotations

import torch
from torch import nn

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
