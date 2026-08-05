"""MERT-v1-95M with a trainable backbone and the probe's layer-mixture head.

This is the fine-tuning counterpart to `mert_probe.MERTProbe`. The head is deliberately
IDENTICAL -- a softmax mixture over the 13 hidden layers, time-mean pooled, then a linear
classifier -- so that a probe-vs-fine-tune comparison isolates the one variable it claims to:
whether the backbone receives gradients.

WHY THIS EXISTS. AST and PANNs report fine-tuned results; MERT reports a frozen probe.
`pretrained_extractors.build_mert_model` records the original plan -- "frozen-feature probe
first ... switch to fine-tuning only if the probe plateaus" -- and the second stage never ran.
Every model-vs-model claim involving MERT is confounded until it does.
"""
from __future__ import annotations

import torch
from torch import nn

from instrument_robustness.config import MERT_MODEL, MERT_REVISION
from instrument_robustness.mert_data import MERT_HIDDEN_SIZE, MERT_NUM_LAYERS
from instrument_robustness.pretrained_extractors import build_mert_model


class MERTFineTune(nn.Module):
    """Trainable MERT backbone + layer-mixture linear head.

    Preconditions: `input_values` is (batch, samples) at MERT_SR, already through
    MERT's Wav2Vec2FeatureExtractor.
    Postcondition: forward returns (batch, num_classes) logits.
    Raises: ValueError if the backbone returns a number of hidden states other than
    MERT_NUM_LAYERS, because the head's layer mixture is sized against that constant and a
    silent mismatch would mix the wrong layers.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        model_id: str = MERT_MODEL,
        revision: str = MERT_REVISION,
        layerdrop: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = build_mert_model(model_id, revision)
        self.backbone.requires_grad_(True)

        # LAYERDROP OFF BY DEFAULT. The go/no-go probe (scc/mert_ft_smoke.py) measured 195 of
        # 211 backbone tensors receiving gradients on one step -- exactly one transformer
        # layer's worth -- because train() mode leaves MERT's pretraining layerdrop active.
        # When a layer is dropped its hidden state passes through unchanged, so the mixture
        # head would see two identical entries competing in one softmax. That is an
        # interaction nobody here can reason about, so remove it rather than model it.
        self.backbone.config.layerdrop = layerdrop

        self.layer_logits = nn.Parameter(torch.zeros(MERT_NUM_LAYERS))
        self.classifier = nn.Linear(MERT_HIDDEN_SIZE, num_classes)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        states = self.backbone(input_values, output_hidden_states=True).hidden_states
        if len(states) != MERT_NUM_LAYERS:
            raise ValueError(
                f"Backbone returned {len(states)} hidden states, expected {MERT_NUM_LAYERS}"
            )
        pooled = torch.stack(states, dim=1).mean(dim=2)      # (batch, 13, 768), time-mean
        weights = torch.softmax(self.layer_logits, dim=0)
        return self.classifier(torch.sum(pooled * weights[None, :, None], dim=1))

    def parameter_groups(self, *, backbone_lr: float, head_lr: float) -> list[dict]:
        """Discriminative learning rates.

        A 95M SSL backbone trained at the head's learning rate does not fine-tune, it is
        destroyed -- the pretrained representation is overwritten in the first few hundred
        steps and the result is a randomly initialised transformer with a good head.
        """
        head = [self.layer_logits, *self.classifier.parameters()]
        return [
            {"params": list(self.backbone.parameters()), "lr": backbone_lr},
            {"params": head, "lr": head_lr},
        ]

    def layer_weights(self) -> list[float]:
        return torch.softmax(self.layer_logits.detach().cpu(), dim=0).tolist()


def load_mert_finetune(path, *, device="cpu"):
    """Load a fine-tuned checkpoint and verify its dataset identity.

    Mirrors `mert_probe.load_mert_probe` field for field. Preconditions: `path` was written by
    train_mert_ft. Postcondition: returns (model in eval mode, checkpoint dict).
    Raises: ValueError on a label-order or class-count mismatch; StaleArtifactError if the
    fingerprint does not match the current config; ValueError if the checkpoint is a FROZEN
    probe, which would otherwise load cleanly here and silently reproduce the probe result
    under a fine-tune's filename.
    """
    from instrument_robustness.config import TARGET_LABELS, assert_fingerprint

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("label_order") != TARGET_LABELS:
        raise ValueError(f"Unexpected MERT label order in {path}")
    if checkpoint.get("num_classes") != len(TARGET_LABELS):
        raise ValueError(f"Unexpected MERT class count in {path}")
    if checkpoint.get("backbone_frozen") is not False:
        raise ValueError(
            f"{path} is not a fine-tuned checkpoint (backbone_frozen="
            f"{checkpoint.get('backbone_frozen')!r}). Loading a frozen probe here would "
            f"reproduce the probe's numbers under the fine-tune's name."
        )
    assert_fingerprint(checkpoint.get("config_fingerprint"), str(path))

    model = MERTFineTune(
        len(TARGET_LABELS),
        model_id=checkpoint.get("model_id", MERT_MODEL),
        revision=checkpoint.get("model_revision", MERT_REVISION),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
