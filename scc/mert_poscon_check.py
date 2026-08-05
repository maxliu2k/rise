"""Is MERT's positional convolution actually loaded from the checkpoint?

    python scc/mert_poscon_check.py

Reads no project data and writes no artifacts.

WHY THIS EXISTS. Loading MERT in the current venv prints:

    not used when initializing MERTModel: ['encoder.pos_conv_embed.conv.weight_g',
                                           'encoder.pos_conv_embed.conv.weight_v']
    newly initialized: ['encoder.pos_conv_embed.conv.parametrizations.weight.original0',
                        'encoder.pos_conv_embed.conv.parametrizations.weight.original1']

That is the torch weight-norm rename: `torch.nn.utils.weight_norm` stored `weight_g`/`weight_v`,
`torch.nn.utils.parametrizations.weight_norm` stores `parametrizations.weight.original0/1`. If
transformers does not translate between them, the checkpoint's positional convolution is
DISCARDED and replaced with a fresh random tensor.

Two consequences, in increasing severity, and this script distinguishes them:

  1. If the random draw is IDENTICAL on every load, every MERT result is self-consistent but
     computed with a randomly initialized positional convolution -- a handicap applied equally
     to training and evaluation.
  2. If the draw DIFFERS between loads, the cached train/validation embeddings and the live
     noise-evaluation backbone used DIFFERENT positional convolutions. The probe would then be
     evaluated on features it was never trained against, and every MERT number in the study is
     invalid rather than merely handicapped.

Exit 0 = positional conv loaded from checkpoint (no problem).
Exit 2 = randomly initialized but stable across loads (case 1).
Exit 3 = randomly initialized AND unstable across loads (case 2).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from instrument_robustness.config import MERT_MODEL, MERT_REVISION  # noqa: E402
from instrument_robustness.pretrained_extractors import build_mert_model  # noqa: E402


def pos_conv_weight(model) -> torch.Tensor:
    """The effective positional-convolution weight, however it is parametrized.

    Raises: AttributeError if the module layout is not what this check assumes, because a
    silently skipped check is worse than a crash.
    """
    conv = model.encoder.pos_conv_embed.conv
    return conv.weight.detach().float().cpu().clone()


def main() -> int:
    print(f"torch {torch.__version__}")
    import transformers
    print(f"transformers {transformers.__version__}")
    print(f"model {MERT_MODEL} @ {MERT_REVISION[:12]}\n")

    # What does the checkpoint actually contain?
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    try:
        path = hf_hub_download(MERT_MODEL, "pytorch_model.bin", revision=MERT_REVISION)
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        path = hf_hub_download(MERT_MODEL, "model.safetensors", revision=MERT_REVISION)
        state = load_file(path)
    keys = [k for k in state if "pos_conv" in k]
    print("checkpoint pos_conv keys:", keys)

    first = pos_conv_weight(build_mert_model(MERT_MODEL, MERT_REVISION))
    second = pos_conv_weight(build_mert_model(MERT_MODEL, MERT_REVISION))
    stable = torch.equal(first, second)
    print(f"\nweight shape {tuple(first.shape)}  norm {first.norm().item():.4f}")
    print(f"identical across two independent loads: {stable}")

    # Reconstruct what the checkpoint's weight-norm pair WOULD produce, and compare. If the
    # loaded weight matches, transformers translated the names and nothing is wrong.
    g = next((state[k] for k in keys if k.endswith("weight_g")), None)
    v = next((state[k] for k in keys if k.endswith("weight_v")), None)
    if g is not None and v is not None:
        expected = (g * v / v.norm(dim=(0, 1), keepdim=True)).float()
        loaded = torch.allclose(first, expected, atol=1e-5)
        print(f"matches checkpoint weight_norm(g, v): {loaded}")
        if loaded:
            print("\nOK: positional convolution IS the checkpoint's.")
            return 0
    else:
        print("checkpoint has no weight_g/weight_v pair -- cannot reconstruct")

    if not stable:
        print("\nCRITICAL: positional conv is RANDOM and DIFFERS between loads.")
        print("Cached embeddings and the live noise-eval backbone did not share a backbone.")
        return 3
    print("\nWARNING: positional conv is randomly initialized, but identical on every load.")
    print("MERT results are self-consistent but computed with an unlearned positional conv.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
