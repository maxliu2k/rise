"""Express a seed ensemble's decision as a finite score matrix.

`noise_eval_common.run_noise_evaluation` wants (N, n_classes) scores, takes their argmax, and
REJECTS any non-finite value. The CNN/CRNN combiners in cnn_model return class indices instead, and
`hard_vote` builds its answer with an -inf mask that cannot be handed back. This converts each
combiner into scores that decide identically and are finite throughout.

Torch-free on purpose: the encoding is pure arithmetic, and keeping it out of noise_eval_cnn means
it can be unit-tested without the optional pretrained extras -- on the machine that actually trains
the CNN, the self-check below additionally pins it against the real combiners.
"""
from __future__ import annotations

import numpy as np

COMBINER_NAMES = ("soft_vote", "hard_vote")


def combiner_scores(per_seed_probs: np.ndarray, combiner_name: str) -> np.ndarray:
    """A finite score matrix whose argmax equals the named combiner's decision.

    Preconditions: `per_seed_probs` is (n_seeds, N, n_classes) with each row summing to 1.
    Postcondition: returns (N, n_classes) float64, all finite, with
    `scores.argmax(axis=1) == cnn_model.<combiner_name>(per_seed_probs)` elementwise.
    Raises: ValueError on an unknown combiner, a wrong shape, a non-finite result, or -- when torch
    is importable, so the reference combiners can be loaded -- any disagreement with them.

    HOW hard_vote IS ENCODED. Its rule is "most votes wins, ties broken by summed probability".
    Each class's summed probability is at most n_seeds, so

        score = vote_count * (n_seeds + 1) + summed_probability

    makes any higher-count class outrank any lower-count one regardless of confidence, while
    leaving summed probability to order classes that tie on count. Same ordering as the -inf mask,
    without the -inf.
    """
    if combiner_name not in COMBINER_NAMES:
        raise ValueError(
            f"Unknown combiner {combiner_name!r}; expected one of {sorted(COMBINER_NAMES)}"
        )
    probs = np.asarray(per_seed_probs)
    if probs.ndim != 3:
        raise ValueError(f"expected (n_seeds, N, n_classes), got {probs.shape}")
    n_seeds, n_rows, _ = probs.shape
    if n_seeds == 0 or n_rows == 0:
        raise ValueError(f"empty ensemble or batch: {probs.shape}")

    if combiner_name == "soft_vote":
        scores = probs.mean(axis=0).astype(np.float64)
    else:
        votes = probs.argmax(axis=2)
        counts = np.zeros((n_rows, probs.shape[2]), dtype=np.float64)
        rows = np.arange(n_rows)
        for member in range(n_seeds):
            counts[rows, votes[member]] += 1.0
        scores = counts * (n_seeds + 1.0) + probs.sum(axis=0).astype(np.float64)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Combiner scores contain non-finite values")
    _assert_matches_reference(probs, scores, combiner_name)
    return scores


def _assert_matches_reference(
    probs: np.ndarray,
    scores: np.ndarray,
    combiner_name: str,
) -> None:
    """Cross-check against cnn_model's combiners when they can be imported.

    Skipped silently only when torch is absent -- in that case nothing in this repository can run
    a CNN anyway, so there is no wrong number to guard against.
    """
    try:
        from instrument_robustness.cnn_model import hard_vote, soft_vote
    except Exception:  # torch missing; the CNN cannot run at all here
        return
    reference = {"soft_vote": soft_vote, "hard_vote": hard_vote}[combiner_name](probs)
    if not np.array_equal(scores.argmax(axis=1), reference):
        raise ValueError(
            f"Score matrix argmax disagrees with {combiner_name}; refusing to report it"
        )
