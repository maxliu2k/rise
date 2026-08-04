"""Clean accuracy against noise robustness, per instrument.

    python scripts/clean_vs_robust.py

Writes docs/figures/fig9_clean_vs_robust.{png,pdf}. Read-only with respect to results.

WHAT IT ARGUES. The poster's model-level claim is that clean accuracy does not predict
robustness. This is the same claim at the INSTRUMENT level, which is stronger evidence because
it does not depend on the six models differing from one another: it holds within each of them.

  x  clean errors, summed over all six models (artifacts/<model>/noise/*_test_clean.csv)
  y  mean recall-loss AUC under noise, averaged over models
     (artifacts/failure_analysis/instrument_recall_loss_summary.csv)

If clean accuracy predicted robustness the points would trend upward. Tuba is the extreme case:
zero clean errors across every model, and the largest recall loss of all twelve instruments
under all three noise types.

Both inputs are committed, so this runs off-cluster.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import TARGET_LABELS  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"
MODELS = ("svm", "cnn", "crnn", "mert", "panns", "ast")


def clean_errors_by_instrument() -> pd.Series:
    """Misclassification count per TRUE instrument, summed over the six models.

    Raises: FileNotFoundError if any model's clean predictions are absent -- a total summed over
    five models silently understates every instrument and must not be plotted.
    """
    totals = pd.Series(0, index=list(TARGET_LABELS), dtype=int)
    for model in MODELS:
        matches = glob.glob(str(ARTIFACTS / model / "noise" / "*_test_clean.csv"))
        if not matches:
            raise FileNotFoundError(f"no clean predictions for {model}")
        frame = pd.read_csv(matches[0])
        wrong = frame[frame["true_label"] != frame["predicted_label"]]
        totals = totals.add(wrong["true_label"].value_counts(), fill_value=0)
    return totals.astype(int)


def recall_loss_by_instrument() -> pd.DataFrame:
    """Mean recall-loss AUC per instrument per noise type, averaged over models."""
    path = ARTIFACTS / "failure_analysis" / "instrument_recall_loss_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} -- run instrument_robustness.failure_analysis first")
    frame = pd.read_csv(path)
    return frame.pivot_table(
        index="label", columns="noise_type", values="mean_recall_loss_auc", aggfunc="mean"
    )


def main() -> int:
    errors = clean_errors_by_instrument()
    losses = recall_loss_by_instrument()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_types = [t for t in ("audience", "studio", "white") if t in losses.columns]
    colours = {"audience": "#c0392b", "studio": "#2c6fa8", "white": "#7f8c8d"}
    markers = {"audience": "o", "studio": "s", "white": "^"}

    figure, axis = plt.subplots(figsize=(7.6, 5.6))
    figure.patch.set_facecolor("white")

    for noise in noise_types:
        x = [errors[label] for label in losses.index]
        y = losses[noise].to_numpy()
        axis.scatter(
            x, y, s=52, color=colours[noise], marker=markers[noise],
            alpha=0.85, edgecolor="white", linewidth=0.8, label=noise, zorder=3,
        )

    # Label each instrument once, at its audience point, so twelve names appear rather than 36.
    anchor = "audience" if "audience" in noise_types else noise_types[0]
    # Manual nudges where two instruments land on nearly the same point. trombone (9, 0.204) and
    # viola (11, 0.205) overlapped exactly; left as-is one name was unreadable.
    nudge = {"viola": (7, 6), "trombone": (7, -11), "clarinet": (7, 4)}
    for label in losses.index:
        axis.annotate(
            label,
            (errors[label], losses.loc[label, anchor]),
            textcoords="offset points", xytext=nudge.get(label, (7, -3)),
            fontsize=8.5, color="#333333", zorder=4,
        )

    # Spearman across instruments: does clean error count predict noise recall loss at all?
    from scipy.stats import spearmanr
    rho, p_value = spearmanr(
        [errors[label] for label in losses.index], losses[anchor].to_numpy()
    )
    axis.text(
        0.98, 0.03,
        f"Spearman ρ = {rho:+.2f}  (p = {p_value:.2f}), {anchor}",
        transform=axis.transAxes, ha="right", va="bottom",
        fontsize=9, color="#555555",
    )

    axis.set_xlabel("clean misclassifications, summed over all six models", fontsize=10)
    axis.set_ylabel("mean recall-loss AUC under noise\n(higher = degrades more)", fontsize=10)
    axis.set_title(
        "Clean accuracy does not predict noise robustness",
        fontsize=13, fontweight="semibold", pad=12,
    )
    axis.legend(title="noise", fontsize=9, title_fontsize=9, frameon=False, loc="upper right")
    axis.grid(True, color="#e8e8e8", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#bbbbbb")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"fig9_clean_vs_robust.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {FIGURES}/fig9_clean_vs_robust.{{png,pdf}}")

    print(f"\nSpearman (clean errors vs {anchor} recall loss): rho={rho:+.3f} p={p_value:.3f}")
    table = pd.DataFrame({"clean_errors": [errors[i] for i in losses.index]}, index=losses.index)
    for noise in noise_types:
        table[noise] = losses[noise].round(3)
    print(table.sort_values(anchor, ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
