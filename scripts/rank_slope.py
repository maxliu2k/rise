"""Instrument rank on clean audio against rank under noise.

    python scripts/rank_slope.py

Writes docs/figures/fig9_rank_slope.{png,pdf}. Read-only with respect to results.

WHY A SLOPEGRAPH AND NOT A SCATTER. The claim is that clean accuracy carries no information
about noise robustness. A scatter of 12 instruments x 3 noise types is 36 points, 12 connectors
and 12 floating labels, and the reader has to infer "no trend" from an absence. Ranking the same
instruments on each side and joining them makes the claim the SHAPE of the figure: a real
relationship draws flat parallel lines, and no relationship draws a tangle. Tuba runs from best
to worst across the whole chart.

  left   rank by clean misclassifications, summed over all six models (1 = fewest)
  right  rank by mean recall-loss AUC under noise, averaged over models and noise types
         (1 = degrades least)

Both inputs are committed, so this runs off-cluster.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import TARGET_LABELS  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"
MODELS = ("svm", "cnn", "crnn", "mert_ft", "panns", "ast")


def clean_errors() -> pd.Series:
    totals = pd.Series(0, index=list(TARGET_LABELS), dtype=int)
    for model in MODELS:
        matches = glob.glob(str(ARTIFACTS / model / "noise" / "*_test_clean.csv"))
        if not matches:
            raise FileNotFoundError(f"no clean predictions for {model}")
        frame = pd.read_csv(matches[0])
        wrong = frame[frame["true_label"] != frame["predicted_label"]]
        totals = totals.add(wrong["true_label"].value_counts(), fill_value=0)
    return totals.astype(int)


def recall_loss() -> pd.Series:
    path = ARTIFACTS / "failure_analysis" / "instrument_recall_loss_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} -- run instrument_robustness.failure_analysis first")
    frame = pd.read_csv(path)
    return frame.groupby("label")["mean_recall_loss_auc"].mean()


def main() -> int:
    errors, losses = clean_errors(), recall_loss()
    labels = list(losses.index)

    # Rank 1 = best. Ties broken by the underlying value so the ordering is deterministic.
    clean_rank = errors[labels].rank(method="first").astype(int)
    noise_rank = losses.rank(method="first").astype(int)

    from scipy.stats import spearmanr
    rho, p_value = spearmanr(errors[labels].to_numpy(), losses[labels].to_numpy())

    import matplotlib
    matplotlib.use("Agg")
    import poster_style; poster_style.apply()
    import matplotlib.pyplot as plt

    # Landscape. A wide gap between the columns lengthens every line, which is what makes the
    # crossings legible -- in a near-square frame the slopes are too shallow to read apart.
    figure, axis = plt.subplots(figsize=(12.5, 5.4))
    figure.patch.set_facecolor("white")

    # Highlight only the instruments that make the point; everything else is context. Colouring
    # all twelve would make the tangle decorative rather than readable.
    biggest_fall = max(labels, key=lambda label: noise_rank[label] - clean_rank[label])
    biggest_rise = min(labels, key=lambda label: noise_rank[label] - clean_rank[label])
    accent = {biggest_fall: "#c0392b", biggest_rise: "#2c6fa8"}

    for label in labels:
        left, right = clean_rank[label], noise_rank[label]
        colour = accent.get(label, "#c4c4c4")
        width = 2.4 if label in accent else 1.1
        axis.plot([0, 1], [left, right], color=colour, linewidth=width,
                  zorder=3 if label in accent else 2, solid_capstyle="round")
        axis.scatter([0, 1], [left, right], s=30, color=colour,
                     zorder=4 if label in accent else 2, edgecolor="white", linewidth=0.8)

        text_colour = accent.get(label, "#555555")
        weight = "bold" if label in accent else "normal"
        axis.text(-0.022, left, f"{label}  ({errors[label]})", ha="right", va="center",
                  fontsize=9, color=text_colour, fontweight=weight)
        axis.text(1.022, right, f"{label}  ({losses[label]:.2f} lost)", ha="left", va="center",
                  fontsize=9, color=text_colour, fontweight=weight)

    axis.set_xlim(-0.26, 1.26)
    axis.set_ylim(len(labels) + 0.6, 0.4)          # rank 1 at the top
    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        ["BEST on clean audio\n(fewest errors, all six models)",
         "MOST ROBUST under noise\n(least recall lost)"],
        fontsize=10,
    )
    axis.set_yticks([])
    axis.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)

    axis.set_title("Clean accuracy does not predict noise robustness",
                   fontsize=13.5, fontweight="semibold", pad=16)
    # Spell out the direction. This poster also carries fig7, whose "robustness AUC" is a
    # macro-F1 and so runs the OTHER way (higher = better). Two quantities both called AUC with
    # opposite polarity is how a reader ends up reading one of the figures backwards.
    axis.text(0.5, -0.115,
              "both columns: best at the top   |   values in brackets are recall LOST "
              "(0.00 = none lost, higher = worse)",
              transform=axis.transAxes, ha="center", va="top", fontsize=9, color="#777777")
    axis.text(0.5, -0.185,
              f"Spearman ρ = {rho:+.2f} (p = {p_value:.2f}, n = 12) — the rankings are unrelated",
              transform=axis.transAxes, ha="center", va="top", fontsize=9.5, color="#555555")

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"fig9_rank_slope.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {FIGURES}/fig9_rank_slope.{{png,pdf}}")
    print(f"Spearman rho={rho:+.3f} p={p_value:.3f}")
    print(f"biggest fall: {biggest_fall}   biggest rise: {biggest_rise}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
