"""Retention-vs-SNR, one row of three panels, for the 5-page paper.

    python scripts/fig6b_retention_row.py

Writes docs/figures/fig6b_retention_row.{png,pdf}. Read-only with respect to results.

WHY THIS EXISTS ALONGSIDE fig6. fig6_robustness_curves is two rows -- absolute macro-F1 on top,
retention below -- and at IEEE \textwidth it costs 43% of a page. The paper's metric is retention:
Eq.4 defines it, Eq.5 integrates it, and the results table reports its AUC. Nothing in the paper
reads the absolute row. Dropping it halves the figure to ~24% of a page and removes the only place
where two different y-quantities share one caption.

WHAT IS LOST, STATED HONESTLY. The chance line (1/12) is meaningful on absolute macro-F1 and
meaningless on retention, so it is not drawn here. A reader can no longer see that SVM approaches
chance at low SNR. fig6 remains in the repository for the poster and for anyone who wants it.

The legend sits BELOW the axes. In fig6 it is inside the third panel, where it overlaps the curves
and the chance line -- fine on a 36x48 poster, not fine at column width.

Panel titles use the paper's category names (white / human non-speech / environmental) rather than
the code tags (white / audience / studio). The code tags are handles; `audience` is ESC-50 human
non-speech events and `studio` is DEMAND ambience from 18 environments, most of which are not
studios. docs/FAILURE_ANALYSIS_PLAN.md asks for the descriptive names in paper text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import NOISE_TYPES  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"

# Same keys and order as noise_figures.MODELS, so colours match the rest of the figure set.
MODELS: dict[str, str] = {
    "svm": "SVM",
    "cnn": "CNN",
    "crnn": "CRNN",
    "mert_ft": "MERT",
    "panns": "PANNs",
    "ast": "AST",
}
PAPER_LABEL = {
    "white": "white (Gaussian)",
    "audience": "human non-speech (ESC-50)",
    "studio": "environmental (DEMAND)",
}


def load(name: str) -> pd.DataFrame:
    """One model's sweep summary.

    Raises: FileNotFoundError -- a missing model must stop the render, not silently draw five
    curves where the caption promises six.
    """
    path = ARTIFACTS / name / "noise" / "noise_sweep_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no sweep summary for {name}: {path}")
    return pd.read_csv(path)


def curve(frame: pd.DataFrame, noise_type: str) -> pd.DataFrame:
    """Mean and spread of retention across replicates at each SNR."""
    subset = frame[frame["noise_type"] == noise_type]
    grouped = (
        subset.groupby("snr_db")["macro_f1_retention"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("snr_db")
    )
    return grouped


def main() -> int:
    frames = {name: load(name) for name in MODELS}

    import matplotlib
    matplotlib.use("Agg")
    import poster_style; poster_style.apply()
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    figure, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(9.3, 2.9), sharey=True)
    handles = None
    for column, noise_type in enumerate(NOISE_TYPES):
        axis = axes[column]
        for index, (name, frame) in enumerate(frames.items()):
            grouped = curve(frame, noise_type)
            axis.plot(grouped["snr_db"], grouped["mean"], marker="o", ms=3, lw=1.4,
                      color=colours[index % len(colours)], label=MODELS[name])
            spread = grouped["std"].fillna(0.0)
            if float(spread.max()) > 0:
                axis.fill_between(grouped["snr_db"], grouped["mean"] - spread,
                                  grouped["mean"] + spread,
                                  color=colours[index % len(colours)], alpha=0.15, lw=0)
        axis.set_title(PAPER_LABEL.get(noise_type, noise_type), fontsize=10)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylim(0, 1.05)
        if handles is None:
            handles, _ = axis.get_legend_handles_labels()

    axes[0].set_ylabel("retention (noisy / clean)")
    # Legend OUTSIDE the axes, one row beneath all three panels.
    figure.legend(handles, list(MODELS.values()), loc="lower center", ncol=6,
                  frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    figure.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"fig6b_retention_row.{suffix}")
    plt.close(figure)
    print(f"wrote {FIGURES}/fig6b_retention_row.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
