"""Native rebuilds of the poster's Figure 9a and 9b from the committed failure analysis.

    python scripts/fig9_pair_figures.py

Writes docs/figures/fig9a_distance_heatmap.{png,pdf} and fig9b_ast_scatter.{png,pdf}.
Read-only with respect to results.

WHY. The originals were raster exports in DejaVu Sans on a board standardised on Times New
Roman. Every number they show lives in artifacts/failure_analysis/ -- the 18 Spearman tests in
distance_confusion_tests.csv, the 66 pair distances in acoustic_distances.csv, the per-pair
confusion in pair_confusion_summary.csv -- so the figures are rebuilt from the data rather than
restyled as images. Values are READ, never recomputed: the statistics stay exactly the ones
Allan's pre-registered analysis produced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

FA = ROOT / "artifacts" / "failure_analysis"
FIGURES = ROOT / "docs" / "figures"

MODELS = ["svm", "cnn", "crnn", "mert", "panns", "ast"]
MODEL_LABEL = {"svm": "SVM", "cnn": "CNN", "crnn": "CRNN", "mert": "MERT",
               "panns": "PANNs", "ast": "AST"}
NOISE_ORDER = ["white", "audience", "studio"]
NOISE_LABEL = {"white": "White Gaussian", "audience": "ESC-50\nhuman non-speech",
               "studio": "DEMAND\nenvironmental ambience"}


def heatmap(plt) -> None:
    tests = pd.read_csv(FA / "distance_confusion_tests.csv")
    rho = np.zeros((len(MODELS), len(NOISE_ORDER)))
    starred = np.zeros_like(rho, dtype=bool)
    for _, row in tests.iterrows():
        i, j = MODELS.index(row["model"]), NOISE_ORDER.index(row["noise_type"])
        rho[i, j] = row["spearman_rho"]
        starred[i, j] = bool(row["bh_rejected_at_0.05"])

    figure, axis = plt.subplots(figsize=(7.4, 6.2))
    figure.patch.set_facecolor("white")
    image = axis.imshow(rho, cmap="RdBu_r", vmin=-0.65, vmax=0.65, aspect="auto")
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            dark = abs(rho[i, j]) > 0.45
            axis.text(j, i, f"{rho[i, j]:.3f}" + (" *" if starred[i, j] else ""),
                      ha="center", va="center", fontsize=12,
                      color="white" if dark else "#1a1a1a")
    axis.set_xticks(range(len(NOISE_ORDER)))
    axis.set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER], fontsize=10)
    axis.set_yticks(range(len(MODELS)))
    axis.set_yticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=11)
    axis.set_title("Acoustic distance vs noise-induced confusion\n"
                   "66 instrument pairs per cell · negative ρ: closer pairs "
                   "confused more", fontsize=12.5, pad=12)
    bar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    bar.set_label("Spearman correlation (ρ)", fontsize=10)
    axis.text(0.5, -0.10, "* BH-adjusted q < 0.05   ·   association, not causation",
              transform=axis.transAxes, ha="center", fontsize=10, color="#39597a")
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"fig9a_distance_heatmap.{suffix}")
    plt.close(figure)
    print("wrote fig9a_distance_heatmap.{png,pdf}")


def scatter(plt) -> None:
    dist = pd.read_csv(FA / "acoustic_distances.csv")
    pairs = pd.read_csv(FA / "pair_confusion_summary.csv")
    tests = pd.read_csv(FA / "distance_confusion_tests.csv")

    ast = pairs[(pairs["model"] == "ast") & (pairs["noise_type"] == "audience")]
    merged = ast.merge(dist, on=["instrument_a", "instrument_b"], validate="one_to_one")
    if len(merged) != 66:
        raise ValueError(f"expected 66 pairs, joined {len(merged)}")
    t = tests[(tests["model"] == "ast") & (tests["noise_type"] == "audience")].iloc[0]

    figure, axis = plt.subplots(figsize=(6.8, 6.2))
    figure.patch.set_facecolor("white")
    axis.scatter(merged["acoustic_distance"], merged["mean_confusion_increase_auc"],
                 s=42, color="#2c6fa8", alpha=0.75, edgecolor="white", linewidth=0.7)
    axis.set_xlabel("Acoustic distance between class centroids\n"
                    "(standardized 88-D handcrafted feature space)", fontsize=10.5)
    axis.set_ylabel("Noise-induced confusion\n(mean confusion-increase AUC)", fontsize=10.5)
    axis.set_title("AST — ESC-50 human non-speech", fontsize=13, pad=12)
    axis.text(0.97, 0.95,
              f"Spearman ρ = {t['spearman_rho']:.3f}\n"
              f"BH-adjusted q = {t['bh_q_value']:.5f}\n"
              f"{int(t['n_pairs'])} pairs · {int(t['n_permutations']):,} permutations",
              transform=axis.transAxes, ha="right", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef3fa",
                        edgecolor="#9db9e0", linewidth=0.8))
    axis.grid(True)
    axis.text(0.5, -0.17, "One cell of the full heatmap, shown for illustration.\n"
              "Association only; the feature space is not a perceptual model.",
              transform=axis.transAxes, ha="center", fontsize=9.5, color="#39597a")
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"fig9b_ast_scatter.{suffix}")
    plt.close(figure)
    print("wrote fig9b_ast_scatter.{png,pdf}")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import poster_style
    poster_style.apply()
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    heatmap(plt)
    scatter(plt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
