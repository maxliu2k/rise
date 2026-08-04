"""Poster figures and tables for the noise sweep.

    python scripts/noise_figures.py                  # figures + tables from whatever has landed
    python scripts/noise_figures.py --require-all    # refuse unless all six models are present

Reads artifacts/<model>/noise/noise_sweep_summary.csv for every model in MODELS and writes
docs/figures/fig6_robustness_curves.{png,pdf} and fig7_robustness_auc.{png,pdf}, plus markdown
tables on stdout for pasting into the poster.

It is read-only with respect to results. The only files it writes are under docs/figures/.

WHAT IT REFUSES TO DO. A model whose results are absent is NAMED in the output and excluded --
never silently skipped, and never quietly averaged over. A figure that shows five curves when the
reader believes there are six is exactly the "silently wrong result" this repo keeps producing,
and it is worse on a poster than in a log because nobody can check it. --require-all turns the
absence into a hard failure for the final render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import NOISE_TYPES, SNRS  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"

# Display order and labels. SVM first (weakest, CPU-only), pretrained last.
MODELS: dict[str, str] = {
    "svm": "SVM",
    "cnn": "CNN",
    "crnn": "CRNN",
    "mert": "MERT",
    "panns": "PANNs",
    "ast": "AST",
}
NOISE_LABEL = {
    "white": "white (synthetic)",
    "audience": "audience (ESC-50 human)",
    "studio": "studio (DEMAND room tone)",
}
CHANCE = 1.0 / 12.0


def load_model(name: str) -> pd.DataFrame | None:
    """Return one model's sweep summary, or None if it has not been evaluated.

    Postcondition: the returned frame has one row per (noise_type, snr_db, replicate) plus the
    clean row, with a `macro_f1_retention` column already computed by the evaluator.
    """
    path = ARTIFACTS / name / "noise" / "noise_sweep_summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame["model"] = name
    return frame


def curve(frame: pd.DataFrame, noise_type: str, column: str) -> pd.DataFrame:
    """Mean and spread across replicates at each SNR, for one noise type.

    Replicates are averaged, not concatenated: two draws of the same condition are two estimates
    of one number, and treating them as separate curve points would duplicate every SNR.
    """
    subset = frame[frame["noise_type"] == noise_type]
    grouped = subset.groupby("snr_db")[column].agg(["mean", "std", "count"]).reset_index()
    return grouped.sort_values("snr_db")


def robustness_auc(snrs: np.ndarray, values: np.ndarray) -> float:
    """dB-weighted mean of a curve, invariant to how densely the grid was sampled.

    Uses the trapezoid rule over SNR and divides by the span, so adding levels where a model
    happens to do well does not inflate the score. This mirrors robustness_curve.robustness_auc;
    it is recomputed here only so the figure and the table cannot disagree about what was plotted.
    """
    order = np.argsort(snrs)
    x, y = snrs[order], values[order]
    if x.size < 2:
        return float("nan")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def setup_matplotlib():
    """Same house style as scripts/dataset_tables.py, so the poster's figures match."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    FIGURES.mkdir(parents=True, exist_ok=True)
    return plt


def save(plt, name: str) -> None:
    plt.savefig(FIGURES / f"{name}.png")
    plt.savefig(FIGURES / f"{name}.pdf")
    plt.close()


def figure_curves(plt, frames: dict[str, pd.DataFrame]) -> None:
    """Top row: absolute macro-F1. Bottom row: retention. One column per noise type.

    Both rows are shown because they answer different questions and a poster reader will ask
    both: absolute macro-F1 says which model to deploy, retention says which model DEGRADES
    least, and the strongest clean model need not be the most robust one.
    """
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        2, len(NOISE_TYPES), figsize=(3.1 * len(NOISE_TYPES), 5.2), sharex=True
    )
    if len(NOISE_TYPES) == 1:
        axes = axes.reshape(2, 1)

    for col, noise_type in enumerate(NOISE_TYPES):
        for index, (name, frame) in enumerate(frames.items()):
            colour = colours[index % len(colours)]
            for row, column in enumerate(("macro_f1", "macro_f1_retention")):
                grouped = curve(frame, noise_type, column)
                if grouped.empty:
                    continue
                axis = axes[row, col]
                axis.plot(grouped["snr_db"], grouped["mean"], marker="o", ms=3,
                          lw=1.4, color=colour, label=MODELS[name])
                spread = grouped["std"].fillna(0.0)
                if float(spread.max()) > 0:
                    axis.fill_between(grouped["snr_db"],
                                      grouped["mean"] - spread, grouped["mean"] + spread,
                                      color=colour, alpha=0.15, lw=0)

        axes[0, col].set_title(NOISE_LABEL.get(noise_type, noise_type))
        axes[0, col].axhline(CHANCE, color="0.4", ls=":", lw=1)
        axes[1, col].set_xlabel("SNR (dB)")
        for row in (0, 1):
            axes[row, col].invert_xaxis()

    axes[0, 0].set_ylabel("macro-F1")
    axes[1, 0].set_ylabel("retention (noisy / clean)")
    axes[0, 0].text(0.02, CHANCE + 0.02, "chance", transform=axes[0, 0].get_yaxis_transform(),
                    fontsize=7, color="0.4")
    axes[0, -1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    save(plt, "fig6_robustness_curves")


def figure_auc(plt, frames: dict[str, pd.DataFrame]) -> None:
    """Grouped bars: dB-weighted robustness AUC per model per noise type."""
    names = list(frames)
    width = 0.8 / max(len(names), 1)
    fig, axis = plt.subplots(figsize=(1.9 * len(NOISE_TYPES) + 2.2, 3.0))
    positions = np.arange(len(NOISE_TYPES))

    for index, name in enumerate(names):
        heights = []
        for noise_type in NOISE_TYPES:
            grouped = curve(frames[name], noise_type, "macro_f1")
            heights.append(
                robustness_auc(grouped["snr_db"].to_numpy(), grouped["mean"].to_numpy())
                if not grouped.empty else np.nan
            )
        axis.bar(positions + index * width - 0.4 + width / 2, heights, width * 0.92,
                 label=MODELS[name])

    axis.axhline(CHANCE, color="0.4", ls=":", lw=1)
    axis.set_xticks(positions)
    axis.set_xticklabels([NOISE_LABEL.get(t, t) for t in NOISE_TYPES], fontsize=8)
    axis.set_ylabel("robustness AUC\n(dB-weighted mean macro-F1)")
    axis.legend(fontsize=7, frameon=False, ncol=3)
    fig.tight_layout()
    save(plt, "fig7_robustness_auc")


def markdown_tables(frames: dict[str, pd.DataFrame], missing: list[str]) -> None:
    """Print poster-ready tables. Missing models are stated, not omitted in silence."""
    print("\n## Clean baseline and robustness AUC\n")
    print("| model | clean macro-F1 | " + " | ".join(
        f"AUC {t}" for t in NOISE_TYPES) + " |")
    print("|---" * (2 + len(NOISE_TYPES)) + "|")
    for name, frame in frames.items():
        clean = frame[~frame["noise_type"].isin(NOISE_TYPES)]["macro_f1"]
        cells = []
        for noise_type in NOISE_TYPES:
            grouped = curve(frame, noise_type, "macro_f1")
            value = robustness_auc(grouped["snr_db"].to_numpy(), grouped["mean"].to_numpy())
            cells.append(f"{value:.4f}")
        baseline = f"{float(clean.iloc[0]):.4f}" if len(clean) else "?"
        print(f"| {MODELS[name]} | {baseline} | " + " | ".join(cells) + " |")

    for noise_type in NOISE_TYPES:
        print(f"\n## macro-F1 vs SNR -- {NOISE_LABEL.get(noise_type, noise_type)}\n")
        print("| model | " + " | ".join(f"{s} dB" for s in SNRS) + " |")
        print("|---" * (1 + len(SNRS)) + "|")
        for name, frame in frames.items():
            grouped = curve(frame, noise_type, "macro_f1").set_index("snr_db")["mean"]
            cells = [f"{grouped[s]:.3f}" if s in grouped.index else "-" for s in SNRS]
            print(f"| {MODELS[name]} | " + " | ".join(cells) + " |")

    if missing:
        print("\n> **Not evaluated:** " + ", ".join(MODELS[m] for m in missing) +
              ". These models are absent from every figure and table above.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all", action="store_true",
                        help="exit non-zero unless every model in MODELS has results")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    frames, missing = {}, []
    for name in MODELS:
        frame = load_model(name)
        if frame is None:
            missing.append(name)
        else:
            frames[name] = frame

    print(f"models with results : {', '.join(frames) or '(none)'}")
    print(f"models MISSING      : {', '.join(missing) or '(none)'}")
    if not frames:
        print("nothing to plot", file=sys.stderr)
        return 1
    if missing and args.require_all:
        print(f"--require-all: refusing to render with {len(missing)} model(s) missing",
              file=sys.stderr)
        return 1

    # Tables do not need matplotlib, so a missing plotting library must not cost you the numbers.
    # matplotlib is not declared as a dependency of this package and was absent from the SCC core
    # venv on 2026-08-04, which killed this script before it printed anything useful.
    figures_failed = None
    if not args.no_figures:
        try:
            plt = setup_matplotlib()
            figure_curves(plt, frames)
            figure_auc(plt, frames)
            print(f"wrote {FIGURES}/fig6_robustness_curves.{{png,pdf}}")
            print(f"wrote {FIGURES}/fig7_robustness_auc.{{png,pdf}}")
        except ImportError as error:
            figures_failed = f"{error} -- install it, then re-run for the figures"
            print(f"! FIGURES SKIPPED: {figures_failed}", file=sys.stderr)

    markdown_tables(frames, missing)
    if figures_failed:
        print(f"\n> **Figures were not written:** {figures_failed}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
