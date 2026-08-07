"""One confusion-matrix panel per model, in a single readable grid.

    python scripts/confusion_grid.py                  # 2x3 grid, row-normalised
    python scripts/confusion_grid.py --errors-only    # mask the diagonal: show ONLY confusions
    python scripts/confusion_grid.py --require-all    # refuse unless all six models are present

Reads artifacts/<model>/noise/<model>_test_clean.csv -- the CLEAN condition of the noise sweep,
which is the same 1,255-window test split every model reproduces before any noisy condition is
scored. Writes docs/figures/fig8_confusion_grid.{png,pdf}. Read-only with respect to results.

WHY ROW-NORMALISED AND NOT COUNTS. Every model here is between 97% and 99% accurate; AST
misclassifies 11 windows out of 1,255. On a raw-count colour scale that is six panels of
identical-looking diagonal, and the off-diagonal structure -- the only part a reader cannot get
from the macro-F1 number already in the caption -- is invisible. Dividing each row by its class
support makes the diagonal recall and the off-diagonal the share of that instrument sent
elsewhere, which is comparable across models and across classes of different size.

WHY A POWER NORM. Even row-normalised, a 99%-accurate model puts ~0.99 on the diagonal and
~0.01 off it. A linear colour ramp spends its whole range on the diagonal. gamma < 1 expands the
low end so a 2% confusion is distinguishable from a 0.2% one, which is the comparison the figure
exists to support.

WHY --errors-only EXISTS. With the diagonal masked, the colour scale is set by the largest
CONFUSION rather than by recall, so the panels stop being six diagonals and start being six
error patterns. For a poster that is usually the more informative rendering; the diagonal
carries no information the per-panel macro-F1 does not already state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import TARGET_LABELS  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"

# Display order: weakest/classical first, audio-pretrained last, matching noise_figures.MODELS.
MODELS: dict[str, str] = {
    "svm": "SVM",
    "cnn": "CNN",
    "crnn": "CRNN",
    "mert_ft": "MERT",
    "panns": "PANNs",
    "ast": "AST",
}


def clean_predictions(name: str) -> pd.DataFrame | None:
    """Return one model's CLEAN-condition predictions, or None if it has not been evaluated.

    Preconditions: none.
    Postcondition: frame with `true_label` and `predicted_label`, or None.
    Raises: ValueError if the file exists but lacks the label columns, because a silently empty
    panel is exactly the failure this repo keeps producing.
    """
    # GLOB, do not reconstruct the prefix. Each adapter chooses its own `file_prefix`, and they
    # are not all `<model>_test_`: PANNs writes `panns_ft_test_clean.csv` because the reported
    # result is the fine-tune rather than the linear probe. Guessing the name silently dropped
    # PANNs from the grid, which is the "five curves where the reader believes six" failure.
    directory = ARTIFACTS / name / "noise"
    matches = sorted(directory.glob("*_test_clean.csv")) if directory.is_dir() else []
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"{directory} has several clean prediction files: {matches}")
    frame = pd.read_csv(matches[0])
    missing = {"true_label", "predicted_label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return frame


def confusion(frame: pd.DataFrame) -> np.ndarray:
    """Counts, rows = true class, columns = predicted class, in TARGET_LABELS order.

    Postcondition: shape (12, 12), integer, sums to len(frame).
    Raises: ValueError on a label outside TARGET_LABELS -- an unknown class means the file was
    written against a different class set and must not be plotted beside the others.
    """
    index = {label: position for position, label in enumerate(TARGET_LABELS)}
    matrix = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=int)
    for true_label, predicted_label in zip(frame["true_label"], frame["predicted_label"]):
        if true_label not in index or predicted_label not in index:
            raise ValueError(f"label outside TARGET_LABELS: {true_label!r}/{predicted_label!r}")
        matrix[index[true_label], index[predicted_label]] += 1
    return matrix


def macro_f1_from(matrix: np.ndarray) -> float:
    """Macro-F1 straight from the confusion matrix, so the caption cannot drift from the picture."""
    scores = []
    for position in range(matrix.shape[0]):
        true_positive = matrix[position, position]
        predicted = matrix[:, position].sum()
        actual = matrix[position, :].sum()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return float(np.mean(scores))


def row_normalised(matrix: np.ndarray) -> np.ndarray:
    """Each row divided by its support, so cells are the share of that TRUE class."""
    support = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.where(support > 0, matrix / support, 0.0)
    return shares


def build(errors_only: bool, present: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import poster_style; poster_style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    from matplotlib.colors import LinearSegmentedColormap

    names = [name for name in MODELS if name in present]
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.3 * columns, 4.3 * rows), squeeze=False
    )
    figure.patch.set_facecolor("white")

    panels = {}
    for name in names:
        shares = row_normalised(present[name])
        if errors_only:
            shares = shares.copy()
            np.fill_diagonal(shares, np.nan)
        panels[name] = shares

    # One colour scale across every panel, or the panels are not comparable. With the diagonal
    # masked the scale is set by the worst confusion, which is what we want to compare.
    finite = np.concatenate([panel[np.isfinite(panel)].ravel() for panel in panels.values()])
    high = float(np.nanmax(finite)) if finite.size else 1.0
    norm = PowerNorm(gamma=0.45, vmin=0.0, vmax=high)

    # Sequential white -> deep ink. ZERO MUST BE WHITE: a cell with no confusions should read as
    # empty paper, not as a colour. That rules out viridis and magma_r, whose low end is a
    # saturated yellow -- the previous version drew six panels of pale yellow background, which
    # is what made it look muddy. Single-hue also survives greyscale printing and the common
    # colour-vision deficiencies, which a poster cannot control for.
    ramp = ["#ffffff", "#e3ebf7", "#9db9e0", "#4a76b8", "#20447d", "#0d1f3d"]
    if errors_only:
        ramp = ["#ffffff", "#fbe2df", "#f0a89c", "#d95f4c", "#a32c1f", "#5e1109"]
    colours = LinearSegmentedColormap.from_list("rise", ramp)
    colours.set_bad(color="#ededed")   # masked diagonal reads as absent, not as zero

    image = None
    for position, name in enumerate(names):
        axis = axes[position // columns][position % columns]
        image = axis.imshow(panels[name], cmap=colours, norm=norm, aspect="equal")

        matrix = present[name]
        errors = int(matrix.sum() - np.trace(matrix))
        axis.set_title(MODELS[name], fontsize=13, fontweight="semibold", pad=13)
        # Stats as a lighter subtitle rather than a second title line: the model name is what a
        # reader scans for, the counts are what they check afterwards.
        axis.text(
            0.5, 1.015,
            f"macro-F1 {macro_f1_from(matrix):.3f}   ·   {errors} / {matrix.sum():,} wrong",
            transform=axis.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="#5a5a5a",
        )

        axis.set_xticks(range(len(TARGET_LABELS)))
        axis.set_yticks(range(len(TARGET_LABELS)))
        # Tick LABELS only on the outer edges. Repeating twelve instrument names six times is
        # what makes a grid of confusion matrices unreadable.
        bottom_row = position // columns == rows - 1
        left_column = position % columns == 0
        axis.set_xticklabels(
            TARGET_LABELS if bottom_row else [], rotation=90, fontsize=7.5
        )
        axis.set_yticklabels(TARGET_LABELS if left_column else [], fontsize=7.5)
        if bottom_row:
            axis.set_xlabel("predicted instrument", fontsize=9.5, labelpad=6)
        if left_column:
            axis.set_ylabel("true instrument", fontsize=9.5, labelpad=6)
        axis.tick_params(length=0, colors="#444444")

        # Hairline white gutters between cells. Without them a run of adjacent dark cells fuses
        # into one blob and you cannot count instruments along an edge.
        axis.set_xticks(np.arange(-0.5, len(TARGET_LABELS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(TARGET_LABELS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7)
        axis.tick_params(which="minor", length=0)
        for spine in axis.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)

    for position in range(len(names), rows * columns):
        axes[position // columns][position % columns].axis("off")

    label = (
        "share of true class sent to the wrong instrument"
        if errors_only
        else "share of true class (diagonal = recall)"
    )
    bar = figure.colorbar(
        image, ax=axes, fraction=0.018, pad=0.018, aspect=34
    )
    bar.set_label(label, fontsize=9.5, labelpad=10)
    bar.ax.tick_params(labelsize=8, length=2, colors="#444444")
    bar.outline.set_edgecolor("#cccccc")
    bar.outline.set_linewidth(0.8)
    # Ticks as percentages: "2%" is read instantly, "0.02" needs a translation step.
    from matplotlib.ticker import FuncFormatter
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = "fig8_confusion_grid_errors" if errors_only else "fig8_confusion_grid"
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {FIGURES}/{stem}.{{png,pdf}}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="mask the diagonal and scale colour by the worst confusion",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="exit non-zero unless all six models have clean predictions",
    )
    arguments = parser.parse_args()

    present: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in MODELS:
        frame = clean_predictions(name)
        if frame is None:
            missing.append(name)
            continue
        present[name] = confusion(frame)

    print("models with clean predictions:", ", ".join(present) or "(none)")
    print("models MISSING                :", ", ".join(missing) or "(none)")
    if missing and arguments.require_all:
        print("refusing to render an incomplete grid (--require-all)", file=sys.stderr)
        return 1
    if not present:
        print("nothing to plot", file=sys.stderr)
        return 1

    build(arguments.errors_only, present)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
