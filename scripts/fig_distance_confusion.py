"""Acoustic distance against noise-induced confusion (paper Figure 3).

    python scripts/fig_distance_confusion.py

Writes docs/figures/fig_distance_confusion{,_compact}.{png,pdf}. Read-only with respect to
results.

WHAT IT SHOWS. Panel A is the whole pre-registered family: Spearman rho between inter-instrument
acoustic distance and noise-induced confusion, for each of the six models crossed with the three
noise categories, with the cells surviving Benjamini-Hochberg at q<0.05 outlined and starred.
Panel B opens up the strongest single cell, AST under human non-speech noise, as the underlying
66-pair scatter. Together they answer "does this hold in general" and "what does it look like"
without the reader having to take either on trust.

NO REGRESSION LINE IN PANEL B. The reported statistic is Spearman, which tests monotonicity and
nothing else. Drawing a least-squares line through the cloud would assert a linear relationship
that was never tested and that the rank-based statistic cannot support. The scatter and the
annotated rho are the honest presentation.

SYMMETRIC DIVERGING SCALE. rho runs -0.607 to +0.004, so the sign is the finding: negative means
acoustically closer instrument pairs are confused MORE under noise, which is the hypothesis. A
sequential scale would bury the sign. Limits are symmetric about zero so that equal magnitudes in
opposite directions get equal visual weight.

THE SEVEN-CELL ASSERTION. The paper says "seven of the 18". This script counts the rejections and
refuses to render if that count is not seven, because a figure that quietly disagrees with its own
caption is the failure mode this repository keeps hitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import NOISE_TYPES  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figures"
ANALYSIS = ARTIFACTS / "failure_analysis"
TESTS = ANALYSIS / "distance_confusion_tests.csv"
DISTANCES = ANALYSIS / "acoustic_distances.csv"
PAIRS = ANALYSIS / "pair_confusion_summary.csv"

MODELS: dict[str, str] = {
    "svm": "SVM",
    "cnn": "CNN",
    "crnn": "CRNN",
    "mert_ft": "MERT",
    "panns": "PANNs",
    "ast": "AST",
}
SHORT_LABEL = {
    "white": "white",
    "audience": "human\nnon-speech",
    "studio": "environ-\nmental",
}

# The cell Panel B opens up. Named here rather than recomputed so that the panel cannot silently
# drift to a different model-noise pair than the caption claims.
FOCUS_MODEL, FOCUS_NOISE = "ast", "audience"
EXPECTED_REJECTIONS = 7
EXPECTED_PAIRS = 66
RHO_LIMIT = 0.65


def load_tests() -> pd.DataFrame:
    """The 18 pre-registered model-by-noise correlation tests.

    Preconditions: failure_analysis has been run.
    Postcondition: one row per (model, noise_type), with `spearman_rho` and
    `bh_rejected_at_0.05`.
    Raises: FileNotFoundError if absent; ValueError if the family is not the declared 18, if any
    combination is missing, if a rho falls outside the plotted range, or if the number of BH
    rejections is not the seven the paper reports.
    """
    if not TESTS.is_file():
        raise FileNotFoundError(f"no distance-confusion tests: {TESTS}")
    frame = pd.read_csv(TESTS)

    expected = len(MODELS) * len(NOISE_TYPES)
    if len(frame) != expected:
        raise ValueError(f"{TESTS}: expected {expected} tests, found {len(frame)}")

    present = set(zip(frame["model"], frame["noise_type"]))
    missing = [(m, n) for m in MODELS for n in NOISE_TYPES if (m, n) not in present]
    if missing:
        raise ValueError(f"{TESTS} is missing {missing}")

    if frame["spearman_rho"].abs().max() > RHO_LIMIT:
        raise ValueError(
            f"{TESTS}: a rho exceeds the plotted limit {RHO_LIMIT}; widen RHO_LIMIT rather than "
            f"letting a cell clip: max |rho| = {frame['spearman_rho'].abs().max():.4f}"
        )

    rejected = int(frame["bh_rejected_at_0.05"].sum())
    if rejected != EXPECTED_REJECTIONS:
        raise ValueError(
            f"{TESTS}: {rejected} BH rejections, but the paper says {EXPECTED_REJECTIONS}. "
            "Update the text and this constant together, or the figure contradicts its caption."
        )
    return frame


def load_focus() -> tuple[pd.DataFrame, pd.Series]:
    """The 66-pair scatter for the focus cell, plus that cell's test row.

    Postcondition: a frame of exactly EXPECTED_PAIRS rows carrying `acoustic_distance` and
    `mean_confusion_increase_auc`, and the matching row of the tests table.
    Raises: ValueError if the join does not cover every instrument pair -- a short join would
    silently drop points from a scatter whose whole content is the shape of the cloud.
    """
    distances = pd.read_csv(DISTANCES)
    if len(distances) != EXPECTED_PAIRS:
        raise ValueError(f"{DISTANCES}: expected {EXPECTED_PAIRS} pairs, found {len(distances)}")

    pairs = pd.read_csv(PAIRS)
    subset = pairs[(pairs["model"] == FOCUS_MODEL) & (pairs["noise_type"] == FOCUS_NOISE)]
    merged = distances.merge(subset, on=["instrument_a", "instrument_b"], how="inner")
    if len(merged) != EXPECTED_PAIRS:
        raise ValueError(
            f"joining {DISTANCES.name} to {PAIRS.name} for {FOCUS_MODEL}/{FOCUS_NOISE} gave "
            f"{len(merged)} rows, expected {EXPECTED_PAIRS}"
        )

    tests = load_tests()
    row = tests[(tests["model"] == FOCUS_MODEL) & (tests["noise_type"] == FOCUS_NOISE)].iloc[0]
    return merged, row


def matrix(tests: pd.DataFrame, column: str) -> pd.DataFrame:
    """Models (rows) by noise categories (columns) for one column of the tests table."""
    grid = tests.pivot(index="model", columns="noise_type", values=column)
    return grid.reindex(index=list(MODELS), columns=list(NOISE_TYPES))


# width, height, tick font, stack panels vertically instead of side by side
VARIANTS: dict[str, tuple[float, float, float, bool]] = {
    "full": (7.16, 2.60, 8.0, False),
    "compact": (3.50, 4.40, 6.5, True),
    # Single IEEE column, panels side by side rather than stacked. Stacking costs 4.4 in of a
    # column for the same information; side by side costs under 2 in.
    "column": (3.50, 1.85, 5.0, False),
}


def render(tests: pd.DataFrame, focus: pd.DataFrame, focus_row: pd.Series, variant: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import poster_style; poster_style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": False})

    width, height, tick, stacked = VARIANTS[variant]
    tight = variant == "column"
    if stacked:
        figure, axes = plt.subplots(2, 1, figsize=(width, height))
    else:
        figure, axes = plt.subplots(
            1, 2, figsize=(width, height), gridspec_kw={"width_ratios": [1.0, 1.25]}
        )
    left, right = axes

    # ---- Panel A: the whole 6x3 family -------------------------------------------------
    rho = matrix(tests, "spearman_rho")
    rejected = matrix(tests, "bh_rejected_at_0.05")
    image = left.imshow(
        rho.to_numpy(), cmap="RdBu", vmin=-RHO_LIMIT, vmax=RHO_LIMIT,
        aspect="auto", interpolation="nearest",
    )
    for r, model in enumerate(MODELS):
        for c, noise in enumerate(NOISE_TYPES):
            value = float(rho.loc[model, noise])
            hit = bool(rejected.loc[model, noise])
            shade = "white" if abs(value) > 0.42 else poster_style.TEXT
            # In one column a cell is ~0.4 in wide; "-0.35*" only fits without its leading zero.
            text = (f"{value:.2f}".replace("-0.", "$-$.").replace("0.00", ".00")
                    if tight else f"{value:.2f}")
            left.text(
                c, r, text + ("*" if hit else ""),
                ha="center", va="center", fontsize=tick, color=shade,
            )
            if hit:
                left.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                    edgecolor=poster_style.TEXT, linewidth=1.4,
                ))
    left.set_xticks(range(len(NOISE_TYPES)))
    left.set_xticklabels([SHORT_LABEL[n] for n in NOISE_TYPES], fontsize=tick)
    left.set_yticks(range(len(MODELS)))
    left.set_yticklabels(list(MODELS.values()), fontsize=tick)
    left.tick_params(length=2)
    left.set_title("A  Spearman $\\rho$" if tight else "A  Spearman $\\rho$, all 18 tests",
                   fontsize=tick + 2, loc="left")
    for spine in left.spines.values():
        spine.set_visible(False)

    bar = figure.colorbar(image, ax=left, fraction=0.035, pad=0.03)
    bar.set_label("$\\rho$", fontsize=tick + 1)
    bar.ax.tick_params(labelsize=tick - 0.5)
    bar.outline.set_visible(False)

    # ---- Panel B: the strongest cell, opened up --------------------------------------
    right.scatter(
        focus["acoustic_distance"], focus["mean_confusion_increase_auc"],
        s=6 if tight else 14, color="#b2182b", alpha=0.75, linewidths=0,
    )
    right.set_xlabel("acoustic distance" if tight else "acoustic distance (Euclidean, 88-D)",
                     fontsize=tick + 1)
    right.set_ylabel("confusion AUC" if tight else "noise-induced\nconfusion AUC",
                     fontsize=tick + 1)
    right.tick_params(labelsize=tick)
    right.set_title(
        f"B  {MODELS[FOCUS_MODEL]}, human" if tight else
        f"B  {MODELS[FOCUS_MODEL]}, human non-speech", fontsize=tick + 2, loc="left",
    )
    right.grid(True, alpha=0.25, linewidth=0.5)
    mantissa, exponent = f"{focus_row['bh_q_value']:.1e}".split("e")
    right.annotate(
        f"$\\rho = {focus_row['spearman_rho']:.3f}$\n"
        f"$q = {mantissa} \\times 10^{{{int(exponent)}}}$",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=tick,
    )
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)

    figure.tight_layout()
    stem = "fig_distance_confusion" + ("" if variant == "full" else f"_{variant}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{stem}.{suffix}")
    plt.close(figure)
    print(f"wrote {FIGURES}/{stem}.{{png,pdf}}")


def main() -> int:
    tests = load_tests()
    focus, focus_row = load_focus()
    print(
        f"{int(tests['bh_rejected_at_0.05'].sum())} BH-significant of {len(tests)}; "
        f"focus {FOCUS_MODEL}/{FOCUS_NOISE} rho={focus_row['spearman_rho']:.4f} "
        f"q={focus_row['bh_q_value']:.2e} over {len(focus)} pairs"
    )
    for variant in VARIANTS:
        render(tests, focus, focus_row, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
