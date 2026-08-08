"""Per-instrument recall loss, three heatmap panels, for the 5-page paper (paper Figure 2).

    python scripts/fig_recall_loss_heat.py

Writes docs/figures/fig_recall_loss_heat{,_compact}.{png,pdf}. Read-only with respect to results.

WHAT IT SHOWS. Rows are the 12 instruments, columns the six models, one panel per noise category,
colour is recall-loss AUC across the SNR sweep (clean recall minus noisy recall, integrated by
Eq.5's trapezoid rule). It is the figure behind the four claims in the paper's Instrument-Specific
Failures subsection, which currently cites no figure at all.

ONE SHARED COLOUR SCALE, FIXED AT [0, 1], NOT PER-PANEL AUTOSCALING. This is the only choice here
that can silently produce a wrong reading. Recall-loss AUC is bounded [0,1] by construction and
the observed cells run 0.023 to 0.992, but the three categories occupy very different parts of
that range: white noise reaches 0.992 while human non-speech tops out at 0.623. Autoscaled panels
would paint both maxima the same colour and tell the reader that human non-speech is as damaging
as white noise -- the exact opposite of the paper's finding. Shared limits are what make the
panels comparable, which is the whole point of putting them side by side.

ROW ORDER IS FIXED ACROSS PANELS. Instruments are sorted once, by recall-loss AUC pooled over all
models and noise categories, and that order is reused in every panel. Sorting each panel
separately would let the same row mean a different instrument in each one.

NO PER-CELL NUMBERS. 12 x 6 x 3 is 216 cells; at IEEE \\textwidth each is about 0.35 x 0.2 in, and
two-decimal text in there is unreadable in print. The colorbar carries the magnitude and the
paper quotes the specific values it needs in prose.

Panel titles use the paper's category names (white / human non-speech / environmental) rather than
the code tags (white / audience / studio), matching fig6b_retention_row.py.
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
SUMMARY = ARTIFACTS / "failure_analysis" / "instrument_recall_loss_summary.csv"

# Same keys and order as fig6b_retention_row.MODELS, so a column here is the same model as a
# curve there.
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
PRETTY_INSTRUMENT = {
    "double-bass": "Double bass",
    "french-horn": "French horn",
}

# Recall-loss AUC is bounded [0,1]; pinning the scale keeps the three panels comparable and keeps
# this render comparable with any future one.
VMIN, VMAX = 0.0, 1.0


def pretty(label: str) -> str:
    """Instrument label as the paper writes it."""
    return PRETTY_INSTRUMENT.get(label, label.capitalize())


def load() -> pd.DataFrame:
    """The per-instrument recall-loss summary, checked for full coverage.

    Preconditions: failure_analysis has been run.
    Postcondition: one row per (model, instrument, noise_type) for every combination the figure
    draws, every value in [0,1], every cell averaged over both replicates.
    Raises: FileNotFoundError if the analysis has not been run; ValueError on any missing
    combination, unexpected replicate count, or out-of-range value -- a hole in this grid would
    otherwise render as a blank cell that looks like a real measurement of zero.
    """
    if not SUMMARY.is_file():
        raise FileNotFoundError(f"no recall-loss summary: {SUMMARY}")
    frame = pd.read_csv(SUMMARY)

    instruments = sorted(frame["label"].unique())
    expected = len(MODELS) * len(instruments) * len(NOISE_TYPES)
    if len(frame) != expected:
        raise ValueError(f"{SUMMARY}: expected {expected} rows, found {len(frame)}")

    present = set(zip(frame["model"], frame["label"], frame["noise_type"]))
    missing = [
        (m, i, n)
        for m in MODELS
        for i in instruments
        for n in NOISE_TYPES
        if (m, i, n) not in present
    ]
    if missing:
        raise ValueError(f"{SUMMARY} is missing {len(missing)} combinations, e.g. {missing[:3]}")

    replicates = set(frame["n_replicates"].unique())
    if replicates != {2}:
        raise ValueError(f"{SUMMARY}: expected 2 replicates everywhere, found {sorted(replicates)}")

    values = frame["mean_recall_loss_auc"]
    if values.min() < VMIN or values.max() > VMAX:
        raise ValueError(
            f"{SUMMARY}: recall-loss AUC outside [{VMIN}, {VMAX}]: "
            f"{values.min():.4f} to {values.max():.4f}"
        )
    return frame


def instrument_order(frame: pd.DataFrame) -> list[str]:
    """Instruments worst-first, pooled over every model and noise category.

    Postcondition: one entry per instrument, descending by mean recall-loss AUC. Used for every
    panel so a given row is the same instrument throughout.
    """
    pooled = frame.groupby("label")["mean_recall_loss_auc"].mean().sort_values(ascending=False)
    return list(pooled.index)


def grid(frame: pd.DataFrame, noise_type: str, instruments: list[str]) -> pd.DataFrame:
    """One panel's matrix: instruments (rows) by models (columns)."""
    subset = frame[frame["noise_type"] == noise_type]
    matrix = subset.pivot(index="label", columns="model", values="mean_recall_loss_auc")
    return matrix.reindex(index=instruments, columns=list(MODELS))


# width in inches, height in inches, tick font, panel-title font, rotate model labels upright
VARIANTS: dict[str, tuple[float, float, float, float, bool]] = {
    "full": (7.16, 3.00, 8.0, 10.0, False),
    "compact": (7.16, 2.40, 6.5, 9.0, False),
    # Single IEEE column. Three panels in 3.5 in leaves ~0.9 in each, so model names go
    # vertical and the panel titles drop to one word; the caption carries the full names.
    "column": (3.50, 2.65, 5.0, 6.0, True),
}
SHORT_TITLE = {"white": "white", "audience": "human", "studio": "environ."}


def render(frame: pd.DataFrame, instruments: list[str], variant: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import poster_style; poster_style.apply()
    import matplotlib.pyplot as plt

    width, height, tick_size, title_size, upright = VARIANTS[variant]

    plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": False})

    figure, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(width, height), sharey=True)
    image = None
    for column, noise_type in enumerate(NOISE_TYPES):
        axis = axes[column]
        matrix = grid(frame, noise_type, instruments)
        image = axis.imshow(
            matrix.to_numpy(), cmap="magma_r", vmin=VMIN, vmax=VMAX,
            aspect="auto", interpolation="nearest",
        )
        title = SHORT_TITLE[noise_type] if upright else PAPER_LABEL.get(noise_type, noise_type)
        axis.set_title(title, fontsize=title_size)
        axis.set_xticks(range(len(MODELS)))
        axis.set_xticklabels(
            list(MODELS.values()),
            rotation=90 if upright else 45,
            ha="center" if upright else "right",
            fontsize=tick_size,
        )
        axis.set_yticks(range(len(instruments)))
        if column == 0:
            axis.set_yticklabels([pretty(i) for i in instruments], fontsize=tick_size)
        # Hairlines between cells so adjacent similar values stay distinguishable.
        axis.set_xticks([x - 0.5 for x in range(1, len(MODELS))], minor=True)
        axis.set_yticks([y - 0.5 for y in range(1, len(instruments))], minor=True)
        axis.grid(which="minor", color="white", linewidth=0.4 if upright else 0.5)
        axis.tick_params(which="minor", length=0)
        axis.tick_params(which="major", length=1.5, pad=1.5)
        for spine in axis.spines.values():
            spine.set_visible(False)

    bar = figure.colorbar(
        image, ax=axes, orientation="vertical",
        fraction=0.035 if upright else 0.02, pad=0.02,
    )
    label = "recall-loss AUC" if upright else "recall-loss AUC (clean $-$ noisy)"
    bar.set_label(label, fontsize=tick_size + 0.5)
    bar.ax.tick_params(labelsize=tick_size)
    bar.outline.set_visible(False)

    stem = "fig_recall_loss_heat" + ("" if variant == "full" else f"_{variant}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{stem}.{suffix}")
    plt.close(figure)
    print(f"wrote {FIGURES}/{stem}.{{png,pdf}}")


def main() -> int:
    frame = load()
    instruments = instrument_order(frame)
    print(f"instrument order (worst first): {', '.join(instruments)}")
    for variant in VARIANTS:
        render(frame, instruments, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
