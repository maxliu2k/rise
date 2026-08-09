"""Retention-vs-SNR curves for the 5-page paper, in three layouts.

    python scripts/fig6b_retention_row.py            # all three variants
    python scripts/fig6b_retention_row.py --variant compact

Writes, per variant, docs/figures/<stem>.{png,pdf}. Read-only with respect to results.

    row      fig6b_retention_row       9.23 x 3.07 in, three panels wide, spread bands
    col      fig6c_retention_col       3.42 x 4.76 in, one IEEE column, panels stacked
    compact  fig6d_retention_compact   7.12 x 2.23 in, full \\textwidth  <-- THE PAPER USES THIS

WHY THIS EXISTS ALONGSIDE fig6. fig6_robustness_curves is two rows -- absolute macro-F1 on top,
retention below -- and at IEEE \\textwidth it costs 43% of a page. The paper's metric is retention:
Eq.4 defines it, Eq.5 integrates it, and the results table reports its AUC. Nothing in the paper
reads the absolute row. Dropping it halves the figure and removes the only place where two
different y-quantities share one caption.

WHY THE COMPACT VARIANT EXISTS AT ALL. `fig6d_retention_compact.png` was committed as a binary in
a2c7c9e ("committing it so it can be uploaded to Overleaf") with no code behind it, so the paper's
Figure 1 could not be regenerated or checked against the data. Its contents were correct -- the
values were verified against noise_sweep_summary.csv by hand -- but REPOSITORY_AUDIT.md lists "a
plot regenerated from a different config than the numbers printed beside it" as one of this repo's
own historical bugs, and an uncheckable figure is how that recurs.

NO SPREAD BANDS IN THE COMPACT VARIANT. The paper's caption says "Each point is the mean of the
two noise realizations", and at a median between-draw range of 0.5 percentage points the band is
thinner than the line. Drawing it would be invisible and would contradict the caption. `row` keeps
the band because at 9.2 inches it is legible and that figure is not the one in the paper.

WHAT IS LOST, STATED HONESTLY. The chance line (1/12) is meaningful on absolute macro-F1 and
meaningless on retention, so it is not drawn. A reader can no longer see that SVM approaches
chance at low SNR. fig6 remains in the repository for the poster and for anyone who wants it.

Panel titles use the paper's category names (white / human non-speech / environmental) rather than
the code tags (white / audience / studio). The code tags are handles; `audience` is ESC-50 human
non-speech events and `studio` is DEMAND ambience from 18 environments, most of which are not
studios. docs/FAILURE_ANALYSIS_PLAN.md asks for the descriptive names in paper text.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from instrument_robustness.config import NOISE_TYPES, SNRS  # noqa: E402

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

# stem, width in, height in, stacked panels, draw spread band, base font, legend columns
VARIANTS: dict[str, tuple[str, float, float, bool, bool, float, int]] = {
    "row":     ("fig6b_retention_row",     9.23, 3.07, False, True,  9.0, 6),
    "col":     ("fig6c_retention_col",     3.42, 4.76, True,  False, 7.0, 3),
    "compact": ("fig6d_retention_compact", 7.12, 2.23, False, False, 8.0, 6),
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
    """Mean and spread of retention across replicates at each SNR.

    Postcondition: one row per SNR, ascending, with `mean` and `std` over the replicates. The
    paper's caption describes `mean`; `std` is only drawn by variants whose flag says so.
    """
    subset = frame[frame["noise_type"] == noise_type]
    return (
        subset.groupby("snr_db")["macro_f1_retention"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("snr_db")
    )


def render(frames: dict[str, pd.DataFrame], variant: str) -> Path:
    """Draw one layout and write both formats. Returns the PNG path."""
    stem, width, height, stacked, bands, base_font, legend_cols = VARIANTS[variant]

    import matplotlib
    matplotlib.use("Agg")
    import poster_style; poster_style.apply()
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": base_font, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
    })
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    rows, columns = (len(NOISE_TYPES), 1) if stacked else (1, len(NOISE_TYPES))
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, height),
        sharey=not stacked, sharex=stacked,
    )
    handles = None
    for position, noise_type in enumerate(NOISE_TYPES):
        axis = axes[position]
        for index, (name, frame) in enumerate(frames.items()):
            grouped = curve(frame, noise_type)
            colour = colours[index % len(colours)]
            axis.plot(grouped["snr_db"], grouped["mean"], marker="o", ms=3, lw=1.4,
                      color=colour, label=MODELS[name])
            if bands:
                spread = grouped["std"].fillna(0.0)
                if float(spread.max()) > 0:
                    axis.fill_between(grouped["snr_db"], grouped["mean"] - spread,
                                      grouped["mean"] + spread, color=colour, alpha=0.15, lw=0)
        axis.set_title(PAPER_LABEL.get(noise_type, noise_type), fontsize=base_font + 1)
        axis.set_ylim(0, 1.05)
        # Pin ticks to SNRs that were actually sampled. Left to matplotlib, the compact variant
        # picks 0/20/40 and a reader cannot tell where the grid starts or that it reaches -10.
        # -5 is omitted because it collides with -10 at this width; the marker still shows it.
        axis.set_xticks([s for s in SNRS if s != -5])
        if stacked:
            axis.set_ylabel("retention")
        if not stacked or position == len(NOISE_TYPES) - 1:
            axis.set_xlabel("SNR (dB)")
        if handles is None:
            handles, _ = axis.get_legend_handles_labels()

    if not stacked:
        axes[0].set_ylabel("retention")
    # Legend OUTSIDE the axes. In fig6 it sits inside the third panel, where it overlaps the
    # curves -- fine on a 36x48 poster, not fine at column width.
    figure.legend(handles, list(MODELS.values()), loc="lower center", ncol=legend_cols,
                  frameon=False, fontsize=base_font, bbox_to_anchor=(0.5, -0.06))
    figure.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{stem}.{suffix}")
    plt.close(figure)
    return FIGURES / f"{stem}.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append",
                        help="render only this variant; repeatable. Default: all three.")
    args = parser.parse_args()

    frames = {name: load(name) for name in MODELS}
    for variant in (args.variant or sorted(VARIANTS)):
        path = render(frames, variant)
        print(f"wrote {path.with_suffix('')}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
