"""One matplotlib style for every poster figure.

    import poster_style; poster_style.apply()

WHY THIS EXISTS. The 2026 poster PDF embeds SIX font faces -- Arial, Arial-Bold, Calibri,
Calibri-Bold, Times New Roman, Times New Roman-Bold -- and every matplotlib figure on it was
rendered in a seventh, DejaVu Sans. Four families on one board is the single most visible
inconsistency on it, and it is invisible to whoever made each piece because each piece looks
fine alone.

Times New Roman is the target: it is the poster's title face, the team standardised the whole
board on it (2026-08-04), and registering the real Windows TTF means figures and body text are
literally the same face rather than two lookalikes. Nothing here changes any DATA; it changes
typeface, sizes and frame weights only.

Font sizes are set for a 36 x 48 in board viewed from about a metre: a figure placed at roughly
one third of the column width needs ~9 pt type in the source to stay readable.
"""
from __future__ import annotations

FAMILY = "Times New Roman"

# Frame and rule colours. Lighter than matplotlib's default black so the data, not the box,
# carries the contrast.
FRAME = "#bbbbbb"
GRID = "#e8e8e8"
TEXT = "#222222"
MUTED = "#5a5a5a"


def apply() -> None:
    """Set rcParams for every figure in this repo. Call before creating a figure."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": FAMILY,
        "font.size": 9.5,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "figure.titlesize": 13,

        "axes.edgecolor": FRAME,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": False,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,          # 200 was soft when a figure is enlarged on a 36x48 board
        "pdf.fonttype": 42,          # embed TrueType so the PDF is not re-rasterised at print
        "ps.fonttype": 42,
    })
