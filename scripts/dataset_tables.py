"""Regenerate every number and figure in docs/DATASET_TABLES.md from the frozen build.

    python scripts/dataset_tables.py                 # print tables, write docs/figures/*
    python scripts/dataset_tables.py --no-figures    # tables only

This script is the documented command behind docs/DATASET_TABLES.md. Nothing in that file may be
hand-edited: if a number there disagrees with this script's output, this script wins and the
document is stale.

It is read-only with respect to the dataset. The only files it writes are under docs/figures/.

Preconditions:
  - all-samples/pipeline/windows.csv exists and its artifact fingerprint matches the current
    config (verified here; the script raises rather than reporting numbers from a stale build).
  - all-samples/manifest.csv exists.
Postconditions:
  - Tables 1-4 are printed to stdout.
  - Unless --no-figures, docs/figures/fig{1..5}_*.{png,pdf} are overwritten.
Raises:
  - AssertionError if the build fails an integrity gate (fingerprint, one-window-per-source,
    split coverage, or pitch-group leakage). A failing gate means the numbers would be wrong, so
    the script must not print them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_robustness.config import (
    MANIFEST_IN,
    TARGET_LABELS,
    WINDOWS_CSV,
    WINDOW_S,
    assert_artifact_fingerprint,
)

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
SPLITS = ("train", "val", "test")
SHORT = {"double-bass": "d-bass", "french-horn": "f-horn"}


def short(label: str) -> str:
    """Abbreviate the two labels that overflow a tick slot. Identity otherwise."""
    return SHORT.get(label, label)


def load_joined(windows_csv: Path) -> pd.DataFrame:
    """Join windows to their source manifest rows and verify the build is trustworthy.

    Postcondition: one row per window, carrying label, split, note, midi, dynamic, technique,
    content_s, pre/post normalisation RMS, and the pitch group id.
    Raises: AssertionError if any integrity gate fails.
    """
    windows = pd.read_csv(windows_csv)
    manifest = pd.read_csv(MANIFEST_IN)
    joined = windows.merge(
        manifest, left_on="source_path", right_on="path", suffixes=("", "_src")
    )
    assert len(joined) == len(windows), (
        f"join lost rows: {len(windows)} windows -> {len(joined)}. A window's source_path is "
        "missing from the manifest, which means the two artifacts came from different builds."
    )

    assert joined.source_path.nunique() == len(joined), (
        "more than one window per source; every table here assumes MAX_WINDOWS_PER_SOURCE = 1"
    )
    assert set(joined.split) == set(SPLITS), f"unexpected splits: {sorted(set(joined.split))}"
    assert set(joined.label) <= set(TARGET_LABELS), "window labels outside TARGET_LABELS"

    joined["grp"] = joined.label + "/" + joined.note
    spanning = joined.groupby("grp").split.nunique()
    assert (spanning == 1).all(), (
        f"{int((spanning > 1).sum())} pitch groups span more than one split -- the no-leak "
        "guarantee is broken and no number below is meaningful"
    )
    return joined


def table1(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument counts by split, distinct notes, and MIDI range."""
    counts = pd.crosstab(joined.label, joined.split)[list(SPLITS)]
    counts["total"] = counts.sum(axis=1)
    counts["notes"] = joined.groupby("label").note.nunique()
    counts["midi_lo"] = joined.groupby("label").midi.min()
    counts["midi_hi"] = joined.groupby("label").midi.max()
    return counts.loc[list(TARGET_LABELS)]


def table2(joined: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Articulation-filter effect: what the single-technique restriction removed."""
    raw = manifest[manifest.label.isin(TARGET_LABELS)]
    raw_counts = raw.label.value_counts()
    kept_counts = joined.label.value_counts()
    return {
        "raw_rows": len(raw),
        "kept_rows": len(joined),
        "dropped": len(raw) - len(joined),
        "dropped_pct": (len(raw) - len(joined)) / len(raw) * 100,
        "raw_techniques": raw.technique.nunique(),
        "kept_techniques": joined.technique.value_counts().to_dict(),
        "raw_imbalance": raw_counts.max() / raw_counts.min(),
        "kept_imbalance": kept_counts.max() / kept_counts.min(),
        "dropped_per_class": (raw_counts.loc[list(TARGET_LABELS)]
                              - kept_counts.loc[list(TARGET_LABELS)]).sort_values(ascending=False),
    }


def table3(joined: pd.DataFrame) -> dict:
    """Distinct source audio per window, i.e. how much of each window is tiled repetition."""
    content = joined.content_s
    return {
        "thresholds": {t: (int((content < t).sum()), float((content < t).mean() * 100))
                       for t in (0.5, 1.0, 1.5, 2.0, WINDOW_S)},
        "untiled": int((content >= WINDOW_S - 1e-3).sum()),
        "median": float(content.median()),
        "mean": float(content.mean()),
        "min": float(content.min()),
        "max": float(content.max()),
        "per_class_median": joined.groupby("label").content_s.median(),
    }


def table4(joined: pd.DataFrame) -> dict:
    """Dynamics, pitch-group structure, and the loudness normalisation outcome."""
    group_sizes = joined.groupby("grp").size()
    guarded = joined[~np.isclose(joined.post_norm_rms, 0.1, atol=1e-6)]
    return {
        "dynamics": joined.dynamic.value_counts(),
        "n_groups": int(joined.grp.nunique()),
        "group_mean": float(group_sizes.mean()),
        "group_median": float(group_sizes.median()),
        "group_min": int(group_sizes.min()),
        "group_max": int(group_sizes.max()),
        "groups_per_split": joined.groupby("split").grp.nunique(),
        "pre_norm_median": joined.groupby("label").pre_norm_rms.median(),
        "n_at_target": int(len(joined) - len(guarded)),
        "n_peak_guarded": int(len(guarded)),
        "guarded_min": float(guarded.post_norm_rms.min()) if len(guarded) else float("nan"),
        "guarded_by_class": guarded.label.value_counts(),
    }


def _setup_matplotlib():
    """Configure the shared figure style and return the pyplot module.

    One place for the house style so five figures cannot drift apart -- a plot that does not match
    the numbers printed beside it is a bug this repo has already shipped once.
    """
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


def _save(plt, name: str) -> None:
    """Write the current figure as PNG (raster, for review) and PDF (vector, for the paper)."""
    plt.savefig(FIGURES / f"{name}.png")
    plt.savefig(FIGURES / f"{name}.pdf")
    plt.close()


def figure1_class_balance(joined: pd.DataFrame, plt) -> None:
    """Recordings per instrument, stacked by split."""
    labels = list(TARGET_LABELS)
    ticks = [short(name) for name in labels]
    counts = pd.crosstab(joined.label, joined.split)[list(SPLITS)].loc[labels]
    _, ax = plt.subplots(figsize=(6.6, 2.9))
    bottom = np.zeros(len(labels))
    for split, colour in zip(SPLITS, ("#3b6ea5", "#7fa6cd", "#c9dcee")):
        ax.bar(ticks, counts[split], bottom=bottom, label=split, color=colour,
               edgecolor="white", linewidth=0.5)
        bottom += counts[split].values
    ax.set_ylabel("recordings")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    ax.set_title(f"Recordings per instrument by split (n = {len(joined):,})",
                 fontsize=9.5, loc="left")
    for index, value in enumerate(counts.sum(axis=1)):
        ax.text(index, value + 12, str(value), ha="center", fontsize=7)
    ax.set_ylim(0, counts.sum(axis=1).max() * 1.14)
    plt.xticks(rotation=35, ha="right")
    _save(plt, "fig1_class_balance")


def figure2_pitch_range(joined: pd.DataFrame, plt) -> None:
    """MIDI range covered per instrument, as a horizontal span per class."""
    labels = list(TARGET_LABELS)
    _, ax = plt.subplots(figsize=(6.6, 3.1))
    span = joined.groupby("label").midi.agg(["min", "max"]).loc[labels]
    positions = np.arange(len(labels))
    ax.barh(positions, span["max"] - span["min"], left=span["min"], height=0.6,
            color="#3b6ea5", alpha=0.75)
    for index, (low, high) in enumerate(zip(span["min"], span["max"])):
        ax.text(low - 1.5, index, str(int(low)), va="center", ha="right", fontsize=6.5)
        ax.text(high + 1.5, index, str(int(high)), va="center", fontsize=6.5)
    ax.set_yticks(positions)
    ax.set_yticklabels([short(name) for name in labels])
    ax.invert_yaxis()
    ax.set_xlabel("MIDI note number")
    ax.set_xlim(joined.midi.min() - 8, joined.midi.max() + 9)
    ax.set_title(f"Pitch range covered per instrument "
                 f"({joined.note.nunique()} distinct pitches, "
                 f"MIDI {joined.midi.min()}–{joined.midi.max()})", fontsize=9.5, loc="left")
    _save(plt, "fig2_pitch_range")


def figure3_articulation_filter(joined: pd.DataFrame, manifest: pd.DataFrame, plt) -> None:
    """Per-instrument counts before and after the single-technique restriction."""
    labels = list(TARGET_LABELS)
    raw = manifest[manifest.label.isin(TARGET_LABELS)]
    all_counts = raw.label.value_counts().loc[labels]
    kept_counts = joined.label.value_counts().loc[labels]
    _, ax = plt.subplots(figsize=(6.6, 2.9))
    positions = np.arange(len(labels))
    width = 0.38
    ax.bar(positions - width / 2, all_counts, width, label="all articulations", color="#c2c2c2")
    ax.bar(positions + width / 2, kept_counts, width,
           label="retained (one per instrument)", color="#3b6ea5")
    ax.set_xticks(positions)
    ax.set_xticklabels([short(name) for name in labels], rotation=35, ha="right")
    ax.set_ylabel("recordings")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Articulation filter: imbalance falls "
                 f"{all_counts.max() / all_counts.min():.2f}:1 → "
                 f"{kept_counts.max() / kept_counts.min():.2f}:1", fontsize=9.5, loc="left")
    _save(plt, "fig3_articulation_filter")


def figure4_window_content(joined: pd.DataFrame, plt) -> None:
    """Distribution of distinct source audio per window, overall and per instrument."""
    labels = list(TARGET_LABELS)
    _, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.32})
    left.hist(joined.content_s, bins=60, color="#3b6ea5", alpha=0.85)
    top = left.get_ylim()[1]
    median = joined.content_s.median()
    left.axvline(WINDOW_S, color="#b03030", ls="--", lw=1.2)
    left.annotate(f"{WINDOW_S:.1f} s window", xy=(0.965, 0.72), xycoords="axes fraction",
                  rotation=90, ha="right", va="top", fontsize=7, color="#b03030")
    left.axvline(median, color="k", ls=":", lw=1)
    left.annotate(f"median {median:.2f} s", xy=(median + 0.08, top * 0.93), fontsize=7)
    left.set_xlabel("distinct source audio per window (s)")
    left.set_ylabel("windows")
    left.set_xlim(0, WINDOW_S + 0.15)
    left.set_ylim(0, top * 1.05)
    left.set_title(f"{(joined.content_s < WINDOW_S).mean() * 100:.1f}% of windows are shorter "
                   f"than {WINDOW_S:.0f} s\nand are filled by tiling", fontsize=9, loc="left")
    per_class = joined.groupby("label").content_s.median().loc[labels].sort_values()
    right.barh(range(len(per_class)), per_class.values, color="#3b6ea5", alpha=0.85, height=0.65)
    right.set_yticks(range(len(per_class)))
    right.set_yticklabels([short(name) for name in per_class.index], fontsize=7.5)
    right.set_xlabel("median distinct audio (s)")
    right.set_title("by instrument", fontsize=9, loc="left")
    _save(plt, "fig4_window_content")


def figure5_loudness(joined: pd.DataFrame, plt) -> None:
    """Per-window RMS before and after Step-5 loudness normalisation."""
    order = joined.groupby("label").pre_norm_rms.median().sort_values().index
    guarded = joined[~np.isclose(joined.post_norm_rms, 0.1, atol=1e-6)]
    _, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.3})
    left.boxplot([joined[joined.label == name].pre_norm_rms for name in order],
                 showfliers=False, patch_artist=True,
                 boxprops=dict(facecolor="#c9dcee", lw=0.6),
                 medianprops=dict(color="#b03030", lw=1.3),
                 whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    left.set_xticklabels([short(name) for name in order], rotation=40, ha="right", fontsize=7.5)
    left.set_ylabel("RMS amplitude")
    medians = joined.groupby("label").pre_norm_rms.median()
    left.set_title(f"Before: median RMS spans {medians.min():.3f} ({short(medians.idxmin())}) "
                   f"to {medians.max():.3f} ({short(medians.idxmax())})", fontsize=9, loc="left")
    right.hist(joined.post_norm_rms, bins=np.linspace(
        min(0.075, joined.post_norm_rms.min()), 0.1005, 60), color="#3b6ea5")
    right.set_yscale("log")
    right.set_xlabel("RMS after normalisation")
    right.set_ylabel("windows (log)")
    right.set_title(f"After: {len(joined) - len(guarded):,} exactly at 0.1;\n{len(guarded)} "
                    "attenuated by the 0.99 peak guard", fontsize=9, loc="left")
    if len(guarded):
        right.annotate(f"min {guarded.post_norm_rms.min():.3f}",
                       xy=(guarded.post_norm_rms.min() + 0.0006, 1.5), fontsize=7)
    _save(plt, "fig5_loudness_normalisation")


def make_figures(joined: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Write all five composition figures. Each figure is built by its own function above."""
    plt = _setup_matplotlib()
    figure1_class_balance(joined, plt)
    figure2_pitch_range(joined, plt)
    figure3_articulation_filter(joined, manifest, plt)
    figure4_window_content(joined, plt)
    figure5_loudness(joined, plt)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-csv", type=Path, default=WINDOWS_CSV)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    # step5_normalize is the last writer of windows.csv; step4_window writes it first.
    assert_artifact_fingerprint(args.windows_csv, ("step4_window", "step5_normalize"))
    joined = load_joined(args.windows_csv)
    manifest = pd.read_csv(MANIFEST_IN)

    print("=== TABLE 1 -- composition by instrument ===")
    print(table1(joined).to_string())
    counts = joined.label.value_counts()
    print(f"total {len(joined)} | imbalance {counts.max() / counts.min():.2f}:1 "
          f"| mean {counts.mean():.1f} +/- {counts.std():.1f}")

    print("\n=== TABLE 2 -- articulation filter ===")
    for key, value in table2(joined, manifest).items():
        print(f"{key}:\n{value}" if isinstance(value, pd.Series) else f"{key}: {value}")

    print("\n=== TABLE 3 -- window content ===")
    stats = table3(joined)
    for threshold, (count, pct) in stats.pop("thresholds").items():
        print(f"< {threshold} s: {count} ({pct:.2f}%)")
    for key, value in stats.items():
        print(f"{key}:\n{value}" if isinstance(value, pd.Series) else f"{key}: {value}")

    print("\n=== TABLE 4 -- dynamics, groups, loudness ===")
    for key, value in table4(joined).items():
        print(f"{key}:\n{value}" if isinstance(value, pd.Series) else f"{key}: {value}")

    if not args.no_figures:
        make_figures(joined, manifest)
        print(f"\nfigures written to {FIGURES}")


if __name__ == "__main__":
    main()
