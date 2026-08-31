"""Measure the in-band SNR each noise category actually achieved, and re-express every
model's retention curve on that axis instead of the nominal full-band one.

WHY THIS EXISTS. The headline x-axis is nominal full-band SNR, which fixes total power and
says nothing about WHERE that power sits. The instruments occupy INSTRUMENT_BAND_HZ; white
noise is flat across the whole band to Nyquist, while ESC-50 events and DEMAND ambience are
not. So "white is harsher than recorded noise at matched nominal SNR" has an unmeasured
competing explanation: white may simply be putting more of its energy where the notes live.
The paper's Limitations already concedes the categories were never equalised for in-band
power. This module measures it.

PRE-REGISTERED INTERPRETATION -- written before the first run, per the evidence rules.

Let G_nominal be the gap between the white and human category-mean retention AUC on the
nominal axis. From artifacts/<model>/noise/, that gap is currently 0.714 - 0.427 = 0.287.
Recompute each model's retention against MEASURED mean in-band SNR, interpolate every
category onto the in-band range all three share, and let G_inband be the same gap there.

  * G_inband > 0.14 (more than half the nominal gap survives), ordering unchanged
      -> The category effect is not an artifact of the axis. The abstract's "degradation
         depends on noise category ... not severity alone" STANDS and is strengthened:
         equal in-band power still does not mean equal damage.

  * G_inband < 0.10 (less than a third survives), or the ordering inverts
      -> The nominal axis was doing most of the work. The category claim must be QUALIFIED
         to "at matched nominal full-band SNR" everywhere it appears, and the abstract's
         "not severity alone" RETRACTED. Report both axes.

  * 0.10 <= G_inband <= 0.14
      -> Partially explained. Report both axes; keep the claim only with the in-band
         correction stated alongside it.

Do not edit these thresholds after seeing the output.

WHAT THIS MODULE DOES NOT DO. It does not retrain, re-score, or regenerate audio. It reads
the committed mixtures and the committed per-condition metrics. Retention comes from the same
metrics_*.json the paper's Table II was built from, so the only new quantity is the x
coordinate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS,
    INSTRUMENT_BAND_HZ,
    N_REPLICATES,
    NOISE_TYPES,
    SNRS,
    SR,
    assert_fingerprint,
)
from .noise_metrics import band_snr_db
from .noise_sweep import (
    NOISY_DIR,
    load_clean,
    out_path,
    read_audio_window,
    test_windows,
    window_id_of,
)

# Models whose sweeps feed the paper's Table II, mapped to their artifact directory. MERT is
# the fine-tuned run: artifacts/mert/ is the retired frozen probe (clean macro-F1 0.8931) and
# quoting it here would silently disagree with the paper's 0.9798.
MODEL_DIRS = {
    "SVM": "svm",
    "CNN": "cnn",
    "CRNN": "crnn",
    "MERT": "mert_ft",
    "PANNs": "panns",
    "AST": "ast",
}

# A mixture whose measured FULL-band SNR is further than this from its nominal value is not
# the file we think it is. Generation already verified every mixture to 0.1 dB, so anything past
# this margin means a wrong pairing, replicate, or a truncated read -- not noise.
FULLBAND_TOLERANCE_DB = 0.15

OUT_DIR = ARTIFACTS / "inband_snr"


def _fullband_snr_db(clean: np.ndarray, added: np.ndarray) -> float:
    """Whole-spectrum SNR in dB, the quantity the sweep held to its nominal target."""
    clean_power = float(np.mean(np.square(clean, dtype=np.float64)))
    added_power = float(np.mean(np.square(added, dtype=np.float64)))
    if added_power <= 0.0:
        raise ValueError("added noise has zero power; clean and noisy files are identical")
    return 10.0 * np.log10(clean_power / added_power)


def condition_inband_snr(
    noise_type: str,
    snr: int,
    replicate: int,
    window_rows: pd.DataFrame,
    *,
    noisy_dir: str | Path = NOISY_DIR,
) -> dict[str, float]:
    """Mean in-band SNR actually achieved by one (noise type, nominal SNR, replicate).

    Preconditions: `window_rows` holds the sampled test windows, with a `window_path` column
    relative to the data root. The corresponding mixtures exist under `noisy_dir`.
    Postcondition: a dict carrying the mean/std/median in-band SNR in dB over the sampled
    windows, the count actually measured, and the mean full-band SNR as a control.
    Raises: FileNotFoundError if a mixture is missing -- a partial condition must not average
    into a curve as though it were complete. ValueError if a measured full-band SNR is further
    than FULLBAND_TOLERANCE_DB from `snr`, which means the wrong file was read.
    """
    inband: list[float] = []
    fullband: list[float] = []
    for relative_path in window_rows["window_path"]:
        window_id = window_id_of(relative_path)
        noisy_file = out_path(
            noise_type, snr, window_id, replicate=replicate, noisy_dir=noisy_dir
        )
        if not noisy_file.is_file():
            raise FileNotFoundError(
                f"missing mixture {noisy_file}. Run noise_sweep --check-generated; a "
                f"condition measured from an incomplete set is not a measurement."
            )
        clean = load_clean(relative_path)
        noisy = read_audio_window(noisy_file)
        if clean.shape != noisy.shape:
            raise ValueError(
                f"{noisy_file} is {noisy.shape} but its clean window is {clean.shape}"
            )
        added = noisy.astype(np.float64) - clean.astype(np.float64)
        measured_fullband = _fullband_snr_db(clean, added)
        if abs(measured_fullband - snr) > FULLBAND_TOLERANCE_DB:
            raise ValueError(
                f"{noisy_file} measures {measured_fullband:.3f} dB full-band against a "
                f"nominal {snr} dB. The sweep held every mixture to 0.1 dB, so this is the "
                f"wrong file, not noise."
            )
        inband.append(band_snr_db(clean, added, sample_rate=SR, band=INSTRUMENT_BAND_HZ))
        fullband.append(measured_fullband)
    values = np.asarray(inband, dtype=np.float64)
    return {
        "noise_type": noise_type,
        "snr_nominal_db": int(snr),
        "replicate": int(replicate),
        "n_windows": int(values.size),
        "snr_inband_mean_db": float(values.mean()),
        "snr_inband_std_db": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "snr_inband_median_db": float(np.median(values)),
        "snr_inband_sem_db": (
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        ),
        "snr_fullband_mean_db": float(np.mean(fullband)),
        "inband_offset_db": float(values.mean() - snr),
    }


def measure_conditions(
    *,
    sample: int,
    seed: int = 0,
    noisy_dir: str | Path = NOISY_DIR,
) -> pd.DataFrame:
    """In-band SNR for every (noise type, nominal SNR, replicate) in the frozen grid.

    Preconditions: the completed sweep is present under `noisy_dir`.
    Postcondition: one row per condition, covering all of NOISE_TYPES x SNRS x N_REPLICATES.
    The same windows are used for every condition, so category differences cannot come from
    having sampled different notes.
    """
    frame = test_windows()
    if sample > 0 and sample < len(frame):
        rows = frame.sample(n=sample, random_state=seed).sort_index()
    else:
        rows = frame
    records = [
        condition_inband_snr(noise_type, snr, replicate, rows, noisy_dir=noisy_dir)
        for noise_type in NOISE_TYPES
        for snr in SNRS
        for replicate in range(N_REPLICATES)
    ]
    measured = pd.DataFrame.from_records(records)
    expected = len(NOISE_TYPES) * len(SNRS) * N_REPLICATES
    if len(measured) != expected:
        raise ValueError(f"measured {len(measured)} conditions; the grid has {expected}")
    return measured


def load_retention() -> pd.DataFrame:
    """Retention per model, noise type and nominal SNR, from the committed sweep metrics.

    Postcondition: columns model, noise_type, snr_nominal_db, retention. Retention is the
    two-replicate mean of noisy macro-F1 over that model's clean macro-F1, which is exactly
    equation (4) averaged as the paper defines it.
    Raises: FileNotFoundError naming the first missing metrics file. A model with an
    incomplete sweep is excluded loudly, never averaged over whatever happens to be present.
    """
    records = []
    for model, directory in MODEL_DIRS.items():
        noise_dir = ARTIFACTS / directory / "noise"
        clean_file = noise_dir / "metrics_clean.json"
        if not clean_file.is_file():
            raise FileNotFoundError(f"{clean_file} is missing; sync the sweep from SCC")
        clean_payload = json.loads(clean_file.read_text())
        assert_fingerprint(clean_payload.get("config_fingerprint"), str(clean_file))
        clean_f1 = float(clean_payload.get("test_metrics", clean_payload)["macro_f1"])
        for noise_type in NOISE_TYPES:
            for snr in SNRS:
                scores = []
                for replicate in range(N_REPLICATES):
                    path = noise_dir / f"metrics_{noise_type}_{snr}_r{replicate}.json"
                    if not path.is_file():
                        raise FileNotFoundError(f"{path} is missing; sync the sweep from SCC")
                    payload = json.loads(path.read_text())
                    scores.append(float(payload.get("test_metrics", payload)["macro_f1"]))
                records.append(
                    {
                        "model": model,
                        "noise_type": noise_type,
                        "snr_nominal_db": int(snr),
                        "retention": float(np.mean(scores)) / clean_f1,
                    }
                )
    return pd.DataFrame.from_records(records)


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal area under y(x), divided by the span of x -- the paper's equation (5)."""
    order = np.argsort(x)
    x_sorted, y_sorted = np.asarray(x)[order], np.asarray(y)[order]
    span = float(x_sorted[-1] - x_sorted[0])
    if span <= 0:
        raise ValueError("x span is zero; cannot normalize an AUC over it")
    return float(np.trapezoid(y_sorted, x_sorted)) / span


def auc_on_common_inband_grid(
    retention: pd.DataFrame,
    measured: pd.DataFrame,
    *,
    n_points: int = 25,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recompute each model-category retention AUC over the in-band range all categories share.

    Preconditions: `retention` and `measured` cover the same grid.
    Postcondition: (per-model-and-category AUC frame, summary dict carrying the common
    in-band range and the white-vs-human category-mean gap the pre-registration turns on).

    The common range matters. Each category reaches a different in-band SNR at the same
    nominal value, so comparing raw areas over each category's own range would reintroduce
    exactly the axis problem this is meant to remove. Every curve is interpolated onto the
    overlap and integrated there.
    """
    axis = (
        measured.groupby(["noise_type", "snr_nominal_db"], as_index=False)["snr_inband_mean_db"]
        .mean()
    )
    curves = retention.merge(axis, on=["noise_type", "snr_nominal_db"], validate="many_to_one")
    low = max(
        curves.loc[curves["noise_type"] == t, "snr_inband_mean_db"].min() for t in NOISE_TYPES
    )
    high = min(
        curves.loc[curves["noise_type"] == t, "snr_inband_mean_db"].max() for t in NOISE_TYPES
    )
    if not high > low:
        raise ValueError(
            f"the three categories share no in-band range (low {low:.2f} >= high {high:.2f}); "
            "they cannot be compared on this axis"
        )
    grid = np.linspace(low, high, n_points)
    rows = []
    for (model, noise_type), block in curves.groupby(["model", "noise_type"]):
        block = block.sort_values("snr_inband_mean_db")
        interpolated = np.interp(
            grid, block["snr_inband_mean_db"].to_numpy(), block["retention"].to_numpy()
        )
        rows.append(
            {
                "model": model,
                "noise_type": noise_type,
                "auc_inband": normalized_auc(grid, interpolated),
                "auc_nominal": normalized_auc(
                    block["snr_nominal_db"].to_numpy(), block["retention"].to_numpy()
                ),
            }
        )
    table = pd.DataFrame.from_records(rows)
    means = table.groupby("noise_type")[["auc_inband", "auc_nominal"]].mean()
    summary = {
        "inband_range_low_db": float(low),
        "inband_range_high_db": float(high),
        "gap_nominal_white_vs_human": float(
            means.loc["audience", "auc_nominal"] - means.loc["white", "auc_nominal"]
        ),
        "gap_inband_white_vs_human": float(
            means.loc["audience", "auc_inband"] - means.loc["white", "auc_inband"]
        ),
    }
    return table, summary


def verdict(summary: dict[str, float]) -> str:
    """Apply the pre-registered thresholds. Returns the branch, not an interpretation."""
    gap = summary["gap_inband_white_vs_human"]
    if gap > 0.14:
        return "STANDS: more than half the nominal category gap survives on the in-band axis"
    if gap < 0.10:
        return "QUALIFY/RETRACT: less than a third of the nominal category gap survives"
    return "PARTIAL: between the pre-registered thresholds; report both axes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sample",
        type=int,
        default=250,
        help="test windows per condition (0 = all 1255). The same windows are used for "
        "every condition, so this trades precision on the x coordinate for wall time.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noisy-dir", default=str(NOISY_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"measuring in-band SNR over {INSTRUMENT_BAND_HZ[0]}-{INSTRUMENT_BAND_HZ[1]} Hz")
    measured = measure_conditions(
        sample=args.sample, seed=args.seed, noisy_dir=args.noisy_dir
    )
    measured.to_csv(out_dir / "condition_inband_snr.csv", index=False)

    print("\nmeasured in-band SNR against nominal (mean over sampled windows):")
    pivot = measured.pivot_table(
        index="snr_nominal_db", columns="noise_type", values="inband_offset_db"
    )
    print(pivot.round(2).to_string())
    print("\n(positive = the category put LESS of its power in the instrument band than a "
          "flat spectrum would, so the nominal axis understates how much signal survives)")

    retention = load_retention()
    table, summary = auc_on_common_inband_grid(retention, measured)
    table.to_csv(out_dir / "auc_inband_vs_nominal.csv", index=False)

    print(
        f"\ncommon in-band range: {summary['inband_range_low_db']:.2f} to "
        f"{summary['inband_range_high_db']:.2f} dB"
    )
    print(table.pivot(index="model", columns="noise_type",
                      values=["auc_nominal", "auc_inband"]).round(3).to_string())
    print(
        f"\nwhite-vs-human category-mean gap: nominal "
        f"{summary['gap_nominal_white_vs_human']:.3f} -> in-band "
        f"{summary['gap_inband_white_vs_human']:.3f}"
    )
    summary["verdict"] = verdict(summary)
    summary["sample_windows"] = int(args.sample)
    print(f"PRE-REGISTERED VERDICT -- {summary['verdict']}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir}/condition_inband_snr.csv, auc_inband_vs_nominal.csv, summary.json")


if __name__ == "__main__":
    main()
