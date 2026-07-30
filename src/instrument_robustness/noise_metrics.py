"""Diagnostics that say WHERE and WHEN a mixture's noise actually sits.

The sweep's headline SNR is a single number over the whole window and the whole spectrum
(`noise_sweep.mix_at_snr`). That number is exact and reproducible, but it is also easy to
misread, in three specific ways this module measures:

  1. WHERE IN FREQUENCY. Low-frequency rumble can dominate total power while barely touching the
     band the instrument occupies. Two conditions labelled "0 dB" can mask the signal completely
     differently. -> `band_snr_db`, `octave_snr_db`
  2. WHEN IN TIME. A door slam is silent for most of the window, so satisfying an average-power
     target amplifies the transient enormously; silence around a short note can also make its
     whole-window SNR misleading. -> `active_fraction`, `active_signal_snr_db`,
     `segmental_snr_db`
  3. WHAT THE MODEL RECEIVES. AST resamples to 16 kHz and discards everything above 8 kHz. For
     broadband noise a large share of the noise power never reaches the model, so the SNR the
     model sees is higher than the one on the label. -> `effective_snr_db`

None of these replaces the requested SNR; they are recorded alongside it so a paper can report what
the number means rather than only what it was set to. All are computed from `clean` and the ADDED
component `added = noisy - clean`, which is exact because mixing is additive.

Torch-free and dependency-light (numpy + scipy) so it can be unit-tested anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from instrument_robustness.config import (
    INSTRUMENT_BAND_HZ,
    NOISE_ACTIVE_TOP_DB,
    SEGMENTAL_FRAME,
    SEGMENTAL_HOP,
    SIGNAL_ACTIVE_TOP_DB,
    SR,
)

# Octave centres spanning the 22.05 kHz band. The lowest instrument fundamental here is MIDI 22
# (~29 Hz, tuba) and the highest is MIDI 103 (~2489 Hz, violin), so 31.25-4000 Hz covers every
# fundamental and 8000 Hz carries the harmonics that distinguish timbre.
OCTAVE_CENTERS_HZ = (31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

_EPS = 1e-20


def _as_float64(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a mono waveform, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("expected a non-empty waveform")
    return array


def _power_ratio_db(signal_power: float, noise_power: float) -> float:
    """10*log10(signal/noise), floored so an empty noise band cannot produce inf."""
    return float(10.0 * np.log10((signal_power + _EPS) / (noise_power + _EPS)))


def band_power(signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    """Mean power of `signal` inside [low_hz, high_hz], via Parseval on the real FFT.

    Postcondition: returns a non-negative float. Summing `band_power` over disjoint bands that
    cover 0..Nyquist recovers the whole-signal mean power to floating-point precision, which is the
    property that makes band SNRs comparable to the headline SNR.
    """
    array = _as_float64(signal)
    spectrum = np.fft.rfft(array)
    freqs = np.fft.rfftfreq(array.size, d=1.0 / sample_rate)
    # Parseval for rfft: double the interior bins, leave DC and (for even n) Nyquist alone.
    weights = np.full(spectrum.size, 2.0)
    weights[0] = 1.0
    if array.size % 2 == 0:
        weights[-1] = 1.0
    power = weights * np.abs(spectrum) ** 2 / array.size**2
    inside = (freqs >= low_hz) & (freqs <= high_hz)
    return float(power[inside].sum())


def band_snr_db(
    clean: np.ndarray,
    added: np.ndarray,
    *,
    sample_rate: int = SR,
    band: tuple[float, float] = INSTRUMENT_BAND_HZ,
) -> float:
    """SNR restricted to one frequency band.

    This is the number to quote when asking "did the noise actually cover the instrument?". It can
    differ from the whole-spectrum SNR by many dB for spectrally lopsided noise, and equals it (to
    rounding) for ideal white noise measured over the full band.
    """
    low, high = band
    return _power_ratio_db(
        band_power(clean, sample_rate, low, high),
        band_power(added, sample_rate, low, high),
    )


def octave_snr_db(
    clean: np.ndarray,
    added: np.ndarray,
    *,
    sample_rate: int = SR,
    centers: tuple[float, ...] = OCTAVE_CENTERS_HZ,
) -> list[dict[str, float]]:
    """Per-octave SNR profile: where along the spectrum the masking actually happens.

    Postcondition: one record per centre frequency whose band lies below Nyquist, each with
    `center_hz`, `low_hz`, `high_hz`, `snr_db`, and `clean_share` -- the fraction of the clean
    signal's total power that falls in that band. Bands are the standard 1/1-octave limits
    (centre / sqrt(2), centre * sqrt(2)).

    `clean_share` is not decoration. A band the instrument does not occupy has near-zero clean power,
    so its SNR is an arbitrarily large negative number that says nothing about audibility. Callers
    must weight or filter by it -- see `worst_octave`.
    """
    nyquist = sample_rate / 2.0
    clean_total = band_power(clean, sample_rate, 0.0, nyquist)
    profile: list[dict[str, float]] = []
    for center in centers:
        low = center / np.sqrt(2.0)
        high = center * np.sqrt(2.0)
        if low >= nyquist:
            continue
        high = min(high, nyquist)
        clean_band = band_power(clean, sample_rate, low, high)
        profile.append(
            {
                "center_hz": float(center),
                "low_hz": float(low),
                "high_hz": float(high),
                "snr_db": _power_ratio_db(
                    clean_band,
                    band_power(added, sample_rate, low, high),
                ),
                "clean_share": float(clean_band / (clean_total + _EPS)),
            }
        )
    return profile


# A band must hold at least this share of the clean signal's power to count as one the instrument
# occupies. Below it, the band's SNR is dominated by numerical floor rather than by masking.
MIN_CLEAN_SHARE = 0.01


def worst_octave(
    profile: list[dict[str, float]],
    *,
    min_clean_share: float = MIN_CLEAN_SHARE,
) -> dict[str, float]:
    """The occupied octave band where the signal is most buried.

    Postcondition: the record with the lowest `snr_db` among bands holding at least
    `min_clean_share` of the clean power; if no band clears the threshold, the band holding the most
    clean power (never an unoccupied one).

    WHY THE FILTER IS LOAD-BEARING. Without it this returns whichever octave the instrument happens
    not to occupy: a 440 Hz tone reports -151 dB at 31 Hz, which is true, meaningless, and identical
    for every noise type. Filtering to occupied bands makes it the sentence a reader needs -- "at
    nominal 0 dB, the 500 Hz octave carrying most of the signal was actually at -9 dB".
    """
    if not profile:
        raise ValueError("empty octave profile")
    occupied = [record for record in profile if record["clean_share"] >= min_clean_share]
    if not occupied:
        return max(profile, key=lambda record: record["clean_share"])
    return min(occupied, key=lambda record: record["snr_db"])


def _frames(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Non-padded framing; returns (n_frames, frame). Empty if the signal is shorter than a frame."""
    array = _as_float64(signal)
    if array.size < frame:
        return np.empty((0, frame), dtype=np.float64)
    starts = range(0, array.size - frame + 1, hop)
    return np.stack([array[start : start + frame] for start in starts])


def active_fraction(
    signal: np.ndarray,
    *,
    top_db: float = NOISE_ACTIVE_TOP_DB,
    frame: int = SEGMENTAL_FRAME,
    hop: int = SEGMENTAL_HOP,
) -> float:
    """Fraction of frames within `top_db` of the signal's own loudest frame.

    Postcondition: a float in [0, 1]; 0.0 for an all-zero signal.

    This is what separates continuous ambience from a transient event. Stationary noise sits near
    1.0; a single door slam in a 3-second window sits near 0.05. The same operation may be applied
    independently to the clean instrument; it is an energy-derived activity estimate, not a human
    annotation.
    """
    framed = _frames(signal, frame, hop)
    if framed.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(framed**2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return 0.0
    threshold = peak * (10.0 ** (-abs(top_db) / 20.0))
    return float(np.mean(rms >= threshold))


def active_signal_snr_db(
    clean: np.ndarray,
    added: np.ndarray,
    *,
    frame: int = SEGMENTAL_FRAME,
    hop: int = SEGMENTAL_HOP,
    top_db: float = SIGNAL_ACTIVE_TOP_DB,
) -> dict[str, float]:
    """SNR over frames where the clean instrument is active.

    A clean frame is active when its RMS is within `top_db` of the clean clip's loudest frame.
    This is deliberately independent of noise activity: it answers "how masked is the note while
    the note is sounding?" and complements `segmental_snr_db`, which selects noise-active frames.

    Postcondition: `{"snr_db", "active_fraction", "n_frames", "n_active_frames"}`. An all-zero or
    shorter-than-one-frame clean signal returns finite zeros because provenance must remain
    serialisable; normal pipeline windows are neither case.
    """
    clean_frames = _frames(clean, frame, hop)
    added_frames = _frames(added, frame, hop)
    empty = {
        "snr_db": 0.0,
        "active_fraction": 0.0,
        "n_frames": 0,
        "n_active_frames": 0,
    }
    if clean_frames.size == 0 or added_frames.size == 0:
        return empty
    count = min(len(clean_frames), len(added_frames))
    signal_power = np.mean(clean_frames[:count] ** 2, axis=1)
    noise_power = np.mean(added_frames[:count] ** 2, axis=1)
    signal_rms = np.sqrt(signal_power)
    peak = signal_rms.max()
    if peak <= 0:
        return {**empty, "n_frames": int(count)}
    active = signal_rms >= peak * (10.0 ** (-abs(top_db) / 20.0))
    return {
        "snr_db": _power_ratio_db(
            float(np.mean(signal_power[active])),
            float(np.mean(noise_power[active])),
        ),
        "active_fraction": float(np.mean(active)),
        "n_frames": int(count),
        "n_active_frames": int(active.sum()),
    }


def segmental_snr_db(
    clean: np.ndarray,
    added: np.ndarray,
    *,
    frame: int = SEGMENTAL_FRAME,
    hop: int = SEGMENTAL_HOP,
    top_db: float = NOISE_ACTIVE_TOP_DB,
) -> dict[str, float]:
    """Per-frame SNR, summarised over the frames where the noise is actually sounding.

    Postcondition: `{"min", "p05", "p50", "p95", "std", "n_frames", "n_active_frames"}`. `min` is
    over every frame; the percentiles and `std` are over ACTIVE-NOISE frames only.

    WHY THE PERCENTILES EXCLUDE SILENT-NOISE FRAMES. A 30 ms burst in a 3-second window leaves ~99%
    of frames with no added noise at all, where per-frame SNR is effectively infinite. Percentiles
    over all frames then describe the silence, not the burst -- measured: p05 came out at +161 dB for
    a transient whose burst frames were far below 0 dB. Restricting to active frames answers the
    question item 5 of the audit actually asks: when this noise is making sound, how buried is the
    instrument? `min` is kept over all frames as the unconditional worst case.

    A stationary mixture has nearly all frames active and a narrow spread near the requested SNR. A
    transient one has few active frames and a wide spread -- which is the signature of an average
    power target met by a brief loud event.
    """
    clean_frames = _frames(clean, frame, hop)
    added_frames = _frames(added, frame, hop)
    empty = {
        "min": 0.0,
        "p05": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "std": 0.0,
        "n_frames": 0,
        "n_active_frames": 0,
    }
    if clean_frames.size == 0 or added_frames.size == 0:
        return empty
    count = min(len(clean_frames), len(added_frames))
    signal_power = np.mean(clean_frames[:count] ** 2, axis=1)
    noise_power = np.mean(added_frames[:count] ** 2, axis=1)
    per_frame = 10.0 * np.log10((signal_power + _EPS) / (noise_power + _EPS))

    noise_rms = np.sqrt(noise_power)
    peak = noise_rms.max()
    if peak <= 0:
        return {**empty, "n_frames": int(count)}
    active = noise_rms >= peak * (10.0 ** (-abs(top_db) / 20.0))
    selected = per_frame[active] if active.any() else per_frame
    low, mid, high = np.percentile(selected, [5, 50, 95])
    return {
        "min": float(per_frame.min()),
        "p05": float(low),
        "p50": float(mid),
        "p95": float(high),
        "std": float(np.std(selected)),
        "n_frames": int(count),
        "n_active_frames": int(active.sum()),
    }


def dc_offset(signal: np.ndarray) -> dict[str, float]:
    """Mean value of a signal and the share of its mean power that the offset accounts for.

    Postcondition: `{"offset", "power_share"}`. `power_share` is `mean(x)^2 / mean(x^2)`, i.e. the
    fraction of measured power that is DC rather than signal.

    WHY THIS IS RECORDED RATHER THAN REMOVED. A constant offset inflates mean power, and mean power
    is exactly what sets the noise gain for every SNR -- so a DC-heavy noise clip would be scaled
    down and end up quieter than its label claims. Measured on this dataset the effect is far below
    the generator's 0.1 dB tolerance:

        clean windows (400 sampled):  worst DC power share 1.06e-04  ->  4.59e-04 dB SNR error
        Gaussian draws (200):         worst DC power share 1.76e-04  ->  7.64e-04 dB SNR error

    So no subtraction is applied: it would alter the corpus's actual content to correct an error
    ~130x smaller than the tolerance, and silently transforming the noise is worse than measuring it.
    ESC-50 clips were NOT part of that audit (the corpus was absent), and real recordings can carry a
    genuine offset from AC coupling -- which is why this is computed per mixture at generation time
    and checked, rather than assumed.
    """
    array = _as_float64(signal)
    offset = float(array.mean())
    power = float(np.mean(array**2))
    return {
        "offset": offset,
        "power_share": float(offset**2 / (power + _EPS)),
    }


def effective_snr_db(
    clean: np.ndarray,
    added: np.ndarray,
    *,
    source_sr: int = SR,
    target_sr: int,
) -> float:
    """SNR after resampling both components to a model's input rate.

    Resampling is linear, so the added component may be resampled directly rather than
    reconstructing and re-differencing the mixture.

    WHY THIS DIFFERS FROM THE LABEL. Downsampling to `target_sr` low-passes at target_sr/2. For
    broadband noise against a band-limited instrument, proportionally more noise than signal is
    discarded, so the model receives a HIGHER SNR than the condition name claims -- AST at 16 kHz
    throws away everything above 8 kHz. Reporting the label alone overstates how corrupted the
    pretrained models' inputs were.
    """
    if target_sr == source_sr:
        return _power_ratio_db(
            float(np.mean(_as_float64(clean) ** 2)),
            float(np.mean(_as_float64(added) ** 2)),
        )
    divisor = np.gcd(source_sr, target_sr)
    up, down = target_sr // divisor, source_sr // divisor
    clean_rs = resample_poly(_as_float64(clean), up, down)
    added_rs = resample_poly(_as_float64(added), up, down)
    return _power_ratio_db(
        float(np.mean(clean_rs**2)),
        float(np.mean(added_rs**2)),
    )


def mixture_diagnostics(
    clean: np.ndarray,
    added: np.ndarray,
    *,
    sample_rate: int = SR,
    model_rates: dict[str, int] | None = None,
) -> dict[str, object]:
    """Every per-mixture diagnostic, flat and ready for a provenance CSV row.

    Preconditions: `clean` and `added` are the same length, `added = noisy - clean`.
    Postcondition: a flat dict of scalars, plus `snr_octave_db` as a compact list. Keys are stable;
    downstream code selects columns by name.

    Recorded at generation time on purpose: none of it can be reconstructed later without
    regenerating the audio, which is the mistake that made the ESC-50 category loss unrecoverable.
    """
    clean = _as_float64(clean)
    added = _as_float64(added)
    if clean.shape != added.shape:
        raise ValueError(f"clean {clean.shape} and added {added.shape} differ in length")
    rates = {"ast_16k": 16000, "mert_24k": 24000, "panns_32k": 32000}
    if model_rates is not None:
        rates = model_rates

    profile = octave_snr_db(clean, added, sample_rate=sample_rate)
    worst = worst_octave(profile)
    active_signal = active_signal_snr_db(clean, added)
    segmental = segmental_snr_db(clean, added)
    noise_dc = dc_offset(added)
    diagnostics: dict[str, object] = {
        "snr_band_db": band_snr_db(clean, added, sample_rate=sample_rate),
        "snr_band_low_hz": float(INSTRUMENT_BAND_HZ[0]),
        "snr_band_high_hz": float(INSTRUMENT_BAND_HZ[1]),
        "snr_worst_octave_db": worst["snr_db"],
        "snr_worst_octave_center_hz": worst["center_hz"],
        "snr_octave_db": [round(record["snr_db"], 4) for record in profile],
        "noise_active_fraction": active_fraction(added),
        "signal_active_fraction": active_signal["active_fraction"],
        "snr_signal_active_db": active_signal["snr_db"],
        "snr_signal_active_frames": active_signal["n_active_frames"],
        "noise_dc_offset": noise_dc["offset"],
        "noise_dc_power_share": noise_dc["power_share"],
        "snr_segmental_min_db": segmental["min"],
        "snr_segmental_p05_db": segmental["p05"],
        "snr_segmental_p50_db": segmental["p50"],
        "snr_segmental_p95_db": segmental["p95"],
        "snr_segmental_std_db": segmental["std"],
        "snr_segmental_frames": segmental["n_frames"],
        "snr_segmental_active_frames": segmental["n_active_frames"],
    }
    for name, rate in rates.items():
        diagnostics[f"snr_effective_{name}_db"] = effective_snr_db(
            clean, added, source_sr=sample_rate, target_sr=rate
        )
    return diagnostics


DIAGNOSTIC_COLUMNS = (
    "snr_band_db",
    "snr_band_low_hz",
    "snr_band_high_hz",
    "snr_worst_octave_db",
    "snr_worst_octave_center_hz",
    "snr_octave_db",
    "noise_active_fraction",
    "signal_active_fraction",
    "snr_signal_active_db",
    "snr_signal_active_frames",
    "noise_dc_offset",
    "noise_dc_power_share",
    "snr_segmental_min_db",
    "snr_segmental_p05_db",
    "snr_segmental_p50_db",
    "snr_segmental_p95_db",
    "snr_segmental_std_db",
    "snr_segmental_frames",
    "snr_segmental_active_frames",
    "snr_effective_ast_16k_db",
    "snr_effective_mert_24k_db",
    "snr_effective_panns_32k_db",
)
