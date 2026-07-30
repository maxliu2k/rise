"""Tests for the mixture diagnostics (audit items 1, 5, 6) and the replicate axis (item 3).

Each test is written against the FAILURE the diagnostic exists to catch, not against a golden
number, so it stays meaningful if the implementation is rewritten:

  item 1  a nominal 0 dB condition can leave the instrument's band almost untouched
  item 5  a nominal 0 dB condition can be one brief burst in an otherwise clean window
  item 6  a nominal 0 dB condition can arrive at the model much cleaner after resampling
  item 3  replicate must change the draw, and must not change the SNR
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import butter, lfilter

from instrument_robustness.config import (
    INSTRUMENT_BAND_HZ,
    NOISE_TYPES,
    N_REPLICATES,
    SNRS,
    SR,
    TARGET_LABELS,
)
from instrument_robustness.noise_metrics import (
    DIAGNOSTIC_COLUMNS,
    MIN_CLEAN_SHARE,
    active_signal_snr_db,
    active_fraction,
    band_power,
    band_snr_db,
    effective_snr_db,
    mixture_diagnostics,
    octave_snr_db,
    segmental_snr_db,
    worst_octave,
)
from instrument_robustness.noise_eval_common import noise_conditions
from instrument_robustness.noise_sweep import mix_at_snr, out_path, window_seed

CLIP = 66150


def tone(freq: float, rms: float = 0.1, length: int = CLIP) -> np.ndarray:
    t = np.arange(length) / SR
    return (rms * np.sqrt(2) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def scaled_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """The added component that puts `noise` at `snr_db` against `clean`."""
    noisy, _, _ = mix_at_snr(clean, noise.astype(np.float32), snr_db)
    return noisy - clean


class FrozenNoiseProtocolTests(unittest.TestCase):
    def test_evidence_backed_grid_and_replicates_are_frozen(self) -> None:
        """Changing either value after generation would invalidate the shared noisy corpus."""
        self.assertEqual(SNRS, [60, 50, 40, 30, 20, 10, 0, -10])
        self.assertEqual(N_REPLICATES, 2)
        self.assertEqual(
            1 + len(NOISE_TYPES) * len(SNRS) * N_REPLICATES,
            49,
        )


class BandPowerTests(unittest.TestCase):
    def test_instrument_band_includes_the_lowest_dataset_fundamental(self) -> None:
        """MIDI 22 is about 29 Hz; a 50 Hz lower edge silently excluded the lowest tuba note."""
        self.assertLessEqual(INSTRUMENT_BAND_HZ[0], 29.0)
        self.assertGreaterEqual(INSTRUMENT_BAND_HZ[1], 2489.0)

    def test_parseval_bands_sum_to_total_power(self) -> None:
        """Band powers must be commensurable with the headline SNR's whole-signal power."""
        signal = np.random.default_rng(0).standard_normal(CLIP)
        total = float(np.mean(signal**2))
        self.assertAlmostEqual(band_power(signal, SR, 0.0, SR / 2), total, places=10)
        halves = band_power(signal, SR, 0.0, SR / 4) + band_power(
            signal, SR, SR / 4 + 1e-9, SR / 2
        )
        self.assertAlmostEqual(halves, total, places=10)

    def test_a_tone_puts_its_power_in_its_own_band(self) -> None:
        signal = tone(1000.0)
        inside = band_power(signal, SR, 900.0, 1100.0)
        outside = band_power(signal, SR, 3000.0, 5000.0)
        self.assertGreater(inside, 0.9 * float(np.mean(signal.astype(np.float64) ** 2)))
        self.assertLess(outside, 1e-6)


class Item1BandSnrTests(unittest.TestCase):
    """Whole-spectrum SNR hides WHERE the noise sits."""

    def setUp(self) -> None:
        self.clean = tone(440.0)
        rng = np.random.default_rng(1)
        self.white = rng.standard_normal(CLIP)
        rumble = np.cumsum(rng.standard_normal(CLIP))
        self.rumble = rumble / np.std(rumble)
        coefficients = butter(4, 9000 / (SR / 2), btype="high")
        hf = lfilter(*coefficients, rng.standard_normal(CLIP))
        self.hf = hf / np.std(hf)

    def test_out_of_band_noise_reports_a_far_higher_band_snr(self) -> None:
        """The finding item 1 asks for: at the same nominal SNR, spectrally lopsided noise barely
        touches the instrument band, and the diagnostic says so by a wide margin."""
        white_band = band_snr_db(self.clean, scaled_to_snr(self.clean, self.white, 0.0))
        rumble_band = band_snr_db(self.clean, scaled_to_snr(self.clean, self.rumble, 0.0))
        hf_band = band_snr_db(self.clean, scaled_to_snr(self.clean, self.hf, 0.0))
        self.assertLess(abs(white_band), 5.0)
        self.assertGreater(rumble_band, white_band + 15.0)
        self.assertGreater(hf_band, white_band + 15.0)

    def test_white_noise_band_snr_is_close_to_the_nominal_value(self) -> None:
        for snr in (20.0, 0.0):
            band = band_snr_db(self.clean, scaled_to_snr(self.clean, self.white, snr))
            self.assertLess(abs(band - snr), 6.0)

    def test_worst_octave_ignores_bands_the_instrument_does_not_occupy(self) -> None:
        """Regression: without the clean_share filter this returned ~-150 dB from an empty band,
        identical for every noise type and therefore useless."""
        profile = octave_snr_db(self.clean, scaled_to_snr(self.clean, self.white, 0.0))
        worst = worst_octave(profile)
        self.assertGreaterEqual(worst["clean_share"], MIN_CLEAN_SHARE)
        self.assertGreater(worst["snr_db"], -60.0)
        # A 440 Hz tone lives in the 500 Hz octave.
        self.assertAlmostEqual(worst["center_hz"], 500.0)

    def test_octave_profile_clean_shares_sum_to_about_one(self) -> None:
        profile = octave_snr_db(self.clean, scaled_to_snr(self.clean, self.white, 0.0))
        self.assertGreater(sum(record["clean_share"] for record in profile), 0.9)

    def test_worst_octave_falls_back_rather_than_picking_an_empty_band(self) -> None:
        profile = [
            {"center_hz": 100.0, "snr_db": -99.0, "clean_share": 0.0},
            {"center_hz": 200.0, "snr_db": 5.0, "clean_share": 0.0009},
        ]
        self.assertAlmostEqual(worst_octave(profile)["center_hz"], 200.0)


class Item5TimeStructureTests(unittest.TestCase):
    """Whole-window SNR hides WHEN the noise sits."""

    def setUp(self) -> None:
        self.clean = tone(440.0)
        rng = np.random.default_rng(2)
        self.stationary = rng.standard_normal(CLIP)
        burst = np.zeros(CLIP)
        burst[30000:30700] = rng.standard_normal(700) * 5.0
        self.burst = burst / np.std(burst)

    def test_active_fraction_separates_a_transient_from_ambience(self) -> None:
        self.assertGreater(active_fraction(self.stationary), 0.9)
        self.assertLess(active_fraction(self.burst), 0.2)

    def test_active_fraction_is_zero_for_silence_and_bounded(self) -> None:
        self.assertEqual(active_fraction(np.zeros(CLIP)), 0.0)
        for signal in (self.stationary, self.burst):
            self.assertGreaterEqual(active_fraction(signal), 0.0)
            self.assertLessEqual(active_fraction(signal), 1.0)

    def test_segmental_percentiles_describe_the_burst_not_the_silence(self) -> None:
        """Regression: measured over ALL frames, a 30 ms burst gave p05 = +161 dB, describing the
        99% of frames with no noise. Percentiles are over active-noise frames for this reason."""
        added = scaled_to_snr(self.clean, self.burst, 0.0)
        segmental = segmental_snr_db(self.clean, added)
        self.assertLess(segmental["p05"], 0.0)
        self.assertLess(segmental["p50"], 0.0)
        self.assertLess(segmental["min"], 0.0)
        self.assertLess(segmental["n_active_frames"], segmental["n_frames"] // 4)

    def test_a_transient_has_a_much_wider_spread_than_ambience(self) -> None:
        stationary = segmental_snr_db(
            self.clean, scaled_to_snr(self.clean, self.stationary, 0.0)
        )
        burst = segmental_snr_db(self.clean, scaled_to_snr(self.clean, self.burst, 0.0))
        self.assertLess(burst["min"], stationary["min"])
        self.assertGreater(stationary["n_active_frames"], burst["n_active_frames"] * 5)

    def test_silence_does_not_raise(self) -> None:
        result = segmental_snr_db(self.clean, np.zeros(CLIP))
        self.assertEqual(result["n_active_frames"], 0)


class Item6EffectiveSnrTests(unittest.TestCase):
    """The SNR the model receives differs from the one on the label."""

    def setUp(self) -> None:
        self.clean = tone(440.0)
        rng = np.random.default_rng(3)
        coefficients = butter(4, 9000 / (SR / 2), btype="high")
        hf = lfilter(*coefficients, rng.standard_normal(CLIP))
        self.hf = hf / np.std(hf)
        rumble = np.cumsum(rng.standard_normal(CLIP))
        self.rumble = rumble / np.std(rumble)

    def test_noise_above_the_model_nyquist_mostly_disappears(self) -> None:
        """AST resamples to 16 kHz and low-passes at 8 kHz, so 9 kHz+ noise never reaches it. The
        model receives a far higher SNR than the condition name claims."""
        added = scaled_to_snr(self.clean, self.hf, 0.0)
        at_16k = effective_snr_db(self.clean, added, target_sr=16000)
        at_32k = effective_snr_db(self.clean, added, target_sr=32000)
        self.assertGreater(at_16k, 15.0)
        self.assertGreater(at_16k, at_32k + 10.0)

    def test_low_frequency_noise_survives_every_model_rate(self) -> None:
        added = scaled_to_snr(self.clean, self.rumble, 0.0)
        for rate in (16000, 24000, 32000):
            self.assertLess(
                abs(effective_snr_db(self.clean, added, target_sr=rate)), 3.0
            )

    def test_identity_rate_reproduces_the_whole_window_snr(self) -> None:
        added = scaled_to_snr(self.clean, self.rumble, 7.0)
        self.assertAlmostEqual(
            effective_snr_db(self.clean, added, target_sr=SR), 7.0, places=4
        )


class DiagnosticContractTests(unittest.TestCase):
    def test_every_declared_column_is_produced(self) -> None:
        clean = tone(440.0)
        added = scaled_to_snr(clean, np.random.default_rng(4).standard_normal(CLIP), 0.0)
        diagnostics = mixture_diagnostics(clean, added)
        self.assertEqual(set(DIAGNOSTIC_COLUMNS) - set(diagnostics), set())
        for column in DIAGNOSTIC_COLUMNS:
            if column == "snr_octave_db":
                self.assertIsInstance(diagnostics[column], list)
            else:
                self.assertTrue(
                    np.isfinite(float(diagnostics[column])), f"{column} is not finite"
                )

    def test_band_edges_are_recorded_so_a_paper_can_state_them(self) -> None:
        clean = tone(440.0)
        added = scaled_to_snr(clean, np.random.default_rng(5).standard_normal(CLIP), 0.0)
        diagnostics = mixture_diagnostics(clean, added)
        self.assertEqual(diagnostics["snr_band_low_hz"], INSTRUMENT_BAND_HZ[0])
        self.assertEqual(diagnostics["snr_band_high_hz"], INSTRUMENT_BAND_HZ[1])

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            mixture_diagnostics(tone(440.0), np.zeros(100))


class Item5ActiveInstrumentTests(unittest.TestCase):
    """Whole-window SNR can hide how masked a short instrument event is."""

    def test_active_instrument_snr_excludes_surrounding_silence(self) -> None:
        clean = np.zeros(CLIP, dtype=np.float32)
        start, stop = 25000, 35000
        clean[start:stop] = tone(440.0, length=stop - start)
        noise = np.random.default_rng(40).standard_normal(CLIP)
        added = scaled_to_snr(clean, noise, 0.0)

        result = active_signal_snr_db(clean, added)

        self.assertLess(result["active_fraction"], 0.25)
        self.assertGreater(result["n_active_frames"], 0)
        # The whole-window condition is 0 dB, but the note concentrates its energy into a small
        # part of the clip, so its active frames are substantially cleaner.
        self.assertGreater(result["snr_db"], 5.0)

    def test_stationary_note_uses_almost_every_frame(self) -> None:
        clean = tone(440.0)
        added = scaled_to_snr(
            clean, np.random.default_rng(41).standard_normal(CLIP), 7.0
        )
        result = active_signal_snr_db(clean, added)
        self.assertGreater(result["active_fraction"], 0.99)
        self.assertAlmostEqual(result["snr_db"], 7.0, delta=0.25)


class Item3ReplicateTests(unittest.TestCase):
    """A replicate must be an independent draw of the same condition."""

    def test_replicate_changes_the_seed(self) -> None:
        first = window_seed("w000", "white", "build-a", 0)
        second = window_seed("w000", "white", "build-a", 1)
        self.assertNotEqual(first, second)
        self.assertEqual(first, window_seed("w000", "white", "build-a", 0))

    def test_replicate_defaults_to_zero(self) -> None:
        self.assertEqual(
            window_seed("w000", "white", "build-a"),
            window_seed("w000", "white", "build-a", 0),
        )

    def test_snr_is_still_excluded_from_the_seed(self) -> None:
        """The realization must span the SNR curve; only the gain may change along it."""
        self.assertEqual(
            window_seed("w000", "white", "build-a", 2),
            window_seed("w000", "white", "build-a", 2),
        )

    def test_a_negative_or_fractional_replicate_is_refused(self) -> None:
        for bad in (-1, 0.5):
            with self.assertRaises(ValueError):
                window_seed("w000", "white", "build-a", bad)

    def test_replicates_land_in_separate_directories(self) -> None:
        first = out_path("white", 20, "w000", replicate=0, noisy_dir="/tmp/x")
        second = out_path("white", 20, "w000", replicate=1, noisy_dir="/tmp/x")
        self.assertNotEqual(first, second)
        # The replicate directory is present unconditionally: a layout that changes shape with a
        # config value needs two code paths on every reader.
        self.assertEqual(first.parent.name, "r0")

    def test_conditions_cover_the_full_grid_including_replicates(self) -> None:
        conditions = noise_conditions()
        self.assertEqual(
            len(conditions), 1 + len(NOISE_TYPES) * len(SNRS) * N_REPLICATES
        )
        self.assertEqual(conditions[0].tag, "clean")
        self.assertIsNone(conditions[0].replicate)
        noisy = conditions[1:]
        self.assertEqual(
            len({(c.noise_type, c.snr_db, c.replicate) for c in noisy}), len(noisy)
        )
        self.assertEqual(len({c.tag for c in noisy}), len(noisy))

    def test_a_different_draw_actually_yields_different_noise(self) -> None:
        from instrument_robustness.noise_sweep import draw_noise

        first, _ = draw_noise(
            "white", np.random.default_rng(window_seed("w", "white", "b", 0)), {}
        )
        second, _ = draw_noise(
            "white", np.random.default_rng(window_seed("w", "white", "b", 1)), {}
        )
        self.assertFalse(np.array_equal(first, second))
        # Independent, not merely shifted: near-zero correlation.
        correlation = float(
            np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        )
        self.assertLess(abs(correlation), 0.05)


class GenerateEndToEndTests(unittest.TestCase):
    """Run the real `generate()` and the real `validate_noise_manifest()` against each other.

    Previously impossible: `generate()` bound its paths at import time, so it could only ever run
    against the production data root. Every seed draw, every mixture write and every diagnostic
    happens inside that loop, and none of it can be reconstructed after the fact -- which makes it
    the last function that should go untested. It now takes call-time paths, so this test exercises
    the whole path on a temporary build.

    White noise only, so no ESC-50 corpus is needed and the test stays fast.
    """

    def test_generate_then_validate_with_two_replicates(self) -> None:
        import tempfile
        import unittest.mock as mock
        from pathlib import Path

        import pandas as pd
        import soundfile as sf

        import instrument_robustness.noise_sweep as ns
        from instrument_robustness.config import write_artifact_fingerprint

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            pipeline = root / "pipeline"
            pipeline.mkdir(parents=True)
            manifest_rows, window_rows = [], []
            for index, label in enumerate(TARGET_LABELS):
                window_path = f"work/windows/{label}_A{index}_w000.wav"
                (root / window_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    str(root / window_path),
                    tone(220.0 + 40 * index),
                    SR,
                    subtype="PCM_16",
                )
                manifest_rows.append(
                    {"path": f"raw/{label}_A{index}.mp3", "label": label, "note": f"A{index}"}
                )
                window_rows.append(
                    {
                        "window_path": window_path,
                        "source_path": f"raw/{label}_A{index}.mp3",
                        "label": label,
                        "split": "test",
                    }
                )
            manifest_csv = root / "manifest.csv"
            fingerprint = root / "manifest_fingerprint.json"
            labeled = pipeline / "manifest_labeled.csv"
            windows_csv = pipeline / "windows.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
            write_artifact_fingerprint(
                manifest_csv, "prep_data", fingerprint_path=fingerprint
            )
            pd.DataFrame(manifest_rows).to_csv(labeled, index=False)
            write_artifact_fingerprint(labeled, "step0_filter")
            pd.DataFrame(window_rows).to_csv(windows_csv, index=False)
            write_artifact_fingerprint(windows_csv, "step5_normalize")

            noisy_dir = root / "work" / "windows_noisy"
            with mock.patch.multiple(
                ns,
                NOISY_DIR=noisy_dir,
                N_REPLICATES=2,
                SNRS=[20, 0],
                NOISE_TYPES=["white"],
            ):
                ns.generate(
                    data_root=root,
                    windows_csv=windows_csv,
                    manifest_csv=manifest_csv,
                    manifest_fingerprint=fingerprint,
                    noisy_dir=noisy_dir,
                )
                manifest = ns.validate_noise_manifest(
                    noisy_dir=noisy_dir,
                    data_root=root,
                    windows_csv=windows_csv,
                    manifest_csv=manifest_csv,
                    manifest_fingerprint=fingerprint,
                    verify_audio_hashes=True,
                )

            expected = len(TARGET_LABELS) * 1 * 2 * 2  # windows x types x replicates x snrs
            self.assertEqual(manifest["n_files"], expected)
            self.assertEqual(manifest["n_replicates"], 2)
            self.assertEqual(manifest["seed_scheme"], ns.SEED_SCHEME)

            provenance = pd.read_csv(noisy_dir / "noise_provenance.csv")
            self.assertEqual(len(provenance), expected)
            self.assertEqual(sorted(provenance["replicate"].unique()), [0, 1])
            for column in DIAGNOSTIC_COLUMNS:
                self.assertIn(column, provenance.columns)
            # Requested SNR is achieved in the file that was actually written and reloaded.
            self.assertLess(
                (provenance["realized_snr_db"] - provenance["snr_db"]).abs().max(),
                ns.MAX_SNR_ERROR_DB,
            )
            # The point of a replicate: same condition, genuinely different draw.
            at_zero = provenance[provenance["snr_db"] == 0]
            per_window = at_zero.groupby("window_id")["unscaled_noise_power"].nunique()
            self.assertTrue((per_window > 1).all())
            # ...but the realization is still shared across SNRs within one replicate.
            grouped = provenance.groupby(["window_id", "noise_type", "replicate"])
            self.assertEqual(grouped["seed"].nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
