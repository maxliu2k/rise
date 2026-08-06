# CNN–CRNN comparison at 20 dB

This directory records a targeted secondary comparison of the CNN and CRNN on the canonical
12-class test predictions. It was selected for architectural interpretability: both models consume
the same standardized log-mel representation and use the same training workflow, while the CRNN
adds recurrent temporal processing. This comparison was specified after the model results existed,
so it is exploratory rather than preregistered or confirmatory.

## Declared comparison family

- Models: CNN versus CRNN.
- Conditions: 20 dB under white Gaussian noise, ESC-50 human non-speech (`audience`), and DEMAND
  environmental ambience (`studio`).
- Noise realizations: `r0` and `r1`, analyzed separately.
- Effect: CRNN macro-F1 minus CNN macro-F1. Negative values favor CNN.
- Uncertainty: 2,000 pitch-group bootstrap resamples, seed 0, with the fixed 12-class label order.
- Complementary test: exact pitch-group sign test on paired correctness.
- Multiplicity: Benjamini–Hochberg correction across the six sign-test p-values.

The bootstrap interval estimates uncertainty in the macro-F1 difference. The sign test asks whether
one model wins on correctness in more pitch groups and ignores the magnitude of each win. They are
different summaries and need not agree.

## Results

| condition | CNN macro-F1 | CRNN macro-F1 | CRNN − CNN | 95% bootstrap interval | sign-test p | BH q |
|---|---:|---:|---:|---:|---:|---:|
| White, r0 | 0.3236 | 0.2114 | -0.1122 | [-0.1625, -0.0560] | 0.5601 | 0.8401 |
| White, r1 | 0.3249 | 0.2120 | -0.1130 | [-0.1601, -0.0589] | 0.8877 | 0.8877 |
| ESC-50 human non-speech, r0 | 0.6302 | 0.6658 | +0.0356 | [+0.0027, +0.0657] | 0.2892 | 0.5785 |
| ESC-50 human non-speech, r1 | 0.6576 | 0.6609 | +0.0033 | [-0.0232, +0.0246] | 0.7877 | 0.8877 |
| DEMAND environmental ambience, r0 | 0.4475 | 0.4940 | +0.0465 | [+0.0138, +0.0772] | 0.1550 | 0.4650 |
| DEMAND environmental ambience, r1 | 0.4535 | 0.4792 | +0.0257 | [-0.0015, +0.0493] | 0.1299 | 0.4650 |

CNN's white-noise macro-F1 advantage was consistent across both realizations, and both bootstrap
intervals excluded zero. The recorded-noise estimates favored CRNN, but their bootstrap intervals
did not exclude zero in both realizations. None of the six exact pitch-group sign tests survived
Benjamini–Hochberg correction at q < 0.05. These are targeted effect estimates, not evidence of
general CNN or CRNN superiority.

The six condition JSON files are direct outputs from `instrument_robustness.noise_stats`.
`bh_summary.json` collects the declared family, effect estimates, intervals, raw sign-test p-values,
and adjusted q-values.
