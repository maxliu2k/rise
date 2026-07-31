# Model comparison — 12-class Philharmonia

Regenerate with `python -m instrument_robustness.summarize_results --write`. Every clean number is verified against the current `config_fingerprint()`; a row marked STALE was trained under a different config and must not be quoted.

## Clean test baselines

| model | split | macro_f1 | accuracy | n | status |
| --- | --- | --- | --- | --- | --- |
| AST | test | 0.9917 | 0.9928 | — | canonical |
| SVM | test | 0.9914 | 0.9920 | 1255.0000 | canonical |
| PANNs | test | 0.9845 | 0.9841 | 1255.0000 | canonical |
| MERT | test | 0.9246 | 0.9259 | 1255.0000 | canonical |
| CNN | val (5-seed) | — | — | — | STALE (pre-standardisation, no selection_metric); retrain |
| CRNN | val (5-seed) | — | — | — | STALE (pre-standardisation, no selection_metric); retrain |

## Noise robustness — macro-F1 retention vs clean (replicates averaged)

| model | clean_f1 | whit@20 | natu@20 | mech@20 | whit@0 | natu@0 | mech@0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PANNs | 0.9845 | 0.3820 | 0.7540 | 0.7880 | 0.1770 | 0.3880 | 0.4110 |

Columns are retention at the named noise type and SNR (dB). 1.0 = no degradation; 0.083 macro-F1 is 12-class chance.
