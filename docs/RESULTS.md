# Model comparison — 12-class Philharmonia

Regenerate with `python -m instrument_robustness.summarize_results --write`. Every clean number is verified against the current `config_fingerprint()`; a row marked STALE was trained under a different config and must not be quoted.

## Clean test baselines

| model | split | macro_f1 | accuracy | n | status |
| --- | --- | --- | --- | --- | --- |
| AST | test | 0.9908 | 0.9912 | 1255 | canonical |
| SVM | test | 0.9770 | 0.9785 | 1255 | canonical |
| PANNs | test | 0.9868 | 0.9880 | 1255 | canonical |
| MERT | test | 0.8931 | 0.8956 | 1255 | canonical |
| CNN | test | 0.9708 | 0.9721 | 1255 | canonical |
| CRNN | test | 0.9738 | 0.9753 | 1255 | canonical |

_No noise sweeps found yet._
