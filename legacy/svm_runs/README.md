# Retired local SVM runs

The current official SVM artifacts live in `artifacts/svm/`.

Local-only archived runs are kept under `local-data/`, which is ignored by Git:

- `pre_one_window_1310/` — the superseded run whose test split contained 1,310 examples.
- `one_window_initial_search/` — the first validation-only search on the current 1,255-test-example
  build. Its search boundary motivated the extended search, and it was never evaluated on test.

The promoted official run used the extended validation grid, selected RBF `C=10` and
`gamma=0.003` by validation macro-F1, refit on train plus validation, and evaluated the
1,255-example test split once.
