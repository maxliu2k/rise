GENERATED -- DO NOT EDIT. Trained weights for all six models, one folder per model.

Rebuild after retraining:   python -m instrument_robustness.bundle_weights
Verify nothing has drifted: python -m instrument_robustness.bundle_weights --check

MANIFEST.json records the sha256 of each source artifact. --check recomputes both the source and
the copy, so an edited copy, a retrained source, or a missing file all fail rather than pass
quietly.

These are COPIES. artifacts/<model>/ remains where finalize_* and noise_eval_* read from, and
where the metrics, confusion matrices and status files live. Nothing here is loaded by the code.

ast/model.safetensors is Git LFS (329 MB). Clone with git-lfs installed or you get a pointer file
that will fail to load as a model.
