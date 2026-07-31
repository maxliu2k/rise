GENERATED -- DO NOT EDIT. Trained weights for all six models, flat in one folder.

Rebuild after retraining:   python -m instrument_robustness.bundle_weights
Verify nothing has drifted: python -m instrument_robustness.bundle_weights --check

  ast_finetuned.safetensors     AST, fine-tuned            (Git LFS, 329 MB)
  cnn_seed{42..46}.pt           CNN ensemble, 5 seeds
  crnn_seed{42..46}.pt          CRNN ensemble, 5 seeds
  svm_selected.joblib           fit on TRAIN, config chosen on validation
  svm_final.joblib              refit on TRAIN+VAL, used for the test evaluation
  mert_probe_selected.pt        fit on TRAIN, chosen on validation
  mert_probe_final.pt           refit on TRAIN+VAL, used for the test evaluation
  panns_probe_philharmonia.pt   PANNs linear probe (the TinySOL probe stays in
                                artifacts/panns/ -- cross-dataset experiment, not
                                part of the six-model Philharmonia comparison)

_selected and _final are NOT interchangeable. _selected is what validation chose; _final saw the
validation split during fitting, so scoring it on validation is meaningless. For the seed
ensembles the seed IS the role -- all five are equal and none is "best".

MANIFEST.json records the sha256, byte count and originating artifacts/ path of every file.
--check recomputes both sides, so an edited copy, a retrained source, or a missing file fails
rather than passing quietly.

These are COPIES. artifacts/<model>/ remains where finalize_* and noise_eval_* read from, and
where the metrics, confusion matrices and status files live. Nothing here is loaded by the code.

ast_finetuned.safetensors is Git LFS. Clone with git-lfs installed, or you get a 130-byte pointer
that fails only when something tries to load it as a model.
