GENERATED -- DO NOT EDIT. Trained weights for all six models, flat in one folder.

Rebuild after retraining:   python -m instrument_robustness.bundle_weights
Verify nothing has drifted: python -m instrument_robustness.bundle_weights --check

  cnn_seed{42..46}.pt           CNN ensemble, 5 seeds
  crnn_seed{42..46}.pt          CRNN ensemble, 5 seeds
  svm_selected.joblib           fit on TRAIN, config chosen on validation
  svm_final.joblib              IDENTICAL to svm_selected since e4e455d -- see below
  mert_probe_selected.pt        fit on TRAIN, chosen on validation
  mert_probe_final.pt           refit on TRAIN+VAL, used for the test evaluation
  (AST and PANNs fine-tune are NOT here -- see external_files in MANIFEST.json for their
   sha256, byte count, dataset_fingerprint and SCC path)

_selected and _final are NOT interchangeable IN GENERAL. _selected is what validation chose;
_final saw the validation split during fitting, so scoring it on validation is meaningless. For
the seed ensembles the seed IS the role -- all five are equal and none is "best".

EXCEPT SVM, since e4e455d. finalize_svm now fits the selected configuration on TRAIN ONLY rather
than refitting on TRAIN+VAL, so svm_selected.joblib and svm_final.joblib are the same bytes and
MANIFEST.json records one sha256 for both. That is the protocol, not a copy error -- the SVM
test_summary.json says "final model fit on train only". MERT still refits, so its two probes do
differ. Do not "fix" the duplicate by deleting one; check the protocol field first.

MANIFEST.json records the sha256, byte count and originating artifacts/ path of every file.
--check recomputes both sides, so an edited copy, a retrained source, or a missing file fails
rather than passing quietly.

These are COPIES. artifacts/<model>/ remains where finalize_* and noise_eval_* read from, and
where the metrics, confusion matrices and status files live. Nothing here is loaded by the code.

The two 300 MB checkpoints (AST, PANNs fine-tune) are NOT here. They are EXTERNAL_WEIGHTS
entries in bundle_weights.py, each carrying sha256, byte count and dataset_fingerprint, and are
readable on SCC at /projectnb/rise-grid/models/<fingerprint>/. Verify against SHA256SUMS there.

The PANNs fine-tune is NOT copied into this folder. MANIFEST.json records its exact filename,
SHA-256, release URL and scientific role. Download it explicitly and verify the hash. The included
probe was removed because it cannot reproduce the reported PANNs result.
