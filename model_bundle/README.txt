GENERATED -- DO NOT EDIT ANYTHING IN THIS FOLDER.

Every file here is a copy of one under src/instrument_robustness/. Edit the original and re-run:

    python -m instrument_robustness.bundle_models

MANIFEST.json records the sha256 of each source file at bundle time. To prove nothing has drifted:

    python -m instrument_robustness.bundle_models --check

That exits non-zero if any copy disagrees with its source, so a stale bundle fails rather than
quietly misleading whoever reads it.

The code here is NOT importable as a package and is not on sys.path. It exists so a reader can see
everything belonging to one model in one place. Run models from the real package.

  <model>/     the six models: svm, cnn, crnn, ast, mert, panns
  _shared/     config, featurelib and the pretrained extractors, used by more than one model
  _pipeline/   prep_data, run_pipeline and steps 0-7, which build the features every model reads
  _noise/      the shared noise sweep, metrics and evaluation contract
