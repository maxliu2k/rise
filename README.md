# rise — instrument-classification robustness

Preprocessing + featurization pipeline and model branches for 9-class instrument classification,
built to neutralize two Philharmonia confounds (per-instrument MP3 bitrate; phrase-length) and to
compare models under clean and (later) noisy conditions.

**9 classes:** violin, viola, cello, flute, clarinet, bassoon, trumpet, tuba, trombone
(oboe is absent from this Philharmonia copy → bassoon substitutes).

## Layout

```
src/instrument_robustness/     # installable package (all CODE)
  config.py                    # paths (DATA_ROOT-relative) + all pipeline params
  step0_filter … step7_featurize.py
  featurelib.py                # SVM vector + CNN/CRNN log-mel extractors
  crnn_data.py                 # CRNN loader (reuses CNN features)
  pretrained_extractors.py     # PANNs CNN14 / AST / MERT on-the-fly extractors
  ast_data.py, train_ast.py    # AST on-the-fly DataLoader and fine-tuning command
all-samples/                   # DATA + ARTIFACTS (not code)
  manifest.csv, Strings/ Brass/ Woodwinds/   # raw audio + catalog
  pipeline/                    # manifest_9*.csv, splits.csv, windows.csv, norm_stats.*, pipeline_report.txt
  work/                        # resampled / trimmed / windowed audio
  features/                    # svm/ cnn/ (npz) + crnn/ ast/ mert/ panns/ (docs)
configs/                       # svm.yaml, irmas.yaml
```

Code and data are decoupled: `config.py` finds the data root via `<repo>/all-samples` by default,
or `RISE_DATA_ROOT` (see `.env.example`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core deps (numpy/pandas/librosa/scikit-learn/…)
pip install -e ".[ast]"          # AST fine-tuning only
pip install -e ".[pretrained]"   # + torch/transformers/panns for AST/MERT/PANNs branches
```

## Run the pipeline

```bash
# steps read/write under the data root; run from anywhere once installed
python -m instrument_robustness.step0_filter      # filter manifest to 9 classes
python -m instrument_robustness.step1_resample    # 22050 Hz mono (kills bitrate confound)
python -m instrument_robustness.step2_trim        # silence trim
python -m instrument_robustness.step3_split       # split BY SOURCE FILE (70/15/15)
python -m instrument_robustness.step4_window      # 3.0 s windows (kills phrase-length confound)
python -m instrument_robustness.step5_normalize   # per-window RMS normalize
python -m instrument_robustness.step6_stats       # TRAIN-ONLY normalization stats
python -m instrument_robustness.step7_featurize   # SVM / CNN / CRNN features
```

> Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1` for the
> parallel librosa steps (6–7) to avoid thread oversubscription.

See `all-samples/pipeline/pipeline_report.txt` for the full run report (shapes, per-class per-split
counts, confound checks, invariants).

<<<<<<< ours
## Fine-tune AST

The AST branch reads Step-5-normalized windows directly from `pipeline/windows.csv`; no AST
inputs are materialized. It builds one `ASTFeatureExtractor`, resamples each 22050 Hz waveform to
16 kHz in the DataLoader, then fine-tunes the pretrained model and retains the best validation
checkpoint.

```bash
python -m instrument_robustness.train_ast --epochs 10 --batch-size 8
```

The command downloads `MIT/ast-finetuned-audioset-10-10-0.4593` on first use and writes the best
checkpoint plus `metrics.json` to `all-samples/models/ast/` by default. For noisy runs, pass a
waveform transform to `ASTWindowDataset` or `make_ast_dataloader`; it is applied to the 22050 Hz
window before `ast_input`.

After testing, that output directory also contains `test_by_instrument.csv` with accuracy,
precision, recall, F1, and test-clip counts for each instrument; `test_by_family.csv` with
percentage accuracy for strings, woodwinds, and brass; and `test_confusion_matrix.csv` showing
which instruments were confused with one another.
=======
## Train the SVM baseline

The SVM features are already standardized with training-set statistics. Tune on the validation split and save the search results plus selected model with:

```bash
python -m instrument_robustness.train_svm
```

By default this tunes an RBF SVC over `C` and `gamma`, using validation macro-F1 for selection.
It reads only `train.npz` and `val.npz`; `test.npz` remains untouched for the final evaluation. The defaults and predeclared final-test policy are documented in `configs/models/svm.yaml`.
Outputs under `artifacts/svm/` include the ranked search, validation confusion matrix, selected model, and a summary containing the feature schema, input/output hashes, and software versions.

After the validation results are frozen, fit the selected configuration on the combined train and validation arrays and perform the one permitted test evaluation with:

```bash
python -m instrument_robustness.finalize_svm
```

This command does not tune or standardize again. It writes a final model, test metrics, a test confusion matrix, and a status record under `artifacts/svm/`. The status record makes the command refuse a second test evaluation.
>>>>>>> theirs
