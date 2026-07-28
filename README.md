# rise — instrument-classification robustness

Preprocessing + featurization pipeline and model branches for 12-class instrument classification,
built to neutralize two Philharmonia confounds (per-instrument MP3 bitrate; phrase-length) and to
compare models under clean and (later) noisy conditions.

**12 classes:** bassoon, cello, clarinet, double-bass, flute, french-horn, oboe, trombone,
trumpet, tuba, viola, violin.

## Layout

```
src/instrument_robustness/     # installable package (all CODE)
  config.py                    # paths (DATA_ROOT-relative) + all pipeline params
  step0_filter … step7_featurize.py
  featurelib.py                # SVM vector + CNN/CRNN log-mel extractors
  crnn_data.py                 # CRNN loader (reuses CNN features)
  pretrained_extractors.py     # PANNs CNN14 / AST / MERT on-the-fly extractors
  extract_mert.py              # frozen MERT train/validation embedding extraction
  mert_data.py, mert_probe.py  # MERT data contract + layer-weighted linear probe
  train_mert.py                # validation-only MERT probe selection
  ast_data.py, train_ast.py    # AST on-the-fly DataLoader and fine-tuning command
all-samples/                   # DATA + ARTIFACTS (not code)
  manifest.csv, <instrument>/<note>/         # raw audio + catalog
  pipeline/                    # stage manifests, splits.csv, windows.csv, stats, fingerprints
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

## Data — `prep_data.py` is the official dataset

**Run this first. It is the only supported way to obtain the data.**

```bash
python -m instrument_robustness.prep_data
```

It downloads all **12** instruments from the Internet Archive mirror of the Philharmonia library
(CC-BY-SA 4.0), unpacks them into `<data root>/<instrument>/<note>/`, and writes `manifest.csv` —
the index every step below reads. It also writes `manifest_fingerprint.json`, recording the config
that produced the index.

**Do not use an unfingerprinted data tree or pre-derived feature/window archive.** Every pipeline
stage, feature array, and model artifact records the preprocessing fingerprint; loaders reject
missing or mismatched provenance rather than silently training on incompatible data.

`download_data.py` (Google Drive) is **deprecated** and its archives are **not** interchangeable
with this pipeline — they were built against the old file-level split and zero-padded windows,
carry no fingerprint, and cover only 9 of the 12 classes.

> **Migrating from the 9-class set:** label indices have shifted, so every checkpoint and feature
> array produced before this change is invalid and must be regenerated. This is unavoidable —
> `TARGET_LABELS` fixes the label indices, and oboe, double-bass and french-horn now exist.

## Run the pipeline

```bash
python -m instrument_robustness.prep_data         # fetch data + write manifest.csv  (START HERE)
python -m instrument_robustness.step0_filter      # keep 12 classes and one articulation per class
python -m instrument_robustness.step1_resample    # 22050 Hz mono (kills bitrate confound)
python -m instrument_robustness.step2_trim        # silence trim
python -m instrument_robustness.step3_split       # split BY PITCH GROUP (70/15/15)
python -m instrument_robustness.step4_window      # 3.0 s windows, short notes TILED not padded
python -m instrument_robustness.step5_normalize   # per-window RMS normalize
python -m instrument_robustness.step6_stats       # TRAIN-ONLY normalization stats
python -m instrument_robustness.step7_featurize   # SVM / CNN / CRNN features
```

> Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1` for the
> parallel librosa steps (6–7) to avoid thread oversubscription.

See `all-samples/pipeline/pipeline_report.txt` for the full run report (shapes, per-class per-split
counts, confound checks, invariants).

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

## Start the MERT baseline

MERT uses the authoritative `windows.csv` splits and the Step-5 normalized window audio. It resamples
each 22.05 kHz window to the pretrained `m-a-p/MERT-v1-95M` model's native 24 kHz rate and does not
use the Step-6 SVM/CNN statistics. The first baseline freezes MERT, caches a mean-pooled representation
for each of its 13 hidden states, and trains a learned layer mixture plus a linear 12-class probe.

Install the optional pretrained-model dependencies and make sure the repaired Step-5 windows are
present, then extract train and validation only:

```bash
pip install -e ".[mert]"
python -m instrument_robustness.extract_mert
python -m instrument_robustness.train_mert
```

Neither command reads the MERT test split. Inspect and freeze
`artifacts/mert/validation_summary.json`; then refit on train+validation and perform the one
permitted test extraction/evaluation with:

```bash
python -m instrument_robustness.finalize_mert
```

The finalizer uses the validation-selected learning rate and epoch, requires the exact saved MERT
checkpoint revision, and refuses to run if test extraction or finalization has already started.
On BU SCC, submit `scc/mert_probe.qsub` first and submit `scc/mert_finalize.qsub` only after
validation review. The MERT checkpoint is licensed CC-BY-NC-4.0; this branch is appropriate for
the project's non-commercial research use, but that license must be reviewed before any
commercial use.
