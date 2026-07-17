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
