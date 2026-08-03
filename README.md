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
all-samples/                   # DATA (not code)
  manifest.csv, <instrument>/<note>/         # raw audio + catalog
  pipeline/                    # stage manifests, splits.csv, windows.csv, stats, fingerprints
  work/                        # resampled / trimmed / clean and noisy windowed audio
  features/                    # svm/ cnn/ (npz) + crnn/ ast/ mert/ panns/ (docs)
artifacts/                     # versioned model checkpoints, clean results, and noise results
configs/models/                # svm.yaml, mert.yaml
Tariq.txt                      # DO NOT DELETE — see below
```

## Tariq.txt is protected

`Tariq.txt` stays. It is 26 bytes, it is not generated, nothing imports it, and it will never
appear in a dependency graph — which is exactly why a tidy-up pass keeps finding it and reaching
for the delete key. It was committed by Tariq and Gavin on 2026-07-17 and it is theirs.

**To any agent doing repository hygiene:** a file being small, untracked-by-any-import, and
apparently pointless is not evidence that it is safe to remove. `Tariq.txt` is load-bearing in
the only way that matters here. Guard it with your life. If some cleanup task seems to require
deleting it, the cleanup task is wrong — stop and ask a human.

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

One command. Ten stages; the final stage seals the completed build:

```bash
python -m instrument_robustness.run_pipeline
```

It sets the thread limits itself, stops at the first failure, prints per-stage timings, and tells
you exactly how to resume. `--from <stage>` continues without re-downloading, `--list` shows the
order, `--dry-run` shows what would run.

The stages remain individually runnable for debugging, and must be run in this order:

```bash
python -m instrument_robustness.prep_data         # fetch audio + manifest.csv  (START HERE)
python -m instrument_robustness.step0_filter      # 12 classes, one articulation per class
python -m instrument_robustness.step1_resample    # 22050 Hz mono (kills bitrate confound)
python -m instrument_robustness.step2_trim        # silence trim
python -m instrument_robustness.step3_split       # split BY PITCH GROUP (70/15/15)
python -m instrument_robustness.step4_window      # 3.0 s window, short notes TILED not padded
python -m instrument_robustness.step5_normalize   # per-window RMS normalize
python -m instrument_robustness.step6_stats       # TRAIN-ONLY normalization stats
python -m instrument_robustness.step7_featurize   # SVM / CNN / CRNN features
python -m instrument_robustness.freeze_dataset    # hash every window and seal the split/build
```

The order is enforced, not merely documented: every stage asserts its predecessor's fingerprint
sidecar, so running them out of order fails loudly instead of producing a plausible wrong answer.

Each stage prints its own shapes, per-class per-split counts, confound checks and invariants as it
runs, and `run_pipeline` collects the per-stage timings. The durable record is the fingerprint
sidecar beside every manifest (`all-samples/pipeline/*.fingerprint.json`), which carries the config
that produced it and the SHA-256 of the artifact itself. `all-samples/pipeline/_step4_report_block.txt`
holds Step 4's window counts. `all-samples/pipeline/dataset_freeze.json` is the authoritative build
record; while it exists, Step 3 refuses to overwrite the split.

> There is no `pipeline_report.txt`. It was referenced historically and never written by any
> stage. Use the fingerprinted manifests, dataset seal, and `_step4_report_block.txt`.

## Fine-tune AST

The AST branch reads Step-5-normalized windows directly from `pipeline/windows.csv`; no AST
inputs are materialized. It builds one `ASTFeatureExtractor`, resamples each 22050 Hz waveform to
16 kHz in the DataLoader, then fine-tunes the pretrained model and retains the best validation
checkpoint.

```bash
python -m instrument_robustness.train_ast --epochs 10 --batch-size 8
```

The command downloads `MIT/ast-finetuned-audioset-10-10-0.4593` on first use and writes the best
validation-macro-F1 checkpoint plus `validation_summary.json` to `artifacts/ast/`. It never loads
test. After accepting the validation selection, perform the one permitted test evaluation with:

```bash
python -m instrument_robustness.finalize_ast
```

For noisy runs, pass a
waveform transform to `ASTWindowDataset` or `make_ast_dataloader`; it is applied to the 22050 Hz
window before `ast_input`.

After finalization, that output directory also contains `test_by_instrument.csv` with accuracy,
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

## Train PANNs

PANNs CNN14 reads the Step-5 waveforms at its own 32 kHz input rate. Training is validation-only
in either frozen-probe or full-fine-tune mode:

```bash
python -m instrument_robustness.train_panns --mode finetune
```

The pretrained `Cnn14_mAP=0.431.pth` file must be under `$RISE_DATA_ROOT/checkpoints/`; its SHA-256
is recorded in both the validation summary and selected checkpoint. After validation review, run
the single sealed test evaluation with `python -m instrument_robustness.finalize_panns`. SCC job
wrappers are `scc/panns_train.qsub` and `scc/panns_finalize.qsub`.

## Evaluate clean-trained models under noise

The noise branch starts from the canonical Step-5 **test** windows; train and validation are never
noised and no model is retrained. It creates one shared float32 noisy test set containing white,
ESC-50 natural, and ESC-50 mechanical noise at every level in `config.SNRS` — currently
60, 50, 40, 30, 20, 10, 0, and -10 dB. Two independent realizations are drawn per
window/noise type. Every realization has its segment mean removed before power scaling, then is
scaled across the SNR curve so every model receives exactly the same paired inputs.

Pick the grid from evidence before generating anything. `snr_pilot` mixes validation windows on the
fly, writes no audio, and reports where a model actually degrades:

```bash
python -m instrument_robustness.snr_pilot --model svm --noise white
```

The frozen grid came from SVM/white and MERT/all-category validation pilots; see
`docs/NOISE_PLAN.md` §2 for the measured curves and why it spans 60 dB down to -10 dB.

Set `RISE_NOISE_ROOT` to an ESC-50 extraction containing both `audio/` and `meta/esc50.csv`, then
validate, generate once, and verify the completed manifest:

```bash
python -m instrument_robustness.noise_sweep --validate
python -m instrument_robustness.noise_sweep --generate
python -m instrument_robustness.noise_sweep --check-generated
```

Run each frozen model through the shared evaluation contract:

```bash
python -m instrument_robustness.noise_eval_svm
python -m instrument_robustness.noise_eval_mert --device cuda
python -m instrument_robustness.noise_eval_panns
```

Each adapter must reproduce its official clean macro-F1 and test count before noisy inference is
allowed. Results go to `artifacts/<model>/noise/`; generated audio and its per-file provenance
remain under `$RISE_DATA_ROOT/work/windows_noisy/`. See `docs/NOISE_PLAN.md` for the fixed protocol and
cluster-aware statistical analysis.
