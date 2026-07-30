# BU SCC jobs

## AST training on the 12-class Philharmonia data

Use `main` and keep the generated audio outside the repository:

```bash
cd /project/rise-grid/repos/$USER/instrument-robustness
git switch main
git pull --ff-only origin main
export RISE_DATA_ROOT="/projectnb/rise-grid/$USER/rise-data/philharmonia"
```

Submit the CPU preprocessing job first. It downloads the configured Philharmonia
sources and rebuilds every fingerprinted stage for all 12 instruments:

```bash
prep_job=$(qsub -terse -v RISE_DATA_ROOT="$RISE_DATA_ROOT" scc/ast_prepare.qsub)
qsub -hold_jid "$prep_job" -v RISE_DATA_ROOT="$RISE_DATA_ROOT" train_ast.qsub
qstat -u "$USER"
```

The GPU job verifies all 12 labels, provenance, and every window file before it
downloads AST or begins training.

## MERT probe

Use the shared clone on `main`:

```bash
cd /project/rise-grid/Tariq/instrument-robustness
git switch main
git pull --ff-only origin main
```

Create or reactivate the MERT environment:

```bash
python3 -m venv "/projectnb/rise-grid/venvs/$USER/mert"  # first time only
source "/projectnb/rise-grid/venvs/$USER/mert/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[mert]"
```

MERT needs the current fingerprinted Step-5 window audio—not `download_data.py` and not the retired
9-class archives. Use the shared project data root:

```bash
export RISE_DATA_ROOT=/projectnb/rise-grid/rise-data
```

If `$RISE_DATA_ROOT/pipeline/windows.csv` or `$RISE_DATA_ROOT/work/windows/` is absent, submit the
CPU preparation job and hold the train/validation GPU job behind it:

```bash
prep_job=$(qsub -terse -v RISE_DATA_ROOT="$RISE_DATA_ROOT" scc/mert_prepare.qsub)
qsub -hold_jid "$prep_job" -v RISE_DATA_ROOT="$RISE_DATA_ROOT" scc/mert_probe.qsub
qstat -u "$USER"
```

If the corrected Step-5 data already exists, submit only:

```bash
qsub -v RISE_DATA_ROOT="$RISE_DATA_ROOT" scc/mert_probe.qsub
```

That GPU job extracts only train/validation embeddings and selects the probe using validation
macro-F1. It cannot access test. When it finishes, inspect and freeze:

```text
artifacts/mert/validation_summary.json
artifacts/mert/validation_search.csv
```

Only after accepting the validation choice, submit the separate final job:

```bash
qsub -v RISE_DATA_ROOT="$RISE_DATA_ROOT" scc/mert_finalize.qsub
```

The final job refits on train+validation, extracts test with the exact saved MERT revision, evaluates
test once, and writes a guard record that prevents a second test access.

Before freezing the shared noise grid, run the validation-only MERT SNR pilot with `best_probe.pt`
(not the train+validation `final_probe.pt`):

```bash
qsub -v RISE_DATA_ROOT="$RISE_DATA_ROOT",MERT_OUTPUT_DIR="$PWD/artifacts/mert" \
  scc/mert_snr_pilot.qsub
```

The job writes `snr_pilot.json` beside the MERT validation artifacts and never reads test audio.
It pilots white noise by default. With ESC-50 available, pass
`MERT_PILOT_NOISE=white:natural:mechanical` and `RISE_NOISE_ROOT` to check all three categories
before freezing the grid.

## Shared noise sweep, SVM, and MERT

The clean final models above remain frozen. Put ESC-50 on the shared filesystem with both its
`audio/` directory and `meta/esc50.csv`, then set:

```bash
cd /project/rise-grid/Tariq/instrument-robustness
git switch main
git pull --ff-only origin main

export RISE_DATA_ROOT=/projectnb/rise-grid/rise-data
export RISE_NOISE_ROOT=/projectnb/rise-grid/noise-sources
```

Generate the shared noisy test set once:

```bash
noise_job=$(qsub -terse \
  -v RISE_DATA_ROOT="$RISE_DATA_ROOT",RISE_NOISE_ROOT="$RISE_NOISE_ROOT" \
  scc/noise_generate.qsub)
```

Hold the CPU SVM evaluation and GPU MERT evaluation behind that same generation job:

```bash
qsub -hold_jid "$noise_job" \
  -v RISE_DATA_ROOT="$RISE_DATA_ROOT" \
  scc/svm_noise.qsub

qsub -hold_jid "$noise_job" \
  -v RISE_DATA_ROOT="$RISE_DATA_ROOT" \
  scc/mert_noise.qsub
```

Generated WAVs and provenance remain under
`$RISE_DATA_ROOT/work/windows_noisy/`. Compact predictions, metrics, and summaries are written to
the repository under `artifacts/svm/noise/` and `artifacts/mert/noise/`.
