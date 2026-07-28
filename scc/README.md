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

The job file assumes the repository is cloned at:

```text
/project/rise-grid/repos/<SCC username>/instrument-robustness
```

Create a per-user clone from an SCC OnDemand terminal:

```bash
mkdir -p "/project/rise-grid/repos/$USER"
cd "/project/rise-grid/repos/$USER"
git clone https://github.com/maxliu2k/rise.git instrument-robustness
cd instrument-robustness
git switch allan/MERT
```

MERT needs the full Step-5 window audio, not only the saved SVM/CNN features. Keep the large data in
the non-backed-up project space:

```bash
mkdir -p /projectnb/rise-grid/rise-data
export RISE_DATA_ROOT=/projectnb/rise-grid/rise-data
python download_data.py
```

Create the environment after selecting an available SCC Python module:

```bash
module avail python3
module load python3/<available-version>
python -m venv "/projectnb/rise-grid/venvs/$USER/mert"
source "/projectnb/rise-grid/venvs/$USER/mert/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[mert]"
```

Submit from the repository root. The job requests one GPU, extracts only train/validation embeddings,
and then tunes the frozen probe on validation macro-F1:

```bash
qsub scc/mert_probe.qsub
qstat -u "$USER"
```

The job intentionally has no MERT test-extraction or test-evaluation path yet.
