# BU SCC setup for the MERT probe

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
