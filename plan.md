# Running all six models — plan of record

Written so the commands are auditable before they are run, and so a mistake is visible as a
mismatch against this document rather than as a wrong number three weeks later.

Every command below assumes:

```bash
export RISE_DATA_ROOT=/projectnb/rise-grid/$USER/all-samples   # or wherever your data root is
export PYTHONPATH=$PWD/src
```

On BU SCC also `module load python3/3.9.9` before activating a venv built from that module —
the venv is not self-contained and dies with `libpython3.9.so.1.0: cannot open shared object file`
otherwise.

---

## 0. The data, once, for everyone

```bash
python -m instrument_robustness.prep_data       # only if the audio is not already present
python -m instrument_robustness.run_pipeline    # all TEN stages, ending in freeze_dataset
```

`run_pipeline` sets its own thread limits, stops at the first failure, and prints the `--from`
invocation to resume with. Roughly 7 minutes warm.

**The six models do not all need the same amount of this.** Getting this wrong is the most likely
mistake, because a data root prepared for AST looks complete and is not:

| stage reached | who can run |
|---|---|
| `--to step5_normalize` | AST, MERT, PANNs — they carry their own extractors and read window audio |
| all nine stages | **SVM, CNN, CRNN** — they read the Step-7 feature arrays |

So: run all nine unless you have a reason not to. `scc/cnn_train.qsub` checks for
`features/cnn/{train,val}.npz` specifically rather than `windows.csv`, for exactly this reason.

---

## 1. Per model

Ordered by how much they depend on. Status is as of this document; check the artifacts, not this
table, before trusting it.

### SVM — `artifacts/svm/`
Reads `features/svm/{train,val}.npz`. No CLI arguments.

```bash
python -m instrument_robustness.train_svm       # RBF grid over C, gamma; selects on validation
python -m instrument_robustness.finalize_svm    # refits train+val, spends the ONE test evaluation
```

**Status: RETRAINED 2026-08-02 on sealed build `97b1cdd2`. Clean test macro-F1 **0.9770** (n=1255).
Test evaluation spent — `finalize_svm` refuses a second.**

### CNN — `artifacts/cnn/`
Reads `features/cnn/{train,val}.npz`.

```bash
python -m instrument_robustness.train_cnn                    # seeds 42-46
python -m instrument_robustness.finalize_cnn                 # the one test evaluation
```

Resumable: each finished seed persists a checkpoint, its validation probabilities and a provenance
record, so re-running trains only what is missing. `validation_summary.json` lists `reused_seeds`.

**Status: RETRAINED 2026-08-02 on sealed build `97b1cdd2`, seeds 42–46 on an A40.
Clean test macro-F1 **0.9708** (n=1255). Resumability confirmed: a mid-run restart reused the
already-banked seed 42 rather than repeating it.**

### CRNN — `artifacts/crnn/`
Reads **the same `features/cnn/` arrays as the CNN** — it is a different consumer of one feature
set, not a separate featurization.

```bash
python -m instrument_robustness.train_crnn      # defaults to cuda; see the GPU note below
python -m instrument_robustness.finalize_crnn
```

**Status: RETRAINED 2026-08-02 on sealed build `97b1cdd2`, seeds 42–46 on an L40S.
Clean test macro-F1 **0.9738**, ensemble MCC 0.9730 (n=1255).**

### AST — `artifacts/ast/`
Reads `pipeline/windows.csv` directly and resamples 22050 → 16000 in the DataLoader. Needs
`pip install -e ".[ast]"`.

```bash
python -m instrument_robustness.train_ast --epochs 10 --batch-size 8
```

**Status: RETRAINED 2026-08-02 on sealed build `97b1cdd2`. Clean test macro-F1 **0.9908**
(n=1255) — the best of the six. `train_ast` no longer reads test at all; `finalize_ast` spends
the one permitted evaluation and refuses a second.**

### MERT — `artifacts/mert/`
Two steps: cache frozen embeddings, then train a layer-weighted linear probe. Needs
`pip install -e ".[mert]"` in its **own** venv (see the pin warning below).

```bash
python -m instrument_robustness.extract_mert    # train + validation only
python -m instrument_robustness.train_mert
python -m instrument_robustness.finalize_mert   # after freezing validation_summary.json
```

**Status: RETRAINED 2026-08-02 on sealed build `97b1cdd2`. Clean test macro-F1 **0.8931**
(validation 0.9112, n=1255). Test evaluation spent.**

### PANNs — `artifacts/panns/`
Reads `windows.csv`. Requires the pretrained CNN14 checkpoint at
`$RISE_DATA_ROOT/checkpoints/Cnn14_mAP=0.431.pth` — it is not auto-downloaded.

```bash
python -m instrument_robustness.train_panns --mode probe      # or --mode finetune
```

**Status: RETRAINED 2026-08-02 on sealed build `97b1cdd2` (`--mode finetune`).
Clean test macro-F1 **0.9868** (n=1255). `finalize_panns` now provides the sealed boundary.
Requires the CNN14 base checkpoint — see the note below.**

---

## 2. Noise sweep — once centrally, then per model

```bash
export RISE_NOISE_ROOT=~/Downloads/noise_sources/ESC-50-master   # needs audio/ AND meta/esc50.csv

python -m instrument_robustness.noise_sweep --validate         # measures SNR back out; writes samples
python -m instrument_robustness.noise_sweep --generate         # ~14.8 GiB, once
python -m instrument_robustness.noise_sweep --check-generated

python -m instrument_robustness.noise_eval_svm
python -m instrument_robustness.noise_eval_cnn
python -m instrument_robustness.noise_eval_crnn
python -m instrument_robustness.noise_eval_ast
python -m instrument_robustness.noise_eval_mert --device cuda
python -m instrument_robustness.noise_eval_panns
```

Generate **once**. Every model must read the same files or predictions are not paired, and the
paired bootstrap and sign test in `noise_stats.py` require pairing.

Each adapter reproduces its official clean score before any noisy condition is scored, and aborts
on mismatch.

**No longer blocked on disk.** Measured 2026-08-02: `/projectnb` has **25 G free** and
`/project` 40 G. The "~1.15 GB" figure this document used to carry is stale. ESC-50 is already
extracted at `/projectnb/rise-grid/noise-sources/ESC-50-master`.

**Regenerate from scratch.** The noise seed is
`sha256(dataset_fingerprint|window_id|noise_type|replicate)`, and the fingerprint changed to
`97b1cdd2` with the DATA-001 rebuild — so every previously generated mixture is invalid.

---

## 3. Things that will bite, in the order they will bite

**The test split is now sealed the same way across all six.** Commit `75d81b2` added
`finalize_ast.py` and `finalize_panns.py`, so every model has a `finalize_*` that spends exactly
one test evaluation and refuses a second via a status file and re-hashed inputs. `train_ast` and
`train_panns` no longer load the test split at all — AST prints *"test remains sealed; run
finalize_ast exactly once"* and stops. This was the audit's MODEL-001 and it is closed.

**The metric is now standardised on macro-F1** across all six models — `train_cnn`,
`train_crnn` and `train_ast` select on `validation_macro_f1` like SVM and MERT already did.
Balanced accuracy and MCC are still recorded everywhere, per seed and per combiner, because
`CLAUDE.md` and `docs/FINDINGS.md` §7 are right that macro-F1 flatters a collapsed classifier
under imbalance; macro-F1 won on comparability, not on merit, and MCC stays as the collapse
detector. **All six were retrained on 2026-08-02 under this metric**, so the table is now
internally consistent. See `docs/AUDIT_CHECKLIST.md` #10.

> AST was standardised late. Commit `48ff616` claimed all six models while leaving `train_ast`
> selecting on `(balanced_accuracy, mcc, accuracy)`; the audit caught it as MODEL-002. The code is
> fixed, but **every saved AST artifact predates the fix** and was selected on balanced accuracy —
> including the canonical `new-ast-results-20260730-022036` (test macro-F1 0.9917) and every AST
> noise result derived from it. Those numbers are not wrong, but they answer a different selection
> question from the other five and must not be tabled beside them until AST is retrained.
>
> `summarize_results` will NOT flag this for you. Its staleness branch only runs when a result has
> no explicit `macro_f1`, and AST's does — so AST still prints `canonical`. Treat the AST row as
> stale by hand until that gate is widened.

**GPU: `gpu_c=6.0` in `scc/cnn_train.qsub` admits P100s, and a `torch 2.8.0+cu128` venv ships no
kernels for them** — `no kernel image is available for execution on the device`. **Raise it to
`gpu_c=7.0`. Do not "fix" it by passing `--device cpu`.** `scc/rise_train.qsub` already asks for
7.0 and is the supported entry point.

An earlier revision of this file claimed "CPU is fine for CNN/CRNN: ~17.5 s/epoch on 8 slots".
**That number was never measured and is retracted.** Measured on the 8,374-window build,
seeds 42–46:

| | CPU (8 slots) | GPU (A40 / L40S) |
|---|---:|---:|
| CNN | 192 s/epoch | **1.4 s/epoch** |
| CRNN | 192 s/epoch | **0.8 s/epoch** |

CRNN finished five seeds, the ensemble and `finalize_crnn` in **2 min 42 s** on an L40S. On CPU
the same work projected to roughly 8.5 h. Thread-limit env vars were tested as a hypothesis for
the CPU figure and made no difference (192.6 s → 192.0 s); the GPU request is the whole effect.

**Do not install `.[ast]` and `.[mert]` into one venv.** `[mert]` pins
`transformers>=4.38,<4.39`; `[ast]` does not, and installing it second can upgrade past the pin.
Separate venvs.

**`ProcessPoolExecutor` respects `NSLOTS`** now — but only since the `worker_count` fix. Older
checkouts oversubscribe a batch allocation and die with `BrokenProcessPool`.

**Re-running step6/step7 changes the feature file hashes even when the arrays are identical**
(zip timestamps). Anything finalized against the old files will refuse. Plan the order: build
features once, then train, then finalize.

---

## 4. What I would not do without asking

- Merge `stale/cnn-ensemble` into `main`. Different preprocessing, different label indices.
  It is an archive; `docs/FINDINGS.md` carries a header saying which of its numbers are superseded.
- Re-run `finalize_*` for SVM or MERT. Those test evaluations are spent, and re-spending them
  after seeing a number is selection on test.
- Change the SNR grid or replicate count. Both were set from `snr_pilot` measurements, and the
  old inherited grid was entirely at or below chance.
