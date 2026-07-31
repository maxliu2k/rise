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
python -m instrument_robustness.run_pipeline    # all nine stages
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

**Status: trained and finalized. The test evaluation is already spent.**

### CNN — `artifacts/cnn/`
Reads `features/cnn/{train,val}.npz`.

```bash
python -m instrument_robustness.train_cnn                    # seeds 42-46
python -m instrument_robustness.train_cnn --device cpu       # force CPU (see GPU note below)
python -m instrument_robustness.finalize_cnn                 # the one test evaluation
```

Resumable: each finished seed persists a checkpoint, its validation probabilities and a provenance
record, so re-running trains only what is missing. `validation_summary.json` lists `reused_seeds`.

**Status: MUST BE RETRAINED.** The existing 5-seed run (0.9523 ± 0.0082) selected its combiner on
balanced accuracy, before the project standardised on macro-F1, so `summarize_results` now reports
it STALE and `finalize_cnn` refuses it at gate 3. Two independent reasons to retrain, in fact: the
recorded input hashes also no longer match `features/*.npz`, because step6/7 were re-run and
np.savez embeds zip timestamps (the bytes change even when the arrays do not). Both refusals are
correct. Re-run `train_cnn`, then finalize.

### CRNN — `artifacts/crnn/`
Reads **the same `features/cnn/` arrays as the CNN** — it is a different consumer of one feature
set, not a separate featurization.

```bash
python -m instrument_robustness.train_crnn --device cpu
python -m instrument_robustness.finalize_crnn
```

**Status: MUST BE RETRAINED**, same reason as the CNN — the existing 5-seed run
(0.9598 ± 0.0056) selected on balanced accuracy and is reported STALE.

### AST — `artifacts/ast/`
Reads `pipeline/windows.csv` directly and resamples 22050 → 16000 in the DataLoader. Needs
`pip install -e ".[ast]"`.

```bash
python -m instrument_robustness.train_ast --epochs 10 --batch-size 8
```

**Status: trained. See the sealed-test warning below — AST has no `finalize_ast`.**

### MERT — `artifacts/mert/`
Two steps: cache frozen embeddings, then train a layer-weighted linear probe. Needs
`pip install -e ".[mert]"` in its **own** venv (see the pin warning below).

```bash
python -m instrument_robustness.extract_mert    # train + validation only
python -m instrument_robustness.train_mert
python -m instrument_robustness.finalize_mert   # after freezing validation_summary.json
```

**Status: trained and finalized. Test evaluation already spent.**

### PANNs — `artifacts/panns/`
Reads `windows.csv`. Requires the pretrained CNN14 checkpoint at
`$RISE_DATA_ROOT/checkpoints/Cnn14_mAP=0.431.pth` — it is not auto-downloaded.

```bash
python -m instrument_robustness.train_panns --mode probe      # or --mode finetune
```

**Status: trained, probe and finetune, Philharmonia and TinySOL. Released as
`v1.0-panns-12class`. See the sealed-test warning below.**

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

**Blocked: `--generate` needs ~14.8 GiB. `/projectnb/rise-grid` has ~1.15 GB free;
`/project/rise-grid` has ~34 GB.**

---

## 3. Things that will bite, in the order they will bite

**The test split is NOT sealed the same way across the six.** SVM, CNN, CRNN and MERT have a
`finalize_*` that refuses a second test evaluation via a status file and re-hashed inputs.
**AST and PANNs read the test split inside their training run** — `train_ast` builds a test loader
and writes `test_summary.csv`; `train_panns` loads all three splits and evaluates test inline.
There is no seal and nothing stops a re-run after seeing the number. That is a real inconsistency
in the six-model comparison and it is not something a run order can fix.

**The metric is now standardised on macro-F1** across all six models — `train_cnn` and
`train_crnn` select on `validation_macro_f1` like SVM and MERT already did. Balanced accuracy and
MCC are still recorded everywhere, per seed and per combiner, because `CLAUDE.md` and
`docs/FINDINGS.md` §7 are right that macro-F1 flatters a collapsed classifier under imbalance;
macro-F1 won on comparability, not on merit, and MCC stays as the collapse detector. The practical
consequence is that **CNN and CRNN must be retrained** before their numbers can sit in the same
table as the other four. See `docs/AUDIT_CHECKLIST.md` #10.

**GPU: `gpu_c=6.0` in `scc/cnn_train.qsub` admits P100s, and a `torch 2.8.0+cu128` venv ships no
kernels for them** — `no kernel image is available for execution on the device`. Either raise to
`gpu_c=7.0` or pass `--device cpu`. CPU is fine for CNN/CRNN: ~17.5 s/epoch on 8 slots.

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
