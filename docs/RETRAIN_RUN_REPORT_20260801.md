# Retrain run report — 2026-08-01, updated 2026-08-02

Session goal: pull everything, verify the dataset is standardised down to the split seed, retrain
all six models on SCC, report back.

> ## STATUS AS OF 2026-08-02 20:17 EDT — RETRAIN IS RUNNING
>
> The blocker in §4 was cleared. **The dataset was rebuilt and sealed successfully**, and all six
> models are training against it.
>
> ```
> rows after articulation filter: 8378
> rows after conflict exclusion  : 8374     <- the 4 DATA-001 sources are gone
> sealed 8374 windows
> dataset fingerprint: 97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf
> PIPELINE_EXIT=0                            <- all 10 stages
> ```
>
> New split, from `validate_mert_windows()`: **train 5,861 / val 1,258 / test 1,255 = 8,374**
> (was 5,864 / 1,259 / 1,255 = 8,378). Removing the four sources shifted three out of train and
> one out of validation — the pitch-group assignment is deterministic given its input, and the
> input changed, so a small redistribution is expected rather than alarming.
>
> ## ALL SIX COMPLETE — 2026-08-02 22:13 EDT
>
> | model | clean test macro-F1 | n |
> |---|---:|---:|
> | AST | **0.9908** | 1255 |
> | PANNs | **0.9868** | 1255 |
> | SVM | **0.9770** | 1255 |
> | CRNN | **0.9738** | 1255 |
> | CNN | **0.9708** | 1255 |
> | MERT | **0.8931** | 1255 |
>
> All six on the sealed build `97b1cdd2`, all n=1255, each through its sealed one-time finalizer.
>
> **Do not rank these finely.** SVM/AST/MERT/PANNs are single fixed-seed runs; only CNN and CRNN
> carry a 5-seed spread. There are no confidence intervals yet, and the audit's STAT-001 applies:
> differences smaller than the seed spread are not effects. SVM and MERT also changed protocol in
> `e4e455d` (train-only fit), so their numbers are not comparable to pre-`e4e455d` values.
>
> Sections 4 and 5 below are kept as the historical record of what blocked it. What actually had
> to be fixed is in §9.

---

## 1. Pull — done

Local `main` fast-forwarded `e9f7f8d → e4e455d`. Two commits from PaperTP landed while this
session was running:

| commit | what it does |
|---|---|
| `75d81b2` | Repair dataset and seal model evaluation workflows |
| `e4e455d` | Fit SVM and MERT on train only; add dataset composition tables |

The cluster checkout at `/projectnb/rise-grid/maxliu2k/instrument-robustness` was at `7b7e565`
(far behind) and is now also at `e4e455d`. `git pull` over HTTPS needed no credentials.

**MODEL-002 was fixed independently by PaperTP.** `train_ast` now uses
`score = val_metrics["macro_f1"]` and no longer builds a test loader. My own uncommitted fix for
the same issue was redundant and was dropped rather than merged. My documentation corrections
(`internal/plan.md`, `internal/AUDIT_CHECKLIST.md`) were kept, because they record the retraction and one gap
that is still open — see §6.

---

## 2. Is the dataset standardised down to the split seed? — yes, and it is PaperTP's design

No new schema was needed. `75d81b2` already implements what the audit's DATA-001 called for:

- **`config.CONFLICTING_LABEL_PATHS`** names the four byte-identical, conflicting-label sources.
- **`step0_filter.exclude_conflicting_labels`** drops them, and **raises** if any is absent from
  `manifest.csv` — so a silent no-op is impossible.
- **`freeze_dataset.py`** (new 10th pipeline stage) hashes every window, refuses to seal if any
  exact-audio group carries more than one label, checks the label set equals `TARGET_LABELS`, and
  writes `pipeline/dataset_freeze.json` carrying the `dataset_fingerprint` and per-split counts.
- **`step3_split.assert_split_is_unsealed`** refuses to overwrite `splits.csv` while that seal
  exists. That is the "down to the split seed" guarantee.

**The one fact that makes a full retrain unavoidable:** `excluded_conflicting_label_paths` is
inside `config_fingerprint()`. The fingerprint therefore changes, so every existing feature array
and every existing checkpoint is now correctly rejected by its own provenance check. All six
models genuinely must be retrained. This was verified by reading `config.py`, not assumed.

---

## 3. What is staged on SCC

Cluster state established this session:

| item | status |
|---|---|
| OnDemand session | live, authenticated as `maxliu2k`; no credentials were entered |
| Disk | `/projectnb` **25 G free**, `/project` 40 G — the docs' "1.15 GB" figure is stale |
| Raw audio | present, shared at `/projectnb/rise-grid/rise-data/` |
| ESC-50 | present at `/projectnb/rise-grid/noise-sources/ESC-50-master` |
| Main venv | fixed — was missing `mutagen`; `pip install -e .` run, all core imports pass |
| `venv_pretrained` | **built OK** (transformers + panns-inference + torchlibrosa) — serves AST and PANNs |
| MERT venv | **FAILED** — see §5 |
| PANNs CNN14 checkpoint | **found** at `/projectnb/rise-grid/gavinhu/rise-data/philharmonia/checkpoints/Cnn14_mAP=0.431.pth` — no external download needed |

Two job scripts were written and are committed on the cluster checkout:

- **`scc/pipeline_rebuild.qsub`** — full 10-stage rebuild, 4 slots, node-local `TMPDIR` and
  `NUMBA_CACHE_DIR` (the fix for the old `BrokenProcessPool`).
- **`scc/rise_cpu_train.qsub`** — takes `-v RISE_MODEL=svm|cnn|crnn`, trains then finalizes, CPU
  only (avoids the `gpu_c=6.0` / cu128 P100 mismatch still present in `scc/cnn_train.qsub`).

---

## 4. The blocker

`prep_data` dies in one second, every time:

```
File ".../prep_data.py", line 207, in ensure_skeleton
    d.mkdir(parents=True, exist_ok=True)
FileExistsError: [Errno 17] File exists: '.../all-samples/work'
```

`all-samples/` is a **symlink farm** into shared `/projectnb/rise-grid/rise-data/`, and
`work -> /projectnb/rise-grid/rise-data/work` is a **broken symlink** — the target does not exist.
`Path.mkdir(exist_ok=True)` raises on a broken symlink, because the path exists but is not a
directory.

Two consequences, the second more serious than the first:

1. The pipeline cannot create its working tree.
2. **The existing `windows.csv` points at window audio that is gone.** The old build was not
   merely stale; its audio is unreachable. Nothing that reads windows can succeed against this
   data root until it is rebuilt.

### The fix

I attempted this twice and was blocked both times by the permission classifier — once for a
`rm`/`cp --remove-destination` combination, once for an additive `mkdir`/`ln -sfn` build of a
fresh root. I did not attempt to route around it. Run whichever you prefer:

**Option A — repair in place** (smallest change; `work` is a broken link, so nothing is lost):

```bash
cd /projectnb/rise-grid/maxliu2k/all-samples && rm -f work && mkdir -p work
```

**Option B — fresh root, nothing deleted** (leaves the broken one intact for inspection):

```bash
NEW=/projectnb/rise-grid/maxliu2k/all-samples-v2; SRC=/projectnb/rise-grid/rise-data; mkdir -p $NEW/work $NEW/features $NEW/pipeline && for d in bassoon cello clarinet double-bass flute french-horn oboe trombone trumpet tuba viola violin raw archives; do [ -e $SRC/$d ] && ln -s $SRC/$d $NEW/$d; done && cp $SRC/manifest.csv $SRC/manifest_fingerprint.json $NEW/
```

With Option B, pass `RISE_DATA_ROOT=/projectnb/rise-grid/maxliu2k/all-samples-v2` everywhere.

One thing to decide either way: in the current layout `manifest.csv` and
`manifest_fingerprint.json` are symlinks into **shared** `rise-data/`. `prep_data` rewrites both,
so an in-place run writes the new fingerprint into space `gavinhu` and `tariqhsn` also read.
Option B avoids this by making them local copies. I would not run Option A without checking with
the others first.

### Then

```bash
cd /projectnb/rise-grid/maxliu2k/instrument-robustness
qsub scc/pipeline_rebuild.qsub
qsub -N svm_train  -hold_jid rise_pipeline -v RISE_MODEL=svm  -o svm_train.log  scc/rise_cpu_train.qsub
qsub -N cnn_train  -hold_jid rise_pipeline -v RISE_MODEL=cnn  -o cnn_train.log  scc/rise_cpu_train.qsub
qsub -N crnn_train -hold_jid rise_pipeline -v RISE_MODEL=crnn -o crnn_train.log scc/rise_cpu_train.qsub
```

AST and MERT chains are in §5. All jobs from this session were cancelled (`qdel`); the queue is
clean.

---

## 5. Remaining blockers

**MERT venv — permission denied.** `scc/mert_*.qsub` default `MERT_VENV` to
`/projectnb/rise-grid/venvs/$USER/mert`, but that directory is not writable:

```
mkdir: cannot create directory '/projectnb/rise-grid/venvs/maxliu2k': Permission denied
```

Only `venvs/allanyu/` exists. Either ask PaperTP to widen permissions on `venvs/`, or override
with `-v MERT_VENV=/projectnb/rise-grid/maxliu2k/venv_mert` and build it there. It must stay
separate from `venv_pretrained`: `[mert]` pins `transformers>=4.38,<4.39` and `[pretrained]` does
not.

**PANNs checkpoint is Gavin's copy.** It exists but under `gavinhu/`. Symlink or copy it to
`$RISE_DATA_ROOT/checkpoints/Cnn14_mAP=0.431.pth`. Worth confirming with him rather than reaching
into his tree silently. Note the audit's MODEL-003 separately: `train_panns` does not record or
enforce the base checkpoint's hash, so which copy gets used is currently unrecorded.

---

## 6. Two defects found and one still open

**`-hold_jid` releases on completion, not success.** When the first pipeline attempt failed after
one second, SGE released the three training jobs held behind it and they began scheduling against
the *previous* build. This is exactly the silent-wrong-result failure mode this repo's provenance rules exist to
prevent, and nothing in the repo would have caught it. `scc/rise_cpu_train.qsub` now opens with a
gate that reads `dataset_freeze.json`, compares its `dataset_fingerprint` against a freshly
computed `dataset_build_identity()`, and refuses to train on mismatch or absence. If it fires, the
bug it has found is "the pipeline did not actually rebuild the data this run".

**`ensure_skeleton` cannot survive a broken symlink.** `d.mkdir(parents=True, exist_ok=True)` in
`prep_data.py:207` is correct for a missing path and for an existing directory, but raises on a
broken link — which is precisely the state a half-migrated data root lands in. Worth making it
explicit rather than letting the traceback explain it.

**Still open — `summarize_results` will not flag stale AST.** `clean_row` consults
`selection_metric` only when the result has no explicit `macro_f1`. That is true of the CNN/CRNN
5-seed summaries it was written for, but AST's `metrics.json` records `test.macro_f1`, so the
branch never runs and a balanced-accuracy-selected AST row still prints `canonical`. Making the
check unconditional would be right for AST but would falsely fail PANNs, which early-stops on
validation macro-F1 yet writes no `selection_metric` field. Correct fix: emit `selection_metric`
from `train_panns`, then make the check unconditional.

---

## 7. On the keepalive script

I did not write one, deliberately. Holding an OnDemand shell open is solving the wrong problem —
`qsub` batch jobs are detached from the login session and keep running after logout, which is what
a scheduler is for. Every job in §4 survives the session dropping. Deliberately defeating an
institutional session timeout would also be circumventing a security control on BU's systems, and
it buys nothing here.

Check progress on return with `qstat -u maxliu2k` and the `*.log` files in the repo root.

---

## 8. Uncommitted local changes

Left in the working tree, not committed or pushed:

| file | change |
|---|---|
| `internal/plan.md` | AST status → MUST BE RETRAINED; "all six" claim corrected; `summarize_results` gap recorded |
| `internal/AUDIT_CHECKLIST.md` | #10 retraction + the gap in §6 |
| `README.md` | `Tariq.txt` protection notice; `configs/` layout line |
| `configs/data/irmas.yaml` | deleted (0 bytes, no consumer) |

`Tariq.txt` gained three lines from `e9f7f8d` (tariqhsn-bu), so the README's "26 bytes" is now
stale and should be reworded before committing.

---

## 9. What actually had to be fixed (2026-08-02)

Four defects stood between the staged jobs and a running retrain. Three were real repo bugs, not
environment quirks.

### 9.1 `work` was a broken symlink — cleared

Fixed before this session resumed. `all-samples/work` is now a real directory.

### 9.2 Shared `manifest.csv` was not writable — repo-adjacent, fixed

`prep_data` got as far as building the manifest and then died:

```
building manifest ...
  ! unreadable: viola_D6_05_piano_arco-normal.mp3: can't sync to MPEG frame
PermissionError: [Errno 13] Permission denied:
  '/projectnb/rise-grid/maxliu2k/all-samples/manifest.csv'
```

`manifest.csv` and `manifest_fingerprint.json` were symlinks into shared `rise-data/`, which this
account cannot write. Teammate consent was irrelevant — the filesystem refused. Both were replaced
with local copies (`cp --remove-destination`). The instrument audio, `raw/` and `archives/` remain
symlinked and read-only, so nothing shared was modified.

### 9.3 Seven SCC job scripts never `module load` — REAL BUG, fixed

`scc/{train_ast,ast_finalize,mert_prepare,mert_probe,mert_finalize,panns_train,panns_finalize}.qsub`
all activate a venv without loading the Python module it was built from. Every one of them died
instantly:

```
python: error while loading shared libraries: libpython3.9.so.1.0:
        cannot open shared object file: No such file or directory
```

`plan.md` documents this exact failure and the scripts still shipped without the fix; only
`scc/cnn_train.qsub` had it. `module load python3/3.9.9` was inserted immediately before the
`source .../activate` line in all seven. **This change is on the cluster checkout only and needs
committing.**

### 9.4 `urllib3` v2 vs SCC's OpenSSL — environment, fixed

Both pretrained venvs failed to import `huggingface_hub`:

```
ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl'
module is compiled with 'OpenSSL 1.0.2o-fips'
```

SCC's `python3/3.9.9` is built against OpenSSL 1.0.2. Pinned `urllib3<2` in both. Worth adding to
`pyproject.toml` or the SCC docs — anyone building a venv on this cluster hits it.

### Venvs as built

| venv | transformers | serves |
|---|---|---|
| `/projectnb/rise-grid/maxliu2k/venv` | — | SVM, CNN, CRNN |
| `/projectnb/rise-grid/maxliu2k/venv_pretrained` | 4.57.6 | AST, PANNs |
| `/projectnb/rise-grid/maxliu2k/venv_mert` | 4.38.2 | MERT |

The `[mert]` pin (`>=4.38,<4.39`) is respected and isolated, as `plan.md` requires. `venv_mert`
lives under the user's own space because `/projectnb/rise-grid/venvs/` is not writable.

The PANNs CNN14 base checkpoint was copied from `gavinhu/rise-data/philharmonia/checkpoints/` to
`$RISE_DATA_ROOT/checkpoints/` (327,428,481 bytes), with the user's confirmation that this was
permitted.

## 10. Things to watch

**CNN/CRNN epoch time is ~11x worse than previously measured.** `plan.md` records ~17.5 s/epoch on
8 slots; this run is at 204 s (CNN) and 192 s (CRNN). At 40 epochs x 5 seeds that is ~11 h against
a 12 h wall limit. Early stopping (patience 8) should land well inside it, and `train_cnn` is
resumable per seed, so a timeout costs only the unfinished seed. But if these jobs die at the wall,
that is why. Cause not diagnosed — could be node contention or missing thread-limit env vars in
`rise_cpu_train.qsub`.

**`-hold_jid` released the finalizers twice after an upstream failure.** It happened again when the
AST/MERT/PANNs training jobs died on libpython; the finalizers went to `qw` and had to be `qdel`ed
by hand. The CPU models are protected by the seal gate in `rise_cpu_train.qsub`; the pretrained
chains are not, and rely on their own fingerprint assertions. A shared gate for those is worth
adding.

**Uncommitted on the cluster only:** the seven `module load` insertions, plus
`scc/pipeline_rebuild.qsub`, `scc/rise_cpu_train.qsub` and `scc/build_mert_venv.qsub`. None of this
is on `main` yet.

### 9.5 `mert_prepare.qsub` re-runs the pipeline and now collides with the seal — REAL BUG, fixed

`scc/mert_prepare.qsub` lines 31–37 re-ran `prep_data` through `step5_normalize` before validating.
Since `75d81b2` added `assert_split_is_unsealed()`, that is guaranteed to fail on any sealed build:

```
RuntimeError: Dataset is sealed by .../pipeline/dataset_freeze.json.
Refusing to overwrite splits.csv.
```

The guard is correct; the script is what is wrong. Those seven lines were commented out — the
pipeline is built once by `scc/pipeline_rebuild.qsub`, and `mert_prepare` only needs
`validate_mert_windows()`, which it still runs and which passed. **This is a latent bug for anyone
who runs `mert_prepare.qsub` against a sealed build, and it needs committing.**

Note `extract_mert` is invoked from `scc/mert_probe.qsub`, not `mert_prepare.qsub`, despite the
names.
