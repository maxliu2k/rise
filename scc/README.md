# BU SCC jobs

## Start here — the whole six-model run

Verified end to end on 2026-08-02. Every model below was trained and finalized against sealed
build `97b1cdd2` this way.

```bash
export RISE_REPO=/projectnb/rise-grid/$USER/instrument-robustness
export RISE_DATA_ROOT=/projectnb/rise-grid/$USER/all-samples
export RISE_VENV=$RISE_REPO/.venv                      # numpy/sklearn/librosa/torch
export VP=/projectnb/rise-grid/$USER/venv_pretrained   # AST + PANNs
export VM=/projectnb/rise-grid/$USER/venv_mert         # MERT only (pinned transformers)

qsub -v RISE_REPO=$RISE_REPO scc/build_venvs.qsub                       # once
qsub -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT scc/pipeline_rebuild.qsub

# the three from-scratch models (train + finalize, seal-gated)
for m in svm cnn crnn; do
  qsub -N ${m}_train -hold_jid rise_pipeline        -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,RISE_MODEL=$m        -o ${m}_train.log scc/rise_train.qsub
done

# the three pretrained models
qsub -N ast_train -hold_jid rise_pipeline -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,AST_VENV=$VP,AST_OUTPUT_DIR=$RISE_REPO/artifacts/ast -o ast_train.log scc/train_ast.qsub
qsub -N ast_final -hold_jid ast_train    -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,AST_VENV=$VP -o ast_final.log scc/ast_finalize.qsub
qsub -N mert_probe -hold_jid rise_pipeline -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,MERT_VENV=$VM,HF_HOME=/projectnb/rise-grid/$USER/hf_cache -o mert_probe.log scc/mert_probe.qsub
qsub -N mert_final -hold_jid mert_probe    -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,MERT_VENV=$VM,HF_HOME=/projectnb/rise-grid/$USER/hf_cache -o mert_final.log scc/mert_finalize.qsub
qsub -N panns_train -hold_jid rise_pipeline -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,PANNS_VENV=$VP,PANNS_MODE=finetune -o panns_train.log scc/panns_train.qsub
qsub -N panns_final -hold_jid panns_train   -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,PANNS_VENV=$VP -o panns_final.log scc/panns_finalize.qsub
```

### Five things that will waste your evening if you skip them

1. **`module load python3/3.9.9` is mandatory.** A venv built from that module is not
   self-contained; without the module every job dies instantly with
   `libpython3.9.so.1.0: cannot open shared object file`. All the scripts here now do it
   themselves — they did not until 2026-08-02, which is why AST, MERT and PANNs all failed on
   first submission.

2. **`urllib3<2`.** The module's Python is built against OpenSSL 1.0.2; urllib3 v2 needs 1.1.1+.
   Without the pin, importing `huggingface_hub` fails. `scc/build_venvs.qsub` applies it. It is
   deliberately NOT pinned in `pyproject.toml` — it is an SCC constraint, not a project one.

3. **Ask for a GPU, and ask for `gpu_c=7.0`.** Measured on the 8,374-window build: CNN 192 s/epoch
   on 8 CPU slots vs **1.4 s/epoch** on an A40; CRNN 192 s vs **0.8 s**. `gpu_c=6.0` admits P100s,
   which a `torch 2.8.0+cu128` venv has no kernels for — the right response is 7.0, not
   `--device cpu`.

4. **`-hold_jid` releases on COMPLETION, not success.** A pipeline that fails in one second still
   green-lights everything queued behind it. `scc/rise_train.qsub` therefore opens with a gate
   that reads `dataset_freeze.json` and refuses to train unless its fingerprint matches the
   current config. The pretrained chains rely on their own fingerprint assertions instead.

5. **Shared `rise-grid` paths are owned by individuals and are not group-writable.**
   `/projectnb/rise-grid/{huggingface,venvs}` and the shared `rise-data/manifest.csv` each caused
   a `Permission denied` failure. Override `HF_HOME`, `MERT_VENV` and friends into your own space,
   or have the owner run `chmod g+w`.


## Superseded per-model sections

The AST, MERT, noise, and CNN/CRNN sections that used to live here have been removed rather than
left to rot. They described a different world:

- **Different roots.** They used `/project/rise-grid/repos/$USER/...`,
  `/project/rise-grid/Tariq/...` and `RISE_DATA_ROOT=.../rise-data/philharmonia`. Following one
  section and then another gave you two different data roots, silently.
- **Different workflow.** They said CNN and CRNN were validation-only and needed a manual
  finalize step. `scc/rise_train.qsub` trains AND finalizes in one job, behind a seal gate.
- **Nine stages.** The pipeline has ten; the tenth, `freeze_dataset`, is what seals the build.

Everything they covered is in "Start here" above, which was verified end to end on 2026-08-02.

For the noise sweep specifically:

```bash
qsub -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,RISE_NOISE_ROOT=/projectnb/rise-grid/noise-sources/ESC-50-master      scc/noise_generate.qsub                       # ONCE. Every model must read these same files.

qsub -N svm_noise   -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT                  -o svm_noise.log   scc/svm_noise.qsub
qsub -N mert_noise  -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,NOISE_VENV=$VM   -o mert_noise.log  scc/mert_noise.qsub
for m in cnn crnn ast panns; do
  qsub -N ${m}_noise -v RISE_REPO=$RISE_REPO,RISE_DATA_ROOT=$RISE_DATA_ROOT,RISE_MODEL=$m -o ${m}_noise.log scc/rise_noise_eval.qsub
done
```

Generate **once**. Predictions are only paired if every model reads the same realized corpus, and
the paired bootstrap and cluster sign test in `noise_stats.py` require pairing. The audit's
NOISE-001 found saved PANNs results scored against a different corpus than SVM/MERT/AST, which
invalidated every cross-model comparison involving PANNs — that is what generating once prevents.
