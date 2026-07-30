# PANNs CNN14 — 12-class instrument classification

Trained separately on Philharmonia and TinySOL under identical methodology: one onset-aligned
3.0 s window per source file, pitch-group split, one articulation per class.

## Clean results

| Dataset | Mode | Val macro-F1 | Test macro-F1 | Test acc |
|---|---|---|---|---|
| Philharmonia | probe (frozen trunk) | 0.887 | 0.883 | 0.885 |
| Philharmonia | finetune | 0.989 | 0.985 | 0.984 |
| TinySOL | probe (frozen trunk) | 0.780 | 0.826 | 0.788 |
| TinySOL | finetune | 0.987 | 0.994 | 0.990 |

## Cross-dataset result — the headline

Each finetune checkpoint scored on the **other** dataset's audio, no retraining. 12-way decision,
chance 0.083. Full numbers and confusion-matrix commentary in `cross_dataset_eval.json`.

| Model | Within-test | Cross-dataset | Drop |
|---|---|---|---|
| Philharmonia | 0.985 | 0.391 (on TinySOL) | −0.594 |
| TinySOL | 0.994 | 0.139 (on Philharmonia) | −0.855 |

Within-dataset accuracy of ~0.99 radically overstates learned timbre in both directions. The
asymmetry matters: the Philharmonia model retains 4.7x chance and fails in timbrally plausible ways
(viola/violin), while the TinySOL model retains only 1.7x chance and **mode-collapses** onto a few
output classes — the signature of a model keyed to recording conditions rather than instrument
identity. TinySOL is the more acoustically homogeneous corpus, so it offers more surface-level
shortcuts to exploit.

## What is here, and what is not

Committed (small):

- `panns_probe_{philharmonia,tinysol}.pt` — ~100 KB linear heads
- `results_{probe,finetune}_{philharmonia,tinysol}.json` — val/test macro-F1, accuracy, per-class F1
- `cross_dataset_eval.json` — the cross-dataset numbers above
- `*.provenance.json` — one per checkpoint, including the two that live in the release

Distributed via **GitHub Release `v1.0-panns-12class`** (~312 MB each, too large to track):

- `panns_finetune_philharmonia.pt`
- `panns_finetune_tinysol.pt`

Release assets are used rather than Git LFS deliberately: LFS on this account has a shared 1 GB
storage / 1 GB-per-month bandwidth budget, and 624 MB of checkpoints would exhaust the bandwidth on
the first teammate's clone. Release assets fall outside the LFS billing system entirely, and
attaching weights under 2 GB to a release is what PyTorch's own `torch.hub` documentation
recommends.

## Verifying a downloaded checkpoint

Every `.provenance.json` records the checkpoint's own `checkpoint_sha256`, so a download can be
checked without trusting the filename or where it came from:

```bash
curl -L -o panns_finetune_philharmonia.pt \
  https://github.com/maxliu2k/rise/releases/download/v1.0-panns-12class/panns_finetune_philharmonia.pt
shasum -a 256 panns_finetune_philharmonia.pt   # compare with checkpoint_sha256
```

## Why provenance sidecars exist

Both datasets use the same 12-label canonical config, so `config_fingerprint()` is **byte-identical**
between a Philharmonia checkpoint and a TinySOL one. `label_order` does not distinguish them either
(same 12 labels, same order). Without a sidecar, the only thing separating the two is the filename —
and a mislabeled checkpoint would load cleanly, assert cleanly, and produce plausible-looking
numbers against the wrong dataset. That is exactly the failure the fingerprint system exists to
prevent, and it is most dangerous in cross-dataset work, where the entire claim is "trained on A,
tested on B".

Each sidecar therefore records what the fingerprint cannot: the dataset's `manifest_sha256`
(Philharmonia `83b0025c...`, TinySOL `fa01c93a...`) and its `manifest_producer_stage` (`prep_data`
vs `build_tinysol_manifest`) — fields that genuinely differ — alongside the checkpoint's own hash.

This is deliberately kept *beside* `config_fingerprint()` rather than added *inside* it. Adding a
dataset field to the fingerprint would change every fingerprint in the project, invalidating every
manifest, feature array, and checkpoint on both datasets at once. The identity gap is real and worth
closing properly in `train_panns.py` (recording the manifest hash at save time), but that is a
change to shared training code and belongs in its own discussion.

## Reproducing

```bash
python -m instrument_robustness.train_panns --mode finetune          # per dataset, via RISE_DATA_ROOT
RISE_TINYSOL_ROOT=/path/to/TinySOL2020 \
  python -m instrument_robustness.cross_dataset_eval \
    --phil-model panns_finetune_philharmonia.pt \
    --tiny-model panns_finetune_tinysol.pt
```

On BU SCC, `scc/panns_train.qsub` and `scc/cross_dataset_eval.qsub` wrap these with the correct
GPU request, thread limits, and data-root guards.
