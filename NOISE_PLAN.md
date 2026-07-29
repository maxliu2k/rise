# Noise robustness sweep — design and protocol

How we measure the degradation of a **clean-trained** model as noise increases. Inference only:
no model is retrained, and the training and validation splits are never touched.

Implemented in `noise_sweep.py` (generates and fingerprints the audio),
`noise_eval_{svm,mert,panns}.py` (model adapters), `noise_eval_common.py` (the shared fail-closed
evaluation contract), and `noise_stats.py` (paired cluster bootstrap + cluster sign test).

---

## 1. The one rule that makes cross-model comparison possible

**The noisy audio is generated ONCE, centrally, and every model reads the same files.**

If each model generated its own noise, two models would be scored on different audio and their
predictions would no longer be *paired*. Paired predictions are what McNemar and the paired
bootstrap require — without them we can report each model's accuracy but cannot say whether the
difference between two models is real.

Run `noise_sweep.py --generate` once per dataset, then point every model at `work/windows_noisy/`.

## 2. Conditions — 16 total

| | 20 dB | 10 dB | 5 dB | 0 dB | −5 dB |
|---|---|---|---|---|---|
| **white** (Gaussian) | ✓ | ✓ | ✓ | ✓ | ✓ |
| **natural** (ESC-50) | ✓ | ✓ | ✓ | ✓ | ✓ |
| **mechanical** (ESC-50) | ✓ | ✓ | ✓ | ✓ | ✓ |

plus **clean**, shared rather than duplicated → 3 × 5 + 1 = **16**.

For the current 1,310-window Philharmonia test split, the 15 float32 noisy conditions require
roughly 5.2 GB before filesystem overhead. Keep them under the external data root, never in Git.

Lower SNR = more noise. 20 dB is mild, 0 dB is signal and noise at *equal power*, −5 dB means the
noise is louder than the instrument.

- **white** — Gaussian, flat across all frequencies.
- **natural** — ESC-50 targets 0–19 (animals; natural soundscapes and water).
- **mechanical** — ESC-50 targets 30–49 (interior/domestic; exterior/urban).

ESC-50's human-non-speech block (20–29) is deliberately **excluded**: it is neither ambient-natural
nor mechanical, and its speech-like transients would form a third character rather than belong to
either. The split leaves **800 clips per category**, so neither is better sampled.

DEMAND was considered and dropped — two well-separated real-noise characters from one corpus is
enough, and it avoids a ~6 GB dependency.

## 3. SNR is defined by power, not amplitude

```python
p_sig   = mean(clean ** 2)
p_noise = mean(noise ** 2)
alpha   = sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
noisy   = clean + alpha * noise
```

Getting this wrong (amplitude instead of power, or the wrong sign) produces files that *look*
plausible but sit at the wrong SNR. See §6.

## 4. One noise realization, scaled to every SNR

The seed **excludes the SNR**:

```python
seed = sha256(f"{dataset_fingerprint}|{window_id}|{noise_type}").digest()[:4]
```

A single noise waveform is drawn per (window, noise type) and then *scaled* to 20/10/5/0/−5 dB.
If each level drew fresh noise, part of the drop along the curve would be **noise variability**
rather than noise **level** — the comparison would confound "more noise" with "different noise".
Scaling one realization isolates the SNR axis.

`--validate` asserts this directly: the added component at each SNR must be a pure rescaling of
the same waveform (cosine similarity 1.0 across all levels).

The `dataset_fingerprint` hashes the actual `manifest.csv` and Step-5 `windows.csv` identities in
addition to the configuration. A noisy set built against one data build therefore cannot be
silently reused against another.

## 5. Two invariants that are easy to break

**Do NOT re-normalize after adding noise.** Note the reason carefully: rescaling the whole mixture
does **not** change its SNR — SNR is a power *ratio* and is invariant under a common scale factor.
The real reason is that Step 5 normalized every clean window to `TARGET_RMS`, and models were
trained at that reference gain; rescaling the mixture would present an amplitude distribution the
model never saw in training.

**Files are float32 WAV, not 16-bit PCM.** At −5 dB the mixture peaks well above ±1.0 (measured up
to 6.0). Clipping *is* nonlinear and genuinely corrupts the SNR, unlike a linear rescale. float32
has the headroom. **Verify your loader does not silently clamp values outside [−1, 1].**

## 6. Validation before generating anything

`noise_sweep.py --validate` samples a few windows, mixes at every SNR, then **measures the SNR back
out of the mixture** and compares to target. Current result: **0.000000 dB error** across all 15
noisy conditions. It also confirms the single-realization property (§4) and writes listenable 0 dB
samples to `work/windows_noisy/_validation_samples/` — a 0 dB clip should sound like an instrument
buried in noise, not silence or garbage.

Generation writes `noise_provenance.csv`, with the clean hash, ESC-50 source/hash, deterministic
seed, crop offset, scaling factor, realized SNR, peak, and output hash for every noisy file.
`noise_manifest.json` is written last as the completion marker and records the actual dataset
build, ESC-50 metadata/corpus identity, archive hash when the downloaded zip is retained, protocol,
file counts, provenance hash, and software versions. Evaluators refuse missing, partial, stale, or
mismatched manifests.

## 7. Clean-parity gate

Before any noisy condition is scored, the evaluator runs the **clean** files and must reproduce the
model's official clean macro-F1 to within `1e-3` **and** the official test-example count. A missing
official result is an error, not a skipped gate. Any mismatch aborts before noisy results are
written.

## 8. Featurization must match training

Each model featurizes the noisy window through **the exact code path it was trained with** — noise
is added to the 22050 Hz window *first*, then the model's own extractor runs:

- SVM computes the same 88 handcrafted features and applies the saved train-only mean/std.
- MERT uses the same pinned 24 kHz processor/backbone, hidden-state mean pooling, and final probe.
- PANNs uses `panns_input` at 32 kHz and the final fine-tuned CNN14.

Never recompute dataset normalization statistics on noisy audio. The Step-6 stats are part of the
trained model's contract.

## 9. Outputs

Under the repository's **`artifacts/<model>/noise/`** directory (not the external data root):

- `*_test_{condition}.csv` — one row per test window: `window_id`, `source_path`, `pitch_group`,
  `true_label`, `predicted_label`, `correct`, and one column per class.
- `metrics_{condition}.json` — accuracy, macro-F1, per-class precision/recall/F1, confusion matrix,
  and both fingerprints.
- `noise_sweep_summary.csv` — tidy condition × (accuracy, macro-F1).

**Per-class columns are optional and named by what they are.** Use `probability_<class>` for
calibrated probabilities, `score_<class>` for uncalibrated decision values, or omit them entirely.
A model without probabilities — e.g. an `SVC` fitted with `probability=False` — must **not** be
refitted just to satisfy a file schema; paired accuracy tests need only `predicted_label`.

## 10. Statistics — cluster, don't just bootstrap windows

Windows are **not independent**. Several come from one recording, and every recording of the same
(instrument, note) belongs to one pitch group — the unit step3 splits on. Resampling individual
windows treats near-duplicates as fresh evidence and yields confidence intervals that are too
narrow.

`noise_stats.py` resamples whole **clusters** with replacement, paired across the two conditions,
and always computes macro-F1 over the fixed 12-class label order:

```bash
python -m instrument_robustness.noise_stats --a clean --b white_20 \
  --dir artifacts/svm/noise --prefix-a svm_test_ --prefix-b svm_test_
python -m instrument_robustness.noise_stats --a clean --b natural_0 --cluster source_path

# SVM versus MERT on the same condition
python -m instrument_robustness.noise_stats --a white_0 --b white_0 \
  --dir-a artifacts/svm/noise --prefix-a svm_test_ \
  --dir-b artifacts/mert/noise --prefix-b mert_test_
```

`pitch_group` is the default and the conservative choice; `source_path` is the looser one. Window-
not called cluster-aware: the primary paired test is explicitly an exact cluster sign test.
Ordinary window-level McNemar is available only with `--window-mcnemar` and is labeled a
correlation-ignoring sensitivity analysis. Separate `--dir-a/--prefix-a` and
`--dir-b/--prefix-b` arguments support direct model-to-model comparison.

## 11. How to run it

```bash
# once per dataset
python -m instrument_robustness.noise_sweep --validate
python -m instrument_robustness.noise_sweep --generate
python -m instrument_robustness.noise_sweep --check-generated

# per model
python -m instrument_robustness.noise_eval_svm
python -m instrument_robustness.noise_eval_mert --device cuda
python -m instrument_robustness.noise_eval_panns
```

ESC-50 is expected at `~/Downloads/noise_sources/ESC-50-master/`, or set `RISE_NOISE_ROOT`. Both
the audio **and** `meta/esc50.csv` are required — the metadata is what splits natural from
mechanical:

```bash
mkdir -p ~/Downloads/noise_sources/ESC-50-master/meta
curl -L -o ~/Downloads/noise_sources/esc50.zip \
  https://github.com/karoldvl/ESC-50/archive/master.zip
unzip -q ~/Downloads/noise_sources/esc50.zip \
  'ESC-50-master/audio/*' -d ~/Downloads/noise_sources
curl -sL -o ~/Downloads/noise_sources/ESC-50-master/meta/esc50.csv \
  https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv
```

## 12. Adding your model to the sweep

Add a thin model adapter that loads the frozen clean checkpoint and supplies a
`predict_scores(paths)` callback to `noise_eval_common.run_noise_evaluation`. The shared runner
owns the condition list, manifest validation, clean-parity gate, cluster columns, metrics, schema,
and output paths, preventing model-specific copies from drifting.

## 13. Results

⚠️ The PANNs numbers previously recorded here were produced under the **old** grid (2 noise types,
SNR 20/10/0/−5, independent realizations per SNR). They are superseded by this protocol and must be
regenerated before being quoted. Re-running costs minutes: the models are clean-trained and frozen,
so changing the noise design is **re-testing only, never retraining**.

## 14. Not covered here

- **Noise-augmented training** (training *with* noise, then re-evaluating) — a separate experiment
  that does require retraining.
