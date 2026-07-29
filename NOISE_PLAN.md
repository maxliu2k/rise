# Noise robustness sweep — design and protocol

How we measure the degradation of a **clean-trained** model as noise increases. Inference only:
no model is retrained, and the training and validation splits are never touched.

Implemented in `noise_sweep.py` (generates the audio) and `noise_eval_panns.py` (runs one model
over it). Read this before adding a model to the sweep.

---

## 1. The one rule that makes cross-model comparison possible

**The noisy audio is generated ONCE, centrally, and every model reads the same files.**

If each model generated its own noise, two models would be scored on different audio, and their
predictions would no longer be *paired*. Paired predictions are what McNemar's test and paired
bootstrap CIs require — without them we can report each model's accuracy but cannot say whether
the difference between two models is significant. That is the whole point of the group comparison,
so this rule is not negotiable.

Practically: run `noise_sweep.py --generate` once per dataset, then point every model at
`work/windows_noisy/`.

## 2. Conditions — 9 total

| | 20 dB | 10 dB | 0 dB | −5 dB |
|---|---|---|---|---|
| **white** (Gaussian) | ✓ | ✓ | ✓ | ✓ |
| **real** (ESC-50) | ✓ | ✓ | ✓ | ✓ |

plus **clean** — shared between both noise types rather than duplicated, so 4 × 2 + 1 = **9**.

Lower SNR = more noise. 20 dB is mild (instrument clearly audible), −5 dB means the noise is
*louder* than the instrument.

- **white** — Gaussian noise, flat across all frequencies. The harsher case: it fills every mel
  bin and erases the spectral envelope the model relies on.
- **real** — random 3 s excerpts from **ESC-50** (2000 environmental clips: rain, traffic, machinery…),
  resampled to 22050 Hz. More realistic, and spectrally sparse, so it leaves gaps the model can
  still see through. We chose ESC-50 over MUSAN because it is ~600 MB instead of ~11 GB and is
  sufficient here.

## 3. SNR is defined by power, not amplitude

```python
p_sig   = mean(clean ** 2)
p_noise = mean(noise ** 2)
alpha   = sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
noisy   = clean + alpha * noise
```

Getting this wrong (using amplitude, or the wrong sign) produces files that *look* plausible but
sit at the wrong SNR, which silently invalidates the whole sweep. See §6 for how we check it.

## 4. Two invariants that are easy to break

**Do NOT re-normalize after adding noise.** Re-normalizing rescales the mixture and changes the
effective SNR away from the target. The windows were already RMS-normalized before mixing (Step 5),
so the signal power is known and fixed going in.

**Files are written as float32 WAV, not 16-bit PCM.** At −5 dB the mixture peaks well above ±1.0
(we measured up to 9.2). 16-bit PCM clips at ±1.0, which would hard-distort exactly the
low-SNR conditions the experiment is about. float32 has the headroom.

## 5. Deterministic noise — why you don't need to download 4 GB

The RNG for each window is seeded from the window itself:

```python
seed = sha256(f"{window_id}|{noise_type}|{snr}").digest()[:4]
```

So the same window always receives the same noise realization — on any machine, on any rerun.
**Verified: regenerating produces bit-identical files.** Two consequences:

1. Reruns are reproducible, and a partially generated set can be safely resumed.
2. **Teammates do not need the generated audio.** Running `noise_sweep.py --generate` locally
   reproduces it exactly, so only the code is shared. (The audio is 1.4 GB for TinySOL and 2.6 GB
   for Philharmonia, and is git-ignored.)

⚠️ **This only holds if the windows are identical.** The seed is derived from `window_id`, so a
different manifest or a different split produces different windows → different noise → predictions
that are no longer paired. Everyone must build from the same `manifest.csv` (check
`manifest_fingerprint.json`) and the same `splits.csv`.

## 6. Validation before generating anything

`noise_sweep.py --validate` samples a few windows, mixes them at every SNR, then **measures the
SNR back out of the mixture** and compares it to the target. Current result: **0.000000 dB error**
across all 8 noisy conditions (spec was < 0.1 dB). It also writes listenable 0 dB samples to
`work/windows_noisy/_validation_samples/` — a 0 dB clip should sound like an instrument buried in
noise, not like silence or garbage.

Run this first. A power-vs-amplitude error produces plausible-looking files that are quietly wrong.

## 7. Featurization must match training

Each model featurizes the noisy window through **the exact code path it was trained with** —
noise is added to the 22050 Hz window *first*, then the model's own extractor runs. For PANNs that
means `panns_input` (resample to 32 kHz; CNN14 computes its own log-mel internally).

Never recompute dataset normalization statistics on noisy audio. The Step-6 stats are part of the
trained model's contract; recomputing them would let the model adapt to the noise, which is
precisely what we are trying to measure it *failing* to do.

## 8. Outputs

Per dataset, under `features/<model>/noise/`:

- `*_test_{condition}.csv` — one row per test window: `window_id`, `true_label`,
  `predicted_label`, `correct`, and a probability column per class. **This is the file the group's
  McNemar / bootstrap analysis consumes.**
- `metrics_{condition}.json` — accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
- `noise_sweep_summary.csv` — tidy condition × (accuracy, macro-F1).

## 9. How to run it

```bash
# once per dataset
python -m instrument_robustness.noise_sweep --validate    # check the SNR math, listen to a sample
python -m instrument_robustness.noise_sweep --generate    # write the shared noisy windows

# per model
python -m instrument_robustness.noise_eval_panns
```

ESC-50 is expected at `~/Downloads/noise_sources/ESC-50-master/audio/`, or set `RISE_NOISE_ROOT`:

```bash
curl -L -o esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
unzip -q esc50.zip 'ESC-50-master/audio/*' -d ~/Downloads/noise_sources
```

## 10. Adding your model to the sweep

Copy `noise_eval_panns.py`. Change only two things: how the checkpoint is loaded, and how a
window is turned into model input. Keep everything else — the condition list, the shared file
paths, the CSV schema — so the outputs stay comparable and paired.

## 11. Results so far (PANNs CNN14, clean-trained, test macro-F1)

| condition | TinySOL | Philharmonia |
|---|--:|--:|
| clean | 0.9747 | 0.9866 |
| real 20 dB | 0.7196 | 0.7684 |
| real 10 dB | 0.5262 | 0.5735 |
| real 0 dB | 0.3612 | 0.4137 |
| real −5 dB | 0.2620 | 0.3301 |
| white 20 dB | 0.4218 | 0.2762 |
| white 10 dB | 0.1960 | 0.1587 |
| white 0 dB | 0.0642 | 0.1182 |
| white −5 dB | 0.0497 | 0.0626 |

Chance for 12 classes is 0.083. Three observations: a model at ~0.98 clean falls to **0.28–0.42 at
20 dB white noise**, which is a barely-audible amount of noise; white noise is far more destructive
than real noise at every level; and Philharmonia degrades *worse* under white noise despite scoring
*higher* clean — higher clean accuracy did not buy robustness.

## 12. Not covered here

- **Noise-augmented training** (training *with* noise, then re-evaluating) — a separate experiment
  requiring retraining. The sweep above is clean-trained only.
- **Bootstrap CIs and McNemar tests** — computed later from the saved per-clip CSVs, no rerun needed.
