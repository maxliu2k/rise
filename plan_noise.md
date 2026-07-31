# How we measure noise robustness — the methodology, in plain terms

Companion to `plan.md` (which is the run order). This one explains *why* the noise experiment is
built the way it is, without needing the code. The authoritative technical spec is
`docs/NOISE_PLAN.md`; this is the readable version, and it tries not to lose anything important.

---

## The question

**When a model trained on clean audio meets noisy audio, how fast does it fall apart?**

Nothing is retrained. Every model is already finished and frozen. We take the test clips, add
noise at controlled levels, and watch the score drop. Train and validation are never touched.

---

## What "SNR" means

Signal-to-noise ratio, in decibels. It compares the **power** of the instrument to the power of
the noise added to it.

| SNR | what it sounds like |
|---|---|
| 60 dB | a barely-there hiss; the instrument is a million times more powerful |
| 30 dB | clearly audible noise, instrument still obvious |
| 20 dB | mild but unmistakable |
| 0 dB | instrument and noise at **equal power** |
| −10 dB | noise is 10× the instrument |

Lower SNR = more noise. Decibels are logarithmic, so each 10 dB step is a 10× change in the power
ratio — the grid is far wider than the numbers make it look.

**The grid: 60, 50, 40, 30, 20, 10, 0, −10 dB.**

Those high values are not padding. The original grid was `[20, 10, 5, 0, −5]`, inherited from
elsewhere, and a pilot found **every level of it at or below chance** for the SVM — macro-F1 0.093
at 20 dB against 0.083 chance for 12 classes. The interesting collapse happens between 60 and
30 dB, so the grid was moved to where the action is. The mixer was verified first, so the collapse
is a property of the models, not of the mixing.

---

## The conditions

**3 noise types × 8 SNRs × 2 replicates + clean = 49 conditions**, each scored on all 1,255 test
windows.

The three noise types:

- **white** — synthetic Gaussian noise, flat across all frequencies. The clean control case.
- **natural** — real recordings from ESC-50, categories 0–19: animals, rain, wind, thunder.
- **mechanical** — ESC-50 categories 30–49: domestic and urban sounds, engines, machinery.

ESC-50's human non-speech block (20–29: coughing, footsteps, laughing) is **deliberately
excluded** — it is neither environmental nor mechanical, and lumping it into either would blur
what the two categories mean.

---

## Five decisions that make the numbers trustworthy

### 1. The noisy audio is generated **once**, centrally

Every model reads the exact same files. If each model made its own noise, two models would be
scored on different audio, and any difference between them would be part model and part luck of
the draw.

Because they share files, predictions are **paired** — for a given window at a given SNR, we know
what every model said about *that specific piece of audio*. Pairing is what lets us ask "is model
A actually better than B here?" rather than just reporting two numbers side by side.

### 2. One noise draw per replicate, then **scaled** across every SNR

For each (window, noise type, replicate) we draw **one** noise clip. That single clip is then
rescaled to hit 60 dB, 50 dB, 40 dB, and so on.

This keeps two things apart that would otherwise be tangled:

- the **SNR axis** — how loud the noise is
- the **realization axis** — which particular noise clip got drawn

If a fresh clip were drawn at every SNR, a model dipping at 30 dB might just mean it drew an
awkward clip there. Scaling one draw means the curve moves only because the noise got louder.

The two replicates are two independent draws, which is how we see whether a result depends on
*which* noise clip was picked.

### 3. Noise is added to the raw waveform, **before** each model's own feature extraction

Noise goes onto the 22,050 Hz window. Then each model does whatever it normally does — the SVM
computes its 88 handcrafted features, the CNN and CRNN their log-mel spectrogram, MERT resamples
to 24 kHz, AST to 16 kHz, PANNs to 32 kHz.

Critically, **the normalisation statistics are never recomputed on noisy audio**. Those stats are
part of the trained model's contract; refitting them would quietly hand the model information it
never had at training time.

### 4. Files are float32, and the mixture is never re-normalised

At −10 dB the mixture peaks well above ±1.0 (measured up to 6.0). Saving as 16-bit would clip it,
and clipping is a *nonlinear* distortion that genuinely corrupts the SNR. float32 has the room.

We also don't rescale after mixing. That sounds paranoid — rescaling doesn't change a *ratio* —
but the models were trained on audio normalised to a specific loudness, so changing the mixture's
overall level would present an amplitude distribution they never saw.

### 5. Every file's SNR is **measured back out** after it is written

The mixture is written to disk, read back, and its actual SNR recomputed from the difference
between the noisy and clean waveforms. If it misses the target, generation stops.

This one check catches an entire family of problems for free: clipping, quantisation, a wrong
sample rate, a format change. Anything that corrupts the audio shows up as an SNR that doesn't
match what was asked for.

Alongside it, every single file records its full provenance — the clean file's hash, which ESC-50
clip was used and its hash, the random seed, the crop offset, the scaling factor, the achieved SNR
and the output hash.

---

## What the SNR number does *not* tell you

`snr_db` is an average over the **whole window** and the **whole spectrum**. It is exact and
reproducible, and it is easy to over-read. Three specific ways:

**Where in frequency.** Low-frequency rumble can dominate total power while barely touching the
range the instrument occupies. So each mixture also records SNR restricted to the **instrument
band (25–8,000 Hz)** and the worst individual octave. At the same nominal 0 dB, synthetic rumble
measures at least 15 dB cleaner *inside the instrument band* than white noise does.

**When in time.** A brief loud transient can satisfy an average-power target while leaving most of
the clip nearly untouched. So we also record time-resolved SNR — the worst frame, the 5th/50th/95th
percentiles, and what fraction of frames the noise is actually active in.

**Whether the note was even sounding.** Many clips are short notes surrounded by quiet. Noise
spread evenly across the window is harsher during the silence than during the note. So SNR is also
computed over just the frames where the instrument is active.

**What the model actually receives.** AST resamples to 16 kHz, MERT to 24 kHz, PANNs to 32 kHz —
each throws away noise energy above its own Nyquist. The effective SNR after each model's
resampling is recorded per file.

The headline condition remains whole-window SNR. These columns exist so a surprising result can be
checked rather than rationalised.

---

## Reading the curve

Each model produces macro-F1 at each SNR, which becomes a degradation curve. Three summaries:

- **`snr_at_50pct` / `snr_at_90pct`** — the SNR at which the model still holds 50% (or 90%) of its
  clean score. Usually the most legible: *"this model keeps half its clean macro-F1 down to 42 dB."*
- **`robustness_auc`** — the area under the retention curve, integrated over dB. Its virtue is
  being **insensitive to how densely the curve was sampled**. Adding two extra SNR levels where a
  model happens to do well moves the naive mean by 0.093 and the AUC by 0.0004.
- **`mean_retention`** — the unweighted average, reported only so the gap between it and the AUC
  is visible.

Replicates are averaged at each SNR with their spread reported, and model comparisons pair the
same replicate number before differencing.

---

## Deciding whether a difference is real

**Paired cluster bootstrap.** Resampling individual windows would overstate confidence, because
windows from the same pitch group are near-duplicates — the same note at different dynamics. So
resampling happens at the level of **pitch groups**, not windows. All twelve labels are held fixed
in every replicate so macro-F1 stays defined.

**Cluster sign test** and **exact McNemar** are available as complements. The sign test is named
honestly as a sign test — it is not McNemar and doesn't pretend to be.

**Multiple comparisons are corrected.** A full sweep is 48 noisy condition-replicates per model.
Comparing six models, optionally across 12 instruments, is hundreds of hypothesis tests — at
α = 0.05 roughly one in twenty looks significant purely by construction. `benjamini_hochberg`
controls the false-discovery rate, and it requires the family of tests to be **named explicitly**
rather than inferred, so nobody can quietly choose the family after seeing which results they liked.

---

## Two gates before any noisy number is recorded

**Clean parity.** Every model must first reproduce its own official clean macro-F1, to within
0.002, on the exact same test-example count. A missing official result is a hard failure, not a
skipped check. If a model can't reproduce its own clean score, its noisy scores mean nothing.

The 0.002 tolerance is measured, not chosen: the observed macOS-to-SCC difference for the SVM was
0.001020, so the gate is roughly twice the drift that platform differences genuinely cause.

**Manifest completeness.** The generated set is validated against a manifest recording file counts,
provenance rows, hashes and the grid itself. A partial or stale generation is refused rather than
silently evaluated.

---

## Honest limitations

**Two replicates is thin.** The whole point of replicates is to separate "this model is fragile"
from "this window drew an unlucky clip". Two draws give a difference, not a distribution. This was
reduced for a deadline and is a known compromise.

**The metric is macro-F1**, project-wide. Macro-F1 rewards a collapsed classifier more as class
imbalance grows, which is exactly the failure mode heavy noise produces — a model predicting one
class at −10 dB scores better than it deserves. MCC is recorded alongside everywhere for this
reason; treat a macro-F1 that stays suspiciously high at low SNR as a prompt to check MCC.

**The clean-parity gate proves the model loads correctly. It does not validate the noisy audio.**
That is covered separately by the measure-it-back check at generation. Two different guarantees;
don't conflate them.

**Currently blocked on disk.** The 48 noisy conditions are ~14.8 GiB. `/projectnb/rise-grid` has
about 1.15 GB free; `/project/rise-grid` has about 34 GB. Either repoint the output, or generate
one condition at a time — evaluate all six models on it, record predictions, delete the audio, move
on. Streaming caps peak disk at roughly 330 MB and preserves pairing within each condition, at the
cost of orchestrating the six models together.

**ESC-50 is not recording-studio noise.** "Natural" is animals and weather; "mechanical" is
domestic and urban. Neither is the room tone, HVAC hum, or handling noise that a real instrument
recording actually contends with. The results describe robustness to environmental sound, which is
a reasonable proxy and not the same thing.
