# Findings — 12-class instrument ID, medium CNN

Run date: 2026-07-24 – 07-26 · seeds 42/43/44 · CPU (torch 2.4.1+cpu) · Philharmonia
Config fingerprint: `SR=22050`, `CLIP_SECONDS=3.0`, `MAX_CHUNKS_PER_FILE=1`, 12 classes

Reproduce (from repo root, `pip install -e .` or `PYTHONPATH=src`):

```bash
python -m instrument_robustness.prep_data --inventory-only        # inventory + codec gate, no audio
python -m instrument_robustness.prep_data                         # download, cache, split  (~5 min)
python -m instrument_robustness.single.train                      # 3 seeds  (~88 min, CPU)
python -m instrument_robustness.single.noise_eval                 # SNR sweep, 3 colours
```

Every artifact (`manifest.json`, each `model_s*.pt`, `metrics.json`, `snr_results.json`) carries the
config fingerprint, and every consumer asserts it. A stale checkpoint evaluated against a rebuilt
cache **crashes** rather than producing a plausible, meaningless curve.

## Verdict

| Question | Verdict |
|---|---|
| Is the dataset workable? | **GO** — clean, ample, zero loading friction |
| Is the pipeline sound end to end? | **GO** — data → spectrogram → training → eval verified |
| Does 12 classes give headroom to compare models? | **GO** — 0.9600, off the ceiling (2-class was 1.0000) |
| Is the timeline achievable? | **GO on compute** — ~29 min/seed on CPU |
| Is the planned noise sweep usable as specced? | **PARTLY** — 20 dB usable, 10/0 dB collapsed. See §5a. |
| Is the dataset safe at higher sample rates? | **NO** — inert at 22.05 kHz, poisonous at 44.1 kHz. See §4. |

## 1. Headline

| metric | mean | std | min | max |
|---|---|---|---|---|
| **balanced accuracy** | **0.9600** | 0.0138 | 0.9502 | 0.9757 |
| **MCC** | **0.9618** | 0.0097 | 0.9543 | 0.9728 |
| train-val bacc gap | 0.0324 | 0.0046 | 0.0275 | 0.0366 |
| best epoch | 25.7 | 4.6 | 22 | 31 |

Chance is 1/12 = 0.0833. 8,375 clips (5,788 train / 1,303 val / 1,284 test), train imbalance
1.95:1 (class weights 0.818–1.592 applied), 110,956 params, 3 seeds.

Timing: 52.2 s/epoch, 29.2 min/seed, 87.6 min for all three (CPU only; `get_device()` auto-detects,
a GPU only improves it).

**The ceiling is broken, which is what the study needed.** The 2-class pilot scored 1.0000 and left
no headroom to distinguish six models on clean audio. 0.9600 with real per-class structure does —
though see §8: at 0.96 the *clean* task is again close to saturating, and the discriminating power
for a six-model comparison lives in the noise sweep, not here.

**Multi-seed earned its keep.** Per-seed: 0.9502 / 0.9540 / **0.9757**. The 2.55-point spread is
wider than the margin many model comparisons turn on. Report mean ± std over ≥3 seeds throughout.

Train-val gap 0.0324: the architecture is not overfitting at this scale.

**Determinism was used as a check.** The fingerprinting rebuild retrained all three seeds from a
freshly rebuilt cache and reproduced 0.9502 / 0.9540 / 0.9757 to four decimals, with per-epoch
losses matching bit-for-bit. That is the evidence that the provenance changes are behaviour-neutral.

## 2. The model learned timbre, not pitch — the register confound is absent on this evidence

The concern (carried since planning): instrument ranges only partly overlap, so a CNN could score
well on pitch alone and never learn timbre. **Tuba was added specifically to test this** — at
As0–F4 it overlaps double-bass (C1–G4) almost exactly, a same-register / different-family pair.
Pre-registered signature: if the model reads pitch, *this is the pair that collapses into each other*.

**Result: tuba ↔ double-bass cross-confusions are exactly 0, in both directions, summed over 3 seeds.**

| class | recall | errors go to |
|---|---|---|
| tuba | 0.9949 | trombone (2) — same family |
| double-bass | 0.9115 | cello (30) — same family, adjacent register |

Double-bass is only the 11th-easiest class, but **every single one of its errors goes to cello**, not
to its register-mate. Each class's errors stay inside its own family. This is a stronger form of the
argument than the previous framing (which rested on both classes being the two easiest, and would
have looked weakened by double-bass's drop to 0.9115 — it isn't).

Corroborating: the confusions are the ones a musician would make.

| confusion | n (summed over 3 seeds) |
|---|---|
| double-bass → cello | 30 |
| viola → cello | 15 |
| trumpet → clarinet | 13 |
| trumpet → trombone | 11 |
| oboe → cello | 7 |
| oboe → violin | 5 |

Same-family and same-register pairs. Nothing arbitrary.

## 3. Per-class — trumpet is the hard one

| class | recall | ± | precision | support |
|---|---|---|---|---|
| tuba | 0.9949 | 0.0044 | 0.9949 | 131 |
| flute | 0.9943 | 0.0049 | 0.9915 | 117 |
| bassoon | 0.9904 | 0.0167 | 0.9750 | 104 |
| trombone | 0.9854 | 0.0051 | 0.9474 | 114 |
| clarinet | 0.9828 | 0.0000 | 0.9618 | 116 |
| cello | 0.9780 | 0.0048 | 0.8623 | 121 |
| violin | 0.9758 | 0.0161 | 0.9813 | 124 |
| french-horn | 0.9569 | 0.0245 | 0.9921 | 85 |
| viola | 0.9524 | 0.0103 | 0.9758 | 112 |
| oboe | 0.9309 | 0.0550 | 0.9789 | 82 |
| double-bass | 0.9115 | 0.0354 | 0.9749 | 113 |
| **trumpet** | **0.8667** | 0.1047 | 0.9875 | 65 |

**Trumpet — trivially separable at 2 classes — is still the hardest**, and by the widest seed spread
of any class (±0.1047, range 0.785–0.985). It has the fewest clips and confuses with clarinet and
trombone. Note the asymmetry: precision 0.9875 vs recall 0.8667 — when it says trumpet it is right,
but it misses an eighth of them.

**cello has the lowest precision (0.8623)** — it absorbs errors from double-bass, viola, and oboe.
It is the sink for the bowed-string family.

## 4. Latent hazard: bitrate is confounded with class

**Philharmonia encodes at three bitrates that cut across instrument families:**

```
64 kbps: bassoon, clarinet, double-bass, trumpet
80 kbps: trombone, tuba
96 kbps: cello, flute, french-horn, oboe, viola, violin
```

MP3 lowpasses as a function of bitrate, so the encoder partitions the classes into 3 groups for
free — nothing to do with the instruments. `all-samples/inventory.csv` already records these
(2051 @ 64k, 4242 @ 96k, 1803 @ 80k) but nothing downstream acts on them.

**Currently inert, and only because of the sample rate.** Measured across all three groups: every
codec brick wall sits above 19 kHz, and the class-correlated spectral difference above ~14 kHz
(cello-vs-trumpet: +23.5 dB at 15 kHz, +30.2 dB at 18 kHz). At SR=22050 the Nyquist is 11,025 Hz
and the resampler discards all of it. Verified three ways:

- **No aliasing** — top-bin gaps in the cached spectrograms are mixed-sign and tiny (mean +0.172).
- **In-band the classes differ in the physically correct direction** — trumpet brighter than cello,
  as brass should be against bowed strings. A bits-driven artifact would point the other way.
- **The bottom 64 mel bins (0–2.6 kHz)**, nowhere near any codec effect, scored 0.9146 balanced
  accuracy alone on the 2-class task. The signal is real timbre.

**The risk is the plan.** The spec says "per-model rates come later." **At 44.1 kHz this becomes a
free 3-way shortcut** and a model would post an excellent score having learned nothing.
`prep_data.check_bitrates()` prints MITIGATED at the current SR and flips to `*** NOT MITIGATED ***`
above ~28 kHz. Do not silence it.

Caveat: "above ~14 kHz" comes from an 80-file-per-class sample using a −60 dB-from-peak criterion.
The margin plus the three checks make it robust, but it is not a hard bound on every file.
Re-measure before raising SR.

## 5. Noise robustness (12 classes, 3 seeds, additive noise, clean-trained model)

Run with `python -m instrument_robustness.single.noise_eval`. Achieved nominal SNR is on target to
<0.01 dB, and the clean path matches `train.py` to 1e-9 on all 3 seeds (`clean_path_check_passed`).

### 5a. The specced 20/10/0 dB sweep is partly usable — the collapse is later than previously stated

White noise, balanced accuracy vs nominal SNR (chance = 0.0833, clean = 0.9600). The last two
columns characterise *how* the model is failing: the share of all predictions going to its single
most-predicted class, and how many of the 12 classes still receive ≥1% of predictions.

| SNR | balanced acc | MCC | vs clean | top-class share | classes ≥1% |
|---|---|---|---|---|---|
| clean | 0.9600 | 0.9618 | — | — | 12 |
| 60 dB | 0.9609 | 0.9627 | +0.001 | 10.7% | 12 |
| 55 dB | 0.9591 | 0.9610 | −0.001 | 10.7% | 12 |
| 50 dB | 0.9432 | 0.9435 | −0.017 | 10.9% | 12 |
| 47.5 dB | 0.9259 | 0.9247 | −0.034 | 11.3% | 12 |
| **45 dB** | 0.9017 | 0.8979 | −0.058 | 11.8% | 12 |
| 42.5 dB | 0.8711 | 0.8662 | −0.089 | 12.3% | 12 |
| **40 dB** | 0.8275 | 0.8217 | −0.132 | 13.3% | 12 |
| 37.5 dB | 0.7619 | 0.7577 | −0.198 | 15.0% | 12 |
| 35 dB | 0.7011 | 0.6976 | −0.259 | 17.1% | 12 |
| 32.5 dB | 0.6363 | 0.6332 | −0.324 | 19.3% | **12** |
| 30 dB | 0.5769 | 0.5738 | −0.383 | 21.1% | 11 |
| 27.5 dB | 0.5227 | 0.5191 | −0.437 | 23.6% | 11 |
| 25 dB | 0.4787 | 0.4757 | −0.481 | 26.0% | 10 |
| **20 dB** | 0.3826 | 0.3732 | −0.577 | 29.9% | 7 |
| 10 dB | 0.1896 | 0.1458 | −0.770 | 46.4% | 6 |
| 0 dB | 0.1052 | 0.0343 | −0.855 | 75.6% | 4 |

The knee is at **45–40 dB** — noise at ~1% of signal amplitude, quieter than a recording studio.
Degradation starts absurdly early. But the levels split into three regimes:

- **60–32.5 dB: genuine graded degradation.** Predictions stay spread over all 12 classes (top
  share 10.7–19.3% against a uniform 8.3%). This is the band with real resolving power for a model
  comparison. The 2.5 dB grid pins the boundary at **32.5 dB**; the earlier coarse grid could only
  say "60–30".
- **30–25 dB: shedding.** The model drops to 11 classes at 30 dB and 10 at 25 dB. Still well above
  chance and still informative, but a comparison here is partly measuring how fast each model
  loses its label space.
- **20 dB: substantially collapsed.** bacc 0.3826 is clear of chance, but **5 of 12 classes have
  dropped out of the model's effective output vocabulary.**
- **10 dB and 0 dB: collapsed.** By 0 dB, 75.6% of all predictions are double-bass and ~94% are
  double-bass or cello — the model has folded onto the low-register strings. Recall figures for
  those classes at low SNR (double-bass 0.976 at 0 dB) are **attractor artifacts, not robustness.**
  Six models run here would be near-indistinguishable.

*Grid note.* This sweep uses 16 levels (2.5 dB steps across 50–25 dB). `noise_eval` seeds its
per-clip RNG on the condition's index in `SNR_LEVELS_DB`, so re-gridding changes the noise
realisation at every level whose position moved. Measured effect: at the nine levels shared with
the previous 9-point grid, the largest change is **0.0027** (a fifth of the seed spread), and
60 dB — the one level whose index did not move — is bit-identical. Statistically comparable, not
bit-identical.

**Correction to the previous version of this section.** It stated that degradation is "graceful (a
steady slide, not the majority-class collapse the 2-class model showed)." That is true only in the
60–32.5 dB band. Majority-class collapse *does* occur here — it begins at 30 dB and is severe by
10 dB. The earlier claim was measured before the class-space diagnostic existed and is retracted.

This is a clean-trained model meeting noise it never saw, so it measures brittleness to distribution
shift, not achievable robustness. **The noise-aware training experiment has not been run at 12
classes.** The 2-class probe turned the cliff into a gentle slope (0.99 → 0.89 at 0 dB), but that
does not transfer without measurement. That experiment is the one that turns this into a result
rather than a characterisation.

### 5b. In-band energy explains most of the colour gap, but not all of it

Swept white / pink (1/f) / brown (1/f²) at matched *nominal* SNR. On that axis the colours look
wildly different — at nominal 0 dB, brown 0.3823 vs white 0.1052. Most of that is an artifact of
the SNR definition: nominal SNR fixes total power and ignores *where* the power sits. In-band is
200 Hz – 8 kHz, where the notes live.

| noise | in-band SNR at nominal 0 dB |
|---|---|
| white | −0.41 dB (honest) |
| pink | +2.81 dB |
| brown | **+27.34 dB** |

Brown dumps almost all its energy below the band, so a "nominal 0 dB" brown clip is really +27 dB
where it counts. It didn't survive the noise — it was never given it. **Always report in-band SNR
alongside nominal, or the x-axis lies by up to 27 dB.**

**But the colours do not fully collapse onto a single curve.** Interpolating each colour onto a
common grid over their shared in-band range (27.3–59.6 dB):

| in-band SNR | white | pink | brown | spread |
|---|---|---|---|---|
| 27.3 dB | 0.5281 | 0.4165 | 0.3823 | **0.1458** |
| 32.3 dB | 0.6417 | 0.5978 | 0.5442 | 0.0976 |
| 37.3 dB | 0.7664 | 0.7841 | 0.7061 | 0.0781 |
| 42.2 dB | 0.8727 | 0.8797 | 0.8030 | 0.0767 |
| 47.2 dB | 0.9266 | 0.9331 | 0.8976 | 0.0355 |
| 52.1 dB | 0.9513 | 0.9545 | 0.9323 | 0.0221 |
| 57.1 dB | 0.9600 | 0.9588 | 0.9434 | 0.0166 |

Max spread **0.1458 — over 10× the seed std (0.0138)**. The collapse holds only above ~49 dB
in-band, where every colour is already near clean. Below that a consistent ordering persists: at
matched in-band SNR, **white is least harmful and brown most.**

**Correction.** The previous version of this section claimed the colours "collapse to within
~0.03–0.05 (the seed noise floor)" and concluded "colour is irrelevant to robustness... the colour
axis can be dropped." The first claim is not supported at these numbers and the recommendation is
retracted — it is a recommendation someone would have acted on. Sweeping white alone remains the
right default (it is the honest colour, and the harshest at matched in-band SNR *at low SNR*), but
dropping the colour axis would discard a real effect.

**The 16-point grid adds a finding the 9-point grid could not resolve: white and pink CROSS OVER
at ~28 dB nominal.** Pink is gentler above it (35 dB: 0.7995 pink vs 0.7011 white) and harsher
below (20 dB: 0.2860 vs 0.3826; 10 dB: 0.1191 vs 0.1896). The old grid stepped 30 → 20 and showed
only an ambiguous gap. So the colours do not merely differ in magnitude — **their ordering inverts
with SNR**, which is a second, independent reason the colour axis cannot be collapsed to one curve.

Recomputed on this denser grid, the retraction below stands unchanged: max spread 0.1458 against
0.1472 on the 9-point grid.

*Untested hypothesis for the residual:* brown's sub-200 Hz energy falls outside the in-band window
but still lands in the low mel bins, and per-clip z-scoring lets it shift the whole normalisation.
Not measured — stated only to record what to test, not as a mechanism.

### 5c. Open decision

(a) noise-aware / multi-condition training, keep 20/10/0 — proven at 2 classes to give a usable
curve, unmeasured at 12; (b) keep clean training on the re-centred grid; (c) both, as matched vs.
mismatched conditions.

**Option (b) is done.** `SNR_LEVELS_DB` was re-gridded on 2026-07-28 to 16 levels — 2.5 dB steps
across 50–25 dB, where balanced accuracy falls 0.90 → 0.48, plus 60/55 anchoring the clean end and
20/10/0 retained to document the collapse. The old 9-point grid spent 3 points where every model
scores near chance and only 4 in the band that resolves anything. This does not invalidate any
cache or checkpoint: `SNR_LEVELS_DB` is deliberately absent from `config_fingerprint()`, since it
describes an evaluation sweep rather than the meaning of the cached arrays.

**(a) remains unrun**, and it is still the experiment that turns this from characterisation into a
result. Choice between (a) and (c) is still yours.

## 6. Dataset notes

- **Source**: the official `philharmonia.co.uk/assets/audio/samples/...` URLs predate their site
  redesign and no longer resolve. The Internet Archive mirror works. CC-BY-SA 4.0.
- **Filename traps**: the zip/dir name is NOT the instrument field — zips use spaces where
  filenames use hyphens, and `cor anglais.zip` contains `english-horn_*.mp3`.
- **`duration` is not numeric**: `025`, `05`, `1`, `15`, `long`, `very-long`, `phrase`. Parsing it
  as a number breaks on 210 files. `phrase` files are continuous crescendos, **not** sequences of
  separate notes — measured internal silence is 0.00–0.05 s.
- **Two 0-byte MP3s ship in the archive** (`viola_D6_05_piano_arco-normal`,
  `saxophone_Fs3_15_fortissimo_normal`). soundfile rejects them, librosa falls back to audioread,
  and audioread dies with `EOFError`, killing the whole run. Handled and counted, not silently
  dropped. `all-samples/manifest.py` independently found and drops the same file.
- **Strict single-articulation filtering costs almost nothing.** `normal`/`arco-normal` dominates
  (84–89% of files) rather than being a slice. Same insight as `manifest.py`'s
  `PLAIN = {normal, arco-normal}`.
- **Clips are a fixed 3.0 s; short notes are TILED, never zero-padded.** A note shorter than the
  window is looped to fill it, so every sample is real signal. **97.3% of clips are tiled**
  (`manifest.signal_stats.tiled_fraction`) — only 2.7% of notes reach 3.0 s unaided. Median source
  length is 0.906 s (range 0.18–77.6 s); tuba (median 0.575 s) repeats ~5×. Tiling is the dominant
  path, not an edge case. This replaces the previous variable-length design; the note that "nothing
  is padded or tiled" is no longer true.
  - **Why not zero-padding.** `power_to_db(ref=np.max)` clamps digital silence to the −80 dB floor,
    injected noise fills it, and the clip lands outside the training distribution. Measured:
    majority-class collapse at *every* SNR. Zero-padding does not merely add a nuisance — it
    destroys the noise sweep. Do not reintroduce it.
  - **The obvious objection was tested and refuted.** A looped note's repeated attacks encode the
    source note length, which correlates with class (§8) — so the model could read tiling period
    instead of timbre. Pre-registered signature: if so, the classes at the *extremes* of the length
    distribution should be the most noise-robust, since a periodic onset train survives noise better
    than fine spectral detail. Measured (white noise, per-class recall from `snr_results.json`):

    | class | median source s | recall @10 dB | recall @0 dB |
    |---|---|---|---|
    | tuba (shortest, ~5× tiled) | 0.575 | 0.000 | 0.000 |
    | trombone | 0.673 | 0.018 | 0.000 |
    | french-horn | 1.279 | 0.000 | 0.000 |
    | clarinet (longest, least tiled) | 1.305 | 0.170 | 0.000 |

    Both extremes floor at 0.000. There is no monotone relationship between source length and
    noise survival; what survives instead is whatever the collapse attractor happens to be (§5a).
    Corroborating: 0 dB balanced accuracy under tiling (0.1071) is *lower* than the old
    variable-length design (0.1207), the opposite of what an added noise-robust shortcut predicts.
  - **Untested caveat**: this validates tiling *for this CNN*. An SVM with handcrafted onset-rate
    features could still pick up the loop period. Re-check before this becomes the shared dataset.
- **Leak-free split, asserted every run**: grouped by pitch — 544 groups and all 8,375 source
  files, none spanning splits.
- **Measured cost of getting the split wrong: +0.0357 balanced accuracy.**
  `single/split_policy_probe.py`, 3 seeds, identical data and training, only the split policy
  varying:

  | split policy | pitch-groups leaking | balanced accuracy |
  |---|---|---|
  | grouped (correct) | 0.000 | **0.9600 ± 0.0138** |
  | random by file | 0.967 | **0.9957 ± 0.0006** |

  The inflation is **2.6× the seed spread**, so it is a real effect and not seed noise. It is also
  close to the +0.0372 that a single-seed probe suggested on 2026-07-22 — that estimate was right
  in magnitude, but it was one run on the old variable-length config with no fingerprint, and
  should not have been quoted as a number until now. It is superseded by this.

  **The seed spread is the more useful tell.** The leaked arm's std is 0.0006 against 0.0138 — a
  23-fold collapse. Per-seed it scores 0.9959 / 0.9950 / 0.9962. When a split leaks, the task
  becomes so easy that initialisation stops mattering; near-perfect accuracy with almost no seed
  variance is the signature of memorising near-duplicates, not of a good model. That diagnostic
  costs nothing to apply and needs no control experiment: **be suspicious of an implausibly tight
  multi-seed spread.**

  Applies directly to `main`, whose file-level split leaked 93.1% of pitch-groups (406 of 436,
  361 spanning all three splits) until 2026-07-28. Every number that pipeline produced before then
  carries roughly this inflation.

## 7. Metrics: why accuracy and F1 are not reported

Both pay a collapsed classifier the class prior, and both have floors that **drift with the split**:

| majority prior | accuracy | macro F1 | balanced acc | MCC |
|---|---|---|---|---|
| 0.50 | 0.5000 | 0.3333 | 0.5000 | 0.0000 |
| 0.6244 | 0.6240 | 0.3842 | 0.5000 | 0.0000 |
| 0.90 | 0.9000 | **0.4737** | 0.5000 | 0.0000 |

Every row is the same dead model predicting one class. **Macro F1 rises with imbalance** — a
collapsed model scores *better* on more imbalanced data. F1 also discards true negatives by
construction: sound for retrieval, wrong here, where a true negative is a correctly identified
cello. **Balanced accuracy (chance = 1/n_classes) and MCC (0.0 = no information) have fixed
floors.** This departs from the original spec, which asked for precision/recall/F1.

§5a is the concrete payoff: MCC falls to 0.0373 at 0 dB, correctly reporting "no information,"
while the 0.1071 balanced accuracy and a 0.976 per-class recall could both be misread as residual
skill.

## 8. Caveats

- **Studio single notes are an upper bound**, not a forecast for real polyphonic audio. The
  `multi/` mixtures are equal-RMS sums of isolated notes — they exercise the multi-label machinery
  but are not real polyphony (no shared room, no mic bleed, no ensemble interaction). No claim
  about transfer to real audio is supported by anything in this document. This is a scope
  boundary, not a defect: real-audio evaluation is deferred by decision (§9).
- **Clean accuracy is close to saturating again.** At 0.9600 with a 0.0138 seed spread, separating
  six models on clean audio has limited headroom. The discriminating measurement is the noise
  sweep, and per §5a it must be run in the 60–32.5 dB band to discriminate anything (the grid was
  re-centred there on 2026-07-28).
- **Clip length is a real shortcut at 12 classes.** A length-only classifier (depth-4 decision tree
  on `source_seconds`) scores **0.1977 balanced accuracy vs 0.0833 chance** — lift +0.1144,
  optimistic since it is fit and scored on train. Not negligible. Per-class medians: tuba 0.575 s,
  trombone 0.673 s vs clarinet 1.305 s, french-horn 1.279 s. Tiling does not appear to let the CNN
  exploit this (§6), but the confound is in the data regardless.
- **Learning rate.** 1e-4 eliminated the val-loss spikes at the cost of slower convergence on the
  earlier variable-length config. **Not applied — `config.py` still has 1e-3**, and the effect has
  not been re-measured at 3.0 s. Gradient clipping does *not* help: Adam's update is scale-invariant
  in the gradient, so clipping shrinks `m` and `√v` together and leaves the ratio unchanged.
- Timing (~29 min/seed) is **CPU-only** (torch 2.4.1+cpu). Pretrained backbones (AST, PANNs) would
  want a GPU.

## 9. Divergence from `main` worth resolving

- **This branch uses 12 classes; `main` standardised on 9.** `all-samples/inventory.py`'s `FAMILY`
  dict lists only 9 instruments, so `manifest.csv` silently drops **double-bass, french-horn,
  oboe**. Extending that dict would align them. This divergence is deliberate and unresolved —
  check before assuming either.
- **`manifest.csv` is arguably the better data index** — it already carries family, midi,
  duration_s, is_plain, is_phrase. `prep_data.py` currently re-derives all of it from filenames.
  Consuming it instead would avoid two sources of truth. Blocked on layout: its paths point at an
  `all-samples/` tree (`bassoon/As1/*.mp3`) that isn't committed, while `prep_data.py` downloads
  per-instrument zips to `data/raw/bassoon/*.mp3`.
- **`configs/data/irmas.yaml` is present and 0 bytes — real-audio evaluation is deliberately
  deferred.** The study is six models (three from-scratch, three pretrained), and the current scope
  is **Philharmonia only**. That supports the model comparison, the noise sweep, the synthetic
  polyphony axis, and rank-stability between conditions — none of which need real audio. It does
  **not** support any claim about real recordings. State conclusions as "under controlled
  degradation of studio single notes"; an unqualified transfer claim is not available from this
  data, and would need IRMAS or an equivalent populated first.
- **`src/instrument_robustness/init.py` should be `__init__.py`** — `init.py` is not a package
  marker and does nothing. The correct one was added; the original is still in place.
