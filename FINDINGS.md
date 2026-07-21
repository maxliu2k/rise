# Findings — 12-class instrument ID, medium CNN

Run date: 2026-07-16 · seeds 42/43/44 · CPU (torch 2.4.1+cpu) · Philharmonia

Reproduce (from repo root, `pip install -e .` or `PYTHONPATH=src`):

```bash
python -m instrument_robustness.prep_data --inventory-only   # inventory + codec gate, no audio processing
python -m instrument_robustness.prep_data                    # download, cache, split  (~5 min)
python -m instrument_robustness.train                        # 3 seeds                 (~50 min, CPU)
```

## Verdict

| Question | Verdict |
|---|---|
| Is the dataset workable? | **GO** — clean, ample, zero loading friction |
| Is the pipeline sound end to end? | **GO** — data → spectrogram → training → eval verified |
| Does 12 classes give headroom to compare models? | **GO** — 0.9234, no longer at ceiling (2-class was 1.0000) |
| Is the timeline achievable? | **GO on compute** — ~17 min/seed on CPU; compute is not the constraint |
| Is the planned noise sweep usable as specced? | **NO** — the 20/10/0 dB band is a dead zone. See §5. |
| Is the dataset safe at higher sample rates? | **NO** — inert at 22.05 kHz, poisonous at 44.1 kHz. See §4. |

## 1. Headline

| metric | mean | std | min | max |
|---|---|---|---|---|
| **balanced accuracy** | **0.9234** | 0.0105 | 0.9113 | 0.9298 |
| **MCC** | **0.9213** | 0.0097 | 0.9101 | 0.9270 |
| train-val bacc gap | 0.0227 | 0.0070 | 0.0149 | 0.0283 |
| best epoch | 31.3 | 5.0 | 26 | 36 |

Chance is 1/12 = 0.0833. 8,847 clips (6,133 train / 1,356 val / 1,358 test), imbalance 1.74:1,
110,956 params, 3 seeds.

**The ceiling is broken, which is what the study needed.** The 2-class pilot scored 1.0000 and
left no headroom to distinguish six models on clean audio. 0.9234 with real per-class structure
does.

**Multi-seed earned its keep immediately.** Per-seed: 0.9298 / 0.9291 / **0.9113**. That 1.85-point
spread is exactly the margin a six-model comparison would turn on — a single-seed run would have
reported one of those as *the* number. Report mean ± std over ≥3 seeds throughout.

Train-val gap 0.0227: the architecture is not overfitting at this scale.

## 2. The model learned timbre, not pitch — the register confound is weaker than feared

The concern (carried since planning): instrument ranges only partly overlap, so a CNN could score
well on pitch alone and never learn timbre. **Tuba was added specifically to test this** — at
As0–F4 it overlaps double-bass (C1–G4) almost exactly, a same-register / different-family pair.

**Result: tuba (0.9976) and double-bass (0.9913) are the two EASIEST classes.** Same register,
near-perfectly separated. If the model were reading pitch, this is the pair that would collapse.
It doesn't.

Corroborating: the confusions are the ones a musician would make.

| confusion | n (summed over 3 seeds) |
|---|---|
| trombone → bassoon | 29 |
| trombone → french-horn | 21 |
| cello → double-bass | 20 |
| trumpet → trombone | 19 |
| trumpet → clarinet | 19 |
| viola → cello | 17 |

Same-family and same-register pairs. Nothing arbitrary.

## 3. Per-class — trumpet is the hard one

| class | recall | ± | precision | support |
|---|---|---|---|---|
| tuba | 0.9976 | 0.0042 | 0.9462 | 137 |
| double-bass | 0.9913 | 0.0000 | 0.9011 | 115 |
| flute | 0.9615 | 0.0204 | 0.9921 | 130 |
| oboe | 0.9612 | 0.0269 | 0.9331 | 86 |
| violin | 0.9577 | 0.0121 | 0.9758 | 126 |
| clarinet | 0.9475 | 0.0198 | 0.9381 | 127 |
| french-horn | 0.9457 | 0.0661 | 0.8353 | 92 |
| bassoon | 0.9113 | 0.0414 | 0.8636 | 109 |
| viola | 0.8957 | 0.0379 | 0.9717 | 115 |
| cello | 0.8943 | 0.0081 | 0.9200 | 123 |
| trombone | 0.8333 | 0.0093 | 0.9049 | 124 |
| **trumpet** | **0.7838** | 0.0234 | 0.9562 | 74 |

**Trumpet — trivially separable at 2 classes — is now the hardest.** It has the fewest training
clips (343) and confuses with trombone (same family) and clarinet (bright, overlapping register).
Note the asymmetry: precision 0.9562 vs recall 0.7838 — when it says trumpet it is right, but it
misses a fifth of them. Class weights softened this without eliminating it.

french-horn has the widest seed spread (±0.0661) and the lowest precision (0.8353) — it absorbs
trombone errors.

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

Run with `python -m instrument_robustness.noise_eval`. Two findings: the specced sweep measures
nothing, and noise *colour* is a non-result once measured honestly.

### 5a. The specced 20/10/0 dB sweep is a dead zone

A clean-trained model falls apart at inaudible noise. White noise, balanced accuracy vs nominal SNR
(chance = 0.0833, clean = 0.9234):

| SNR | balanced acc | MCC | vs clean |
|---|---|---|---|
| 60 dB | 0.9224 | 0.9200 | −0.001 |
| 50 dB | 0.9094 | 0.9071 | −0.014 |
| **45 dB** | 0.8664 | 0.8648 | −0.057 |
| **40 dB** | 0.7882 | 0.7895 | −0.135 |
| 35 dB | 0.6765 | 0.6758 | −0.247 |
| 30 dB | 0.5415 | 0.5374 | −0.382 |
| **20 dB** | 0.3191 | 0.3024 | −0.604 |
| 10 dB | 0.1709 | 0.1232 | −0.753 |
| 0 dB | 0.1207 | 0.0514 | −0.803 |

The knee is at **45–40 dB** — noise at ~1% of the signal amplitude, quieter than a recording studio.
By 20 dB (audible room noise) it's at a third of clean. **All three specced levels (20/10/0) sit
well past the knee**, so six models run there would be near-indistinguishable — the comparison is
close to vacuous. All the resolving power is between 60 and 30 dB. Degradation is graceful (a steady
slide, not the majority-class collapse the 2-class model showed), but it starts absurdly early.

Worse than the 2-class pilot, as expected: more confusable neighbours means less noise is needed to
push a clip across a boundary. **Not a bug** — achieved SNR is exact to <0.01 dB, the clean path
matches train.py to 1e-9 on all 3 seeds, and degradation is monotone.

This is a clean-trained model meeting noise it never saw, so it measures brittleness to distribution
shift, not achievable robustness. The 2-class probe showed noise-aware training turns the cliff into
a gentle slope (0.99 → 0.89 at 0 dB); **expected to hold at 12 classes but not yet measured here.**
That experiment is the natural next step and the one that turns this into the study's actual result.

### 5b. Noise colour is a non-result — only in-band energy matters

Swept white / pink (1/f) / brown (1/f²) at matched *nominal* SNR. On that axis the colours look
wildly different — at 30 dB, brown 0.9086 vs white 0.5415, a 37-point gap that would tempt the
conclusion "brown noise is gentler." **It's an artifact of the SNR definition.** Nominal SNR fixes
total power and ignores *where* the power sits:

| noise | energy <100 Hz | in-band SNR at nominal 0 dB |
|---|---|---|
| white | 0.8% | −0.4 dB (honest) |
| pink | 51.7% | +2.3 dB |
| brown | 99.7% | **+22.2 dB** |

Brown dumps 99.7% of its energy below 100 Hz, under the music, so a "nominal 0 dB" brown clip is
really +22 dB in the 200 Hz–8 kHz band where the notes live. It didn't survive the noise — it was
never given it. Re-plot the *same* balanced-accuracy numbers against **in-band SNR** and the three
colours collapse to within **~0.03–0.05** (the seed noise floor). See `outputs/noise_colors.png` —
left panel (nominal) shows the illusion, right panel (in-band) shows the collapse.

**Takeaway: colour is irrelevant to robustness; only in-band energy is.** Sweeping white alone (the
honest colour) captures everything — the colour axis can be dropped. And **report in-band SNR
alongside nominal** whatever else is decided, or the x-axis lies by up to 22 dB.

### 5c. Open decision (unchanged)

(a) noise-aware / multi-condition training, keep 20/10/0 — proven at 2 classes to give a usable
curve; (b) keep clean training, re-centre the sweep to the 60–30 dB band where the action is —
measures brittleness, but at near-inaudible noise levels; (c) both, as matched vs. mismatched
conditions. Still yours to make.

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
  dropped. `all-samples/manifest.py` independently found and drops the same file (8097 → 8096).
- **Strict single-articulation filtering costs almost nothing.** `normal`/`arco-normal` dominates
  (84–89% of files) rather than being a slice. Same insight as `manifest.py`'s
  `PLAIN = {normal, arco-normal}`.
- **Nothing is padded or tiled.** Clips are variable length (12–87 frames); short notes keep their
  true length, long files are cut into ≤2.0 s chunks capped at 4/file (one 70.66 s recording would
  otherwise yield 35 near-identical chunks). Zero-padding actively **breaks** the noise sweep:
  `power_to_db` clamps digital silence to the −80 dB floor, noise fills it, and the clip lands
  outside the training distribution. Do not reintroduce it.
- **Leak-free split, asserted every run**: grouped by pitch — 544 groups and all 8,375 source
  files, none spanning splits.

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

## 8. Caveats

- **Studio single notes are an upper bound**, not a forecast for real polyphonic audio.
- **Clip length is a weak shortcut at 12 classes.** A length-only classifier scores 0.1914 balanced
  accuracy vs 0.0833 chance (lift +0.108, optimistic — fit and scored on train). Under the 0.15
  warning line but no longer negligible; per-class medians vary (tuba 0.581 s, clarinet 1.440 s).
- **Learning rate.** 1e-4 eliminates the val-loss spikes (4 → 0; worst val 0.677 → 0.962) at the
  cost of convergence at epoch 11 instead of 4. **Not applied — `config.py` still has 1e-3.**
  Gradient clipping does *not* help: Adam's update is scale-invariant in the gradient, so clipping
  shrinks `m` and `√v` together and leaves the ratio unchanged.
- Timing (~17 min/seed) is **CPU-only** (torch 2.4.1+cpu). `get_device()` already auto-detects; a
  GPU only improves it. Pretrained backbones (AST, PANNs) would want one.

## 9. Divergence from `main` worth resolving

- **`all-samples/inventory.py`'s `FAMILY` dict lists only 9 instruments**, so `manifest.csv` covers
  9 (violin, viola, tuba, cello, flute, clarinet, trombone, bassoon, trumpet) and silently drops
  **double-bass, french-horn, oboe**. This pipeline uses 12. Extending that dict would align them.
- **`manifest.csv` is arguably the better data index** — it already carries family, midi,
  duration_s, is_plain, is_phrase. `prep_data.py` currently re-derives all of it from filenames.
  Consuming `manifest.csv` instead would avoid two sources of truth. Blocked on layout: its paths
  point at an `all-samples/` tree (`bassoon/As1/*.mp3`) that isn't committed, while `prep_data.py`
  downloads per-instrument zips to `data/raw/bassoon/*.mp3`.
- **`configs/data/irmas.yaml` and `configs/models/svm.yaml` suggest IRMAS and an SVM baseline** are
  planned. Everything here is Philharmonia + CNN. Worth confirming the intended direction.
- **`src/instrument_robustness/init.py` should be `__init__.py`** — `init.py` is not a package
  marker and does nothing. Added the correct one; left the original in place.
