# rise

Instrument classification and noise-robustness study. **12 classes** (4 strings, 4 woodwinds,
4 brass) from the Philharmonia sample library, medium CNN, multi-seed.

**Read [FINDINGS.md](FINDINGS.md) first.** Headline: balanced accuracy **0.9600 ± 0.0138**
(MCC 0.9618 ± 0.0097) over 3 seeds against a 0.0833 chance floor. Two study-design problems are
documented there, neither of them bugs:

1. **Most of the planned 20/10/0 dB noise sweep can't discriminate models** — by 10 dB a
   clean-trained model has folded onto 6 of 12 classes (MCC 0.149), and by 0 dB onto 4 (MCC 0.037).
   20 dB is still usable. The full label space is only in play between 60 and 30 dB. See §5a.
2. **Bitrate is confounded with class** (64/80/96 kbps across families). Inert at SR=22050,
   a free 3-way shortcut at 44.1 kHz. Do not raise the sample rate without reading §4.

## Setup

```bash
pip install -e .          # or: pip install -r requirements.txt && export PYTHONPATH=src
```

Python 3.10+, PyTorch (CPU is fine — a 3-seed run is ~88 min). MP3 decoding needs no ffmpeg;
librosa routes through soundfile.

## Data — `prep_data.py` is the canonical source

**`prep_data.py` is the only supported way to obtain this dataset. Do not copy a cache from
anyone, and do not use pre-derived feature or window archives.**

```bash
python -m instrument_robustness.prep_data
```

That one command is the whole acquisition path. It downloads the 12 instruments from the Internet
Archive mirror of the Philharmonia library (CC-BY-SA 4.0), applies the strict articulation filter,
builds the fixed-3.0 s tiled log-mel cache, and writes the **pitch-grouped** split. It is
deterministic: the same `config.py` produces a byte-identical cache on any machine.

**Why this is a rule and not a preference.** Every artifact it writes carries
`config.config_fingerprint()`, and every consumer asserts it — a cache built under a different
`SR`, `CLIP_SECONDS`, `MAX_CHUNKS_PER_FILE`, or `CLASSES` makes training **crash** instead of
silently producing a plausible, meaningless model. A cache you received as a file has no way to
prove which config produced it. Regenerating takes ~5 minutes; recovering from a run you later
discover was trained on the wrong cache does not.

Two things it protects that are easy to lose by copying data:

- **Pitch-grouped splits.** The same note at different dynamics is a near-duplicate. Splitting by
  *file* rather than by *pitch* scatters those across train and test and inflates the score. The
  no-leak assertion runs on every build.
- **Tiling, never zero-padding.** Short notes are looped to fill the window, so every sample is
  real signal. Padding with zeros measurably destroys the noise sweep (see FINDINGS §6).

`data/` is gitignored and fully regenerated. Nothing under it should ever be committed or shared.

> **Note for `main`:** `main` currently distributes derived arrays via `download_data.py` (Google
> Drive) and reads them through `RISE_DATA_ROOT`. Those archives are built against `main`'s
> file-level split and zero-padded windows, carry no fingerprint, and are **not** interchangeable
> with this cache. They also cover only 9 instruments — oboe, double-bass, and french-horn are
> absent. Use `prep_data.py` for anything on this branch.

## Run

```bash
python -m instrument_robustness.prep_data --inventory-only   # inventory + codec gate, no audio processing
python -m instrument_robustness.prep_data                    # download, cache, split  (~5 min; ~250 MB down, ~2.7 GB cache)
python -m instrument_robustness.single.train                 # single-instrument, 3 seeds  (~88 min, CPU)
python -m instrument_robustness.single.train --progress      # ...with a pop-up progress bar
python -m instrument_robustness.multi.train                  # multiple-instrument (mixtures)
```

## Layout

The two classification tasks are separated: **`single/`** (one instrument per clip, softmax)
and **`multi/`** (several instruments per clip, sigmoid). Both build on a shared core so
neither depends on the other.

| path | role |
|---|---|
| **shared** | |
| `config.py` | every tunable constant. **Change `CLASSES` to rescope the study.** |
| `prep_data.py` | download, inventory gate, codec check, filter, cache, pitch-grouped split. Owns the canonical `wav_to_logmel`. |
| `cnn_core.py` | the shared CNN: `MediumCNN`, length-bucketed batching, train/eval primitives, SpecAugment. |
| `progress_popup.py` | optional tkinter progress bar (GUI-guarded, no-ops when headless). |
| **`single/` — single-instrument (multi-class)** | |
| `single/train.py` | multi-seed training, evaluation, plots. |
| `single/noise_eval.py` | waveform noise (white/pink/brown) at controlled SNR over all seeds; reports in-band SNR. See FINDINGS §5. |
| `single/audio_demo.py` | renders test clips at each SNR level to WAV, so the degradation is audible. |
| `single/tune_experiment.py` | weight-decay + SpecAugment head-to-head on val (negative result — see FINDINGS). |
| **`multi/` — multiple-instrument (multi-label)** | |
| `multi/train.py` | synthetic mixtures (sum k notes), sigmoid+BCE, mAP evaluation. Path A toward polyphony. |
| **docs** | |
| `configs/*.yaml` | document data/model settings (`config.py` is the source of truth). |
| `all-samples/` | inventory/manifest CSVs + scripts (from `main`). |
| `FINDINGS.md` | results, evidence, caveats, open decisions. |

Data source is the Philharmonia Orchestra sample library via the Internet Archive mirror
(CC Attribution-ShareAlike 4.0) — the official `philharmonia.co.uk/assets/...` URLs predate their
site redesign and no longer resolve, which is why `prep_data.py` points at the mirror. See
**[Data](#data--prep_datapy-is-the-canonical-source)** above for why that script is the only
supported way in.

## Design notes

Choices that are load-bearing and easy to undo by accident:

- **`SR = 22050` is load-bearing for a non-obvious reason.** The classes are encoded at three
  bitrates that cut across families, so the MP3 encoder leaves a class-correlated spectral edge
  above ~14 kHz. At 22050 the Nyquist is 11,025 Hz and it is discarded. At 44.1 kHz it is inside
  the analysis band and hands the model a free 3-way shortcut. `check_bitrates()` enforces this
  each run — do not silence it.
- **Clips are a fixed 3.0 s; short notes are tiled, never zero-padded.** A note shorter than the
  window is looped to fill it, so every clip is 100% real signal. Zero-padding specifically
  **breaks** the noise sweep: `power_to_db` clamps digital silence to the −80 dB floor, noise fills
  it, and the clip lands outside the training distribution (measured: majority-class collapse at
  every SNR). Tiling was tested for the obvious objection — that a looped note's repeated attacks
  leak the source note length, which correlates with class — and **refuted**: both extremes of the
  length distribution (tuba, ~5× repeats; clarinet, least tiled) floor at 0.000 recall under noise,
  there is no monotone length-vs-survival relationship, and 0 dB accuracy is *lower* than the old
  variable-length design. See FINDINGS §6.
- **Artifacts carry a config fingerprint and consumers assert it.** `config.config_fingerprint()`
  is stamped into `manifest.json`, every checkpoint, and `metrics.json`; `load_manifest()` and
  `noise_eval` refuse to run against one built under a different config. A stale checkpoint
  evaluated on a rebuilt cache otherwise produces a plausible, meaningless curve that nothing
  catches. Likewise `metrics.json` always aggregates the canonical `config.SEEDS` from
  `outputs/seed_metrics/`, so a partial run cannot masquerade as a complete one.
- **Pitch-grouped splits.** The same note at different dynamics makes near-identical clips. A plain
  random split scatters them across train and test and inflates the score. `prep_data.py` keeps
  whole pitch-groups in one split and asserts no group — and no source file — spans two.
- **`noise_eval.py` imports `wav_to_logmel` rather than reimplementing it.** If the sweep ran
  through a different spectrogram path than training did, it would be testing the wrong pipeline.
- **Accuracy and F1 are deliberately not reported.** Both pay a collapsed classifier the class
  prior and both have floors that drift with the split. Balanced accuracy (chance = 1/n_classes)
  and MCC (0.0 = no information) have fixed floors. See FINDINGS §7.
- **Report mean ± std over ≥3 seeds.** Seeds 42/43/44 gave 0.9502/0.9540/0.9757 — a 2.55-point
  spread, wider than the margin many model comparisons turn on.
