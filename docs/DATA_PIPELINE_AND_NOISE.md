# Data Pipeline and Noise Benchmark: Implementation Audit and Methods Reference

This document describes the repository at `main` commit `03e3421` as inspected on
2026-07-29. It is both a plain-English guide and an implementation audit. Counts under
**VERIFIED FROM METADATA** come from the current local, fingerprinted Philharmonia build; they are
not copied from old reports or conversation history.

The evidence labels used throughout are:

- **VERIFIED IMPLEMENTATION** — current executable code establishes the behavior.
- **VERIFIED FROM METADATA** — current generated manifests, arrays, audio headers, or artifacts
  establish the fact and agree with their current configuration fingerprints.
- **PLANNED DESIGN** — documented intent that is not completely implemented.
- **INFERENCE** — a reasoned implication of the implementation, not a directly stored fact.
- **UNRESOLVED** — the repository does not currently provide enough evidence or a required feature.

> **Audit boundary:** The canonical local data producer is
> [`prep_data.py`](../src/instrument_robustness/prep_data.py). The current configuration also accepts
> a `build_tinysol_manifest` producer stage, but that builder is not present on `main`
> ([`config.py` L123–127](../src/instrument_robustness/config.py#L123-L127)). The statistics below
> describe the current Philharmonia build, whose `manifest_fingerprint.json` records
> `stage: prep_data`.

## 1. TL;DR

1. The supported source is 12 Philharmonia instrument archives mirrored by the Internet Archive.
2. Filenames encode instrument, note, nominal length, dynamic, and playing technique.
3. Step 0 keeps one configured articulation per instrument and rejects missing or empty files.
4. Step 1 decodes every retained MP3 to mono, resamples it from 44.1 kHz to 22.05 kHz, and stores
   PCM16 WAV.
5. Step 2 removes quiet leading and trailing regions relative to each recording's own peak RMS.
6. Step 3 assigns whole `(instrument, note)` pitch groups to train, validation, or test at
   approximately 70/15/15; different dynamics of the same pitch cannot cross splits.
7. Step 4 makes non-overlapping 3.0-second windows. Short windows are repeated (tiled), not padded
   with zeros; very short final remainders are dropped.
8. Step 5 RMS-normalizes each window toward 0.1 with a peak guard.
9. Train-only Step 6 statistics standardize the 88 SVM features and 128-bin log-mel features made
   in Step 7. Validation and test reuse those statistics.
10. AST, MERT, and PANNs instead load the same normalized 22.05 kHz windows and apply their own
    resampling and pretrained processors.
11. The implemented robustness experiment keeps models frozen and materializes paired noisy copies
    of only the held-out test windows.
12. The current grid is clean plus white, ESC-50 natural, and ESC-50 mechanical noise at
    20, 10, 5, 0, and -5 dB.
13. The mixer uses whole-window power—not an active-region mask—and writes float32 WAV to avoid
    clipping.
14. One deterministic noise realization per window and category is merely rescaled across SNRs, and
    every implemented adapter reads the same materialized files.
15. The central mixer and SVM/MERT/PANNs adapters exist; CNN, CRNN, and AST noise adapters do not.

## 2. Research objective

The scientific question is: **how much does instrument-classification performance deteriorate when
the same held-out musical examples are corrupted by controlled noise, and do model families
deteriorate differently?**

The first experiment is:

```text
fit and select model on clean train/validation data
                         |
                         v
freeze the selected model
                         |
              +----------+----------+
              |                     |
              v                     v
       clean held-out test    paired noisy copies
                              of that same test
```

This separates three different claims:

- **Clean classification performance** is performance on uncorrupted held-out windows.
- **Inherent robustness** is performance of that same clean-trained, frozen model on corrupted
  copies. This is the implemented noise protocol
  ([`NOISE_PLAN.md` L1–8](../NOISE_PLAN.md#L1-L8)).
- **Noise-aware training/augmentation** would add noise during model fitting and require retraining.
  It is explicitly outside the current protocol
  ([`NOISE_PLAN.md` L212–215](../NOISE_PLAN.md#L212-L215)).

> **Potential validity concern:** A clean/noisy comparison measures robustness only to the defined
> corruption process. It does not by itself establish robustness to microphones, rooms,
> reverberation, competing instruments, or other dataset shifts.

## 3. Complete pipeline diagram

```text
Internet Archive Philharmonia ZIPs (12 instruments, MP3)
    |
    | prep_data.build_rows(): parse five filename fields + MP3 header
    v
all-samples/manifest.csv                         [one row / readable source MP3]
    |
    | Step 0: target labels + one articulation/class + file existence
    v
pipeline/manifest_labeled.csv                    [8,378 retained sources]
    |
    | Step 1: librosa decode, mono, 44.1 kHz -> 22.05 kHz
    v
work/resampled/**/*.wav + manifest_resampled.csv [PCM16]
    |
    | Step 2: relative frame-RMS edge trim, top_db=30
    v
work/trimmed/**/*.wav + manifest_trimmed.csv
    |
    | Step 3: group=(label,note), per-label 70/15/15, seed=0
    v
pipeline/splits.csv                              [one split / source and pitch group]
    |
    | Step 4: 3.0 s, 66,150 samples, 3.0 s hop
    |         tile short/final segments; no zero padding
    v
work/windows/**/*.wav + pipeline/windows.csv     [9,116 windows]
    |
    | Step 5: per-window RMS target 0.1, peak <= 0.99
    v
canonical clean window (mono, 22.05 kHz, PCM16)
    |
    +----------------------------+-----------------------------+
    |                            |                             |
    | clean feature/model path   | clean evaluation            | noise test path
    |                            |                             |
    | Step 6 train-only stats    | model-specific              | noise_sweep:
    | Step 7 features            | validation selection        | test windows only
    |                            | then sealed/final test*      |
    |                            |                             +-- white Gaussian
    |                            |                             +-- ESC-50 natural
    |                            |                             +-- ESC-50 mechanical
    |                            |                                      |
    |                            |                             whole-window SNR scaling
    |                            |                                      |
    |                            |                             float32 noisy WAVs +
    |                            |                             provenance + manifest
    |                            |                                      |
    +----------------------------+--------------------------+-----------+
                                                           |
                 same clean or noisy 22.05 kHz waveform ---+
                    |              |               |
                    v              v               v
              88 features       log-mel       model processor
                  SVM          CNN / CRNN    AST/MERT/PANNs
```

`*` SVM and MERT have explicit one-test-access finalizers. AST and PANNs use validation for
selection, but do not implement the same sealed-test guard; see Sections 15 and 28.

> **Verified implementation:** Clean preprocessing precedes noise addition. The noise is added to
> the canonical Step-5 waveform, and model representations are recomputed afterward
> ([`pretrained_extractors.py` L10–12](../src/instrument_robustness/pretrained_extractors.py#L10-L12),
> [`noise_eval_svm.py` L77–82](../src/instrument_robustness/noise_eval_svm.py#L77-L82)).

## 4. Repository map

| File or directory | Main responsibility | Key symbols | Status |
|---|---|---|---|
| `src/instrument_robustness/config.py` | Data roots, labels, pipeline parameters, fingerprints | `TARGET_LABELS`, `SR`, `WINDOW_S`, `config_fingerprint` | CURRENT |
| `src/instrument_robustness/prep_data.py` | Supported acquisition and canonical source manifest | `download_and_extract`, `build_rows`, `MANIFEST_COLUMNS` | CURRENT |
| `all-samples/inventory.py` | Optional MP3 inventory with channels/bitrate | `main`, `FAMILY` | UNCLEAR/non-authoritative |
| `all-samples/manifest.py` | Old inventory-to-manifest script without fingerprints | top-level script | LEGACY; do not run |
| `download_data.py` | Old Google Drive derived-data downloader | `main` exits immediately | LEGACY/deprecated |
| `step0_filter.py` | Label/articulation/file filter | `main`, `STRICT_ARTICULATIONS` | CURRENT |
| `step1_resample.py` | Decode, mono conversion, resampling, PCM16 output | `resample_one`, `sanity_check` | CURRENT |
| `step2_trim.py` | Relative-RMS leading/trailing trim | `trim_one` | CURRENT |
| `step3_split.py` | Pitch-grouped 70/15/15 split | `assign_groups`, `verify_no_group_leak` | CURRENT |
| `step4_window.py` | Fixed windows, tiling, tiny-tail removal | `window_one`, `tile_to_length` | CURRENT |
| `step5_normalize.py` | Per-window RMS normalization in place | `norm_one` | CURRENT |
| `step6_stats.py` | Train-only SVM/log-mel statistics | `_feats`, `main` | CURRENT |
| `step7_featurize.py` | SVM and CNN arrays; CRNN pointer | `_feats`, `_write_crnn_pointer` | CURRENT |
| `featurelib.py` | Shared handcrafted and log-mel functions | `svm_vector`, `logmel`, `load_window` | CURRENT |
| `crnn_data.py` | Transposes CNN arrays into sequences | `load_crnn` | CURRENT loader; no CRNN trainer |
| `ast_data.py`, `train_ast.py` | On-the-fly AST input and fine-tuning | `ASTWindowDataset`, `train` | CURRENT clean model |
| `mert_data.py`, `extract_mert.py`, `mert_probe.py` | Frozen MERT hidden states and layer-weighted probe | `load_mert_examples`, `extract_mert_batch`, `MERTProbe` | CURRENT clean model |
| `train_panns.py` | PANNs probe/fine-tuning | `WindowWaveformDataset`, `PannsClassifier` | CURRENT code; artifacts absent locally |
| `pretrained_extractors.py` | AST/MERT/PANNs sample-rate and processor bridge | `ast_input`, `mert_batch_input`, `panns_input` | CURRENT |
| `noise_sweep.py` | Shared noisy test WAV generation and validation | `draw_noise`, `mix_at_snr`, `validate_noise_manifest` | CURRENT |
| `noise_eval_common.py` | Shared clean-parity and evaluation contract | `run_noise_evaluation`, `assert_clean_parity` | CURRENT |
| `noise_eval_{svm,mert,panns}.py` | Model-specific noisy inference | each module's `main` | CURRENT |
| `noise_stats.py` | Paired cluster bootstrap and tests | `cluster_bootstrap`, `cluster_sign_test` | CURRENT |
| `NOISE_PLAN.md` | Fixed noise protocol and commands | Sections 1–14 | CURRENT |
| `scc/*.qsub`, `scc/README.md` | SCC preparation/training/noise jobs | `noise_generate.qsub`, model jobs | CURRENT |
| `all-samples/manifest.csv` | Canonical source index | relative `path` key | CURRENT local generated data |
| `all-samples/pipeline/*.csv` | Stage, split, and window contracts | sidecar fingerprints | CURRENT metadata |
| `all-samples/pipeline/norm_stats.{npz,json}` | Train-only feature statistics | SVM and mel means/stds | CURRENT local generated data |
| `all-samples/features/{svm,cnn}/*.npz` | Materialized model arrays | `X`, `y`, metadata | CURRENT local generated data |
| `all-samples/features/*/EXTRACTION_PLAN.md` | Pretrained-input notes | model-specific contracts | MIXED; some stale 9-class text |
| `artifacts/{svm,mert,ast}` | Current clean checkpoints/results | summaries and confusion matrices | CURRENT clean results |
| `legacy/9class_file_split/` | Retired leaking 9-class data/results | historical CSVs/checkpoints | LEGACY |
| `tests/test_preprocessing.py` | Split, tiling, and fingerprint regressions | synthetic unit tests | CURRENT |
| `tests/test_noise.py` | SNR, seed, manifest, parity, and statistics tests | `NoiseTests` | CURRENT |
| `tests/test_{svm,mert,ast}.py` | Model data/test-access contracts | model-specific tests | CURRENT |

> **Unresolved documentation conflict:** [`README.md` L86–87](../README.md#L86-L87) points to
> `all-samples/pipeline/pipeline_report.txt`, but that file is absent in the inspected local build.
> The smaller `_step4_report_block.txt` exists. Do not cite a nonexistent report in a paper.

## 5. Source data and metadata

### Discovery and parsing

**VERIFIED IMPLEMENTATION.** `prep_data.download_and_extract` loops over the configured 12 labels,
downloads one ZIP per instrument, finds `*.mp3`, and moves each file under
`<data root>/<instrument>/<note>/`
([`prep_data.py` L76–114](../src/instrument_robustness/prep_data.py#L76-L114)). A valid basename has
five underscore-separated fields:

```text
<instrument>_<note>_<length>_<dynamic>_<technique>.mp3

bassoon_A2_025_forte_normal.mp3
```

`build_rows` rejects a wrong field count, instrument/directory mismatch, unparseable note, or
unreadable MP3, and counts every rejection
([`prep_data.py` L117–166](../src/instrument_robustness/prep_data.py#L117-L166)). The relative path
is the source-recording identifier. Current metadata contain no duplicate source paths.

Pitch is retained twice:

- `note`: Philharmonia spelling such as `A4` or `As4`, where `s` means sharp.
- `midi`: \(12(o+1)+p\), so A4 is 69
  ([`prep_data.py` L43–46, L67–73](../src/instrument_robustness/prep_data.py#L43-L73)).

Dynamics such as `piano`, `mezzo-forte`, and `fortissimo` are retained as strings. The fifth
filename field is stored as `technique`; there is no separate `articulation` column. Trills are
technique strings such as `major-trill`/`minor-trill`. `is_plain` indicates membership in the
configured one-articulation policy, and `is_phrase` is derived from `length == "phrase"`.

An actual current source-manifest record is:

```csv
path,label,family,duration_s,sample_rate,note,midi,dynamic,technique,is_plain,is_phrase
bassoon/A2/bassoon_A2_025_forte_normal.mp3,bassoon,woodwind,0.3135,44100,A2,45,forte,normal,1,0
```

The canonical manifest retains `path`, `label`, `family`, decoded duration and sample rate, pitch,
dynamic, technique, `is_plain`, and `is_phrase`
([`prep_data.py` L54–55](../src/instrument_robustness/prep_data.py#L54-L55)). It does **not** retain
the filename's exact nominal `length` token except for the phrase flag. Nor does it retain MP3
channels, bitrate, byte size, folder-note check, octave, or basename as separate fields. The
non-authoritative `all-samples/inventory.py` can calculate those fields, but the supported pipeline
does not read its output.

**VERIFIED FROM METADATA.**

- 10,197 MP3s are present; 10,196 are readable and represented in `manifest.csv`.
- The fingerprint records one excluded unreadable file. The known file is
  `viola_D6_05_piano_arco-normal.mp3`.
- All 10,196 readable MP3s are mono at 44,100 Hz.
- Header inspection found class-correlated rounded bitrates: 64, 80, or 96 kb/s. Step 1 is intended
  to remove the different high-frequency coding ceilings by lowering Nyquist to 11,025 Hz, but the
  repository does not prove that every lower-frequency codec artifact disappears.
- The raw manifest contains 235 trill-technique rows. Step 0's strict articulation filter retains
  zero of them.

### Why these distributions matter

Pitch ranges are physically different across instruments, but they can also become a shortcut:
a very low pitch may identify tuba or double bass without requiring much timbral understanding.
Dynamics create near-duplicate recordings of the same pitch, which is why they must remain together
at split time. Technique is strongly class/family-correlated in the archive; the strict filter is
intended to limit that shortcut. Duration, leading/trailing quiet, repeated tiling period, MP3
bitrate, and recording-chain artifacts can also correlate with class.

> **Potential validity concern:** Step 1's `sanity_check` prints per-class spectral ceilings, but its
> Boolean only verifies that no output exceeds the target Nyquist
> ([`step1_resample.py` L50–67](../src/instrument_robustness/step1_resample.py#L50-L67)). It does not
> fail on a large *between-class* ceiling spread despite the module documentation saying to stop and
> investigate. Resampling is a strong mitigation for the bitrate shortcut, not proof of its complete
> removal.

## 6. Authoritative label mapping

**VERIFIED IMPLEMENTATION.** The source of truth is
[`config.py` L32–42](../src/instrument_robustness/config.py#L32-L42):

```python
TARGET_LABELS = [
    "bassoon", "cello", "clarinet", "double-bass", "flute", "french-horn",
    "oboe", "trombone", "trumpet", "tuba", "viola", "violin",
]
```

Therefore the numerical mapping is:

| Index | Label | Index | Label |
|---:|---|---:|---|
| 0 | bassoon | 6 | oboe |
| 1 | cello | 7 | trombone |
| 2 | clarinet | 8 | trumpet |
| 3 | double-bass | 9 | tuba |
| 4 | flute | 10 | viola |
| 5 | french-horn | 11 | violin |

Step 7 constructs this mapping by enumeration and stores `label_names` in every SVM/CNN NPZ
([`step7_featurize.py` L33, L56–82](../src/instrument_robustness/step7_featurize.py#L33-L82)).
MERT and PANNs likewise save the label order. Loaders and clean-result summaries compare it against
`TARGET_LABELS`. The configuration fingerprint also embeds the full ordered list
([`config.py` L154–184](../src/instrument_robustness/config.py#L154-L184)).

**VERIFIED FROM METADATA.** All six current SVM/CNN arrays contain exactly the order above, and the
current SVM, MERT, and AST clean-result files report the same 12 labels.

> **Legacy warning:** `legacy/9class_file_split/` and the disabled Google Drive archives use a
> different 9-class mapping without oboe, double bass, or French horn. Label indices shifted when
> the three classes were added. `download_data.py` now refuses to run
> ([`download_data.py` L253–267](../download_data.py#L253-L267)).

> **Stale documentation:** The AST and PANNs extraction-plan files still contain isolated references
> to a “9-way” head, while current model code uses `len(TARGET_LABELS)` and is 12-way. Current code
> and fingerprinted generated metadata take precedence.

## 7. Source-level data splitting

**VERIFIED IMPLEMENTATION.** Splitting happens after resampling/trimming but before windowing. The
indivisible unit is a pitch group

\[
g_i=(\mathrm{label}_i,\mathrm{note}_i).
\]

All source recordings sharing an instrument and note—including different dynamics and nominal
lengths—receive the same split. Within each instrument, groups are shuffled with Python
`random.Random(SEED)`, sorted largest-first, and greedily assigned to the split with the greatest
remaining file-count deficit
([`step3_split.py` L49–77](../src/instrument_robustness/step3_split.py#L49-L77)). Constants are:

```text
train / validation / test target = 0.70 / 0.15 / 0.15
seed                           = 0
group fields                   = label, note
```

The assignment is performed independently per class for label stratification
([`step3_split.py` L110–119](../src/instrument_robustness/step3_split.py#L110-L119)). Unequal group
sizes mean exact fractions are not guaranteed.

`all-samples/pipeline/splits.csv` is the authoritative source-level assignment for downstream
windowing. Rerunning Step 3 overwrites it; there is no “generate once” lock. Given identical input,
configuration, Python behavior, and seed, the implementation is deterministic. Models should read
the fingerprinted existing file, not independently resplit.

**VERIFIED FROM METADATA.**

- 8,378 retained sources: 5,864 train (69.993%), 1,259 validation (15.027%), 1,255 test (14.980%).
- 544 `(label,note)` groups; zero groups cross splits.
- Every label occurs in every split.

`verify_no_group_leak` asserts that a group has at most one split
([`step3_split.py` L80–92](../src/instrument_robustness/step3_split.py#L80-L92)). Step 4 copies each
source's split tag to every derived window and asserts one split per source
([`step4_window.py` L86–119](../src/instrument_robustness/step4_window.py#L86-L119)).

Randomly splitting windows would be invalid because several windows may be excerpts of the same
source; even different source files at the same pitch and dynamic family can be near-duplicates.
The retired file-level split leaked 406 of 436 old pitch groups, as recorded in
[`step3_split.py` L15–17](../src/instrument_robustness/step3_split.py#L15-L17).

## 8. Audio decoding and resampling

The supported acquisition path discovers MP3 only. `librosa.load(path, sr=22050, mono=True)` decodes
and simultaneously resamples each file; `mono=True` averages channels when necessary
([`step1_resample.py` L27–38](../src/instrument_robustness/step1_resample.py#L27-L38)). Current inputs
are already mono, but the conversion is still explicit.

| Property | Source | Step-1 output |
|---|---|---|
| Format | MP3 | WAV |
| Sample rate | 44,100 Hz for all readable current files | 22,050 Hz |
| Channels | mono for current files | forced mono |
| In-memory type | librosa floating waveform | floating waveform |
| Stored subtype | compressed MP3 | signed 16-bit PCM |
| Path | `<instrument>/<note>/*.mp3` | `work/resampled/<same stem>.wav` |

`soundfile.write(..., subtype="PCM_16")` quantizes the output to signed 16-bit PCM
([`step1_resample.py` L32–36](../src/instrument_robustness/step1_resample.py#L32-L36)). The code does
not explicitly clamp or validate decoded amplitude before this write. It also does not perform DC
offset removal or loudness normalization here.

A fixed sample rate makes:

- 3.0 seconds equal exactly \(3(22050)=66{,}150\) samples;
- FFT/mel dimensions consistent across examples;
- clean/noisy waveform mixing sample-aligned;
- all non-pretrained models comparable at the same bandwidth.

Pretrained models later resample this common waveform to their required rates. This preserves one
canonical clean/noisy source while respecting each pretrained model's input contract.

## 9. Silence detection and trimming

### What was repaired

The retired pipeline produced fixed windows with zero padding. Quiet source clips therefore had
large synthetic-silence regions. Added noise filled those regions, moving noisy spectrograms far
outside the clean training distribution and causing measured majority-class collapse. The current
repair combines conservative edge trimming with **tiling instead of zero padding**. The tiling
change is present in Git history commit `d9b788f` and in current
[`step4_window.py` L1–20](../src/instrument_robustness/step4_window.py#L1-L20).

### Current trim algorithm

**VERIFIED IMPLEMENTATION.** Step 2 calls:

```python
yt, _ = librosa.effects.trim(y, top_db=30)
```

([`step2_trim.py` L18–29](../src/instrument_robustness/step2_trim.py#L18-L29)). With the installed
librosa 0.11 defaults, this uses 2,048-sample frames, a 512-sample hop, maximum frame RMS as the
reference, and an RMS frame calculation. For frame \(k\),

\[
\operatorname{RMS}(k)=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{k,i}^{2}},\qquad N=2048.
\]

Librosa converts RMS amplitude to decibels relative to the recording's maximum frame RMS,
conceptually

\[
D(k)=20\log_{10}
\left(
\frac{\max(\operatorname{RMS}(k),a_{\min})}
{\max(\max_j \operatorname{RMS}(j),a_{\min})}
\right),
\]

and treats frames with \(D(k)>-30\) dB as non-silent. The output is the contiguous interval between
the first and last such frames. It does **not** remove quiet gaps inside that interval.

The threshold is relative to each recording's own peak, not an absolute activity threshold. There
is no explicit pre/post context parameter. An attack or decay remains only to the extent that it is
inside the returned frame-aligned interval or lies above the relative threshold. If trimming leaves
less than 0.10 seconds (2,205 samples), Step 2 restores the entire untrimmed resampled signal and
records `trim_flag="kept_untrimmed"` ([`config.py` L80–82](../src/instrument_robustness/config.py#L80-L82)).

**VERIFIED FROM METADATA.**

- 8,375 sources have `trim_flag=ok`; 3 use `kept_untrimmed`.
- Median duration changes from 0.9927 s to 0.9056 s.
- Mean removed edge duration is 0.0864 s; maximum is 1.9070 s.
- No rejection statistic for “silent recording” exists.

> **Verified implementation limitation:** There is no stored frame activity mask, active interval,
> active sample count, active fraction, or per-window activity statistic. Empty decoded files are
> rejected in Step 1, but an effectively silent nonempty recording is not explicitly rejected.
> With the installed librosa 0.11 peak-relative default, an all-zero array is returned untrimmed
> because every zero-RMS frame equals the zero-valued reference after numerical flooring. Step 5
> would then leave it unchanged. No current generated window is that quiet (minimum recorded
> pre-normalization RMS is 0.00051), but this behavior lacks a regression test.

> **Important distinction:** `content_s` in `windows.csv` is the number of source samples in a
> segment before tiling. It is not an active-audio duration and cannot be used as an SNR mask.

## 10. Windowing, cropping, and padding

The current canonical contract is:

| Setting | Value | Evidence |
|---|---:|---|
| Window duration | 3.0 s | `WINDOW_S` |
| Samples | 66,150 | `3.0 × 22,050` |
| Hop/stride | 3.0 s | `HOP_S` |
| Overlap | none | hop equals window |
| Short-segment policy | repeat/tile to length | `tile_to_length` |
| Tiny final remainder | drop if below 0.5 s | `MIN_WINDOW_CONTENT_S` |
| Only window of a short source | always keep and tile | `wi != 0` exception |
| Long source | consecutive onset-aligned windows | `range(0, n, HOP)` |
| Activity centering | none | starts are fixed multiples of the hop |
| Zero padding in Step 4 | none | empty segments raise |

Constants are in [`config.py` L89–100](../src/instrument_robustness/config.py#L89-L100), and the
implementation is in [`step4_window.py` L43–84](../src/instrument_robustness/step4_window.py#L43-L84).

For a 0.9-second trimmed note:

```text
recorded segment: | attack -- sustain -- decay |
3 s window:       | attack--decay | attack--decay | attack---|
                   <--------- repeated source samples -------->
```

For a 7.2-second source:

```text
source:  |--------- 3.0 ---------|--------- 3.0 ---------|-1.2-|
outputs: |          w000          |          w001          | w002 tiled to 3 s
starts:  0.0 s                    3.0 s                    6.0 s
```

If the final remainder were below 0.5 s, it would be dropped. `content_s` records 3.0 for a full
window or the real pre-tile remainder for a tiled window. The path preserves instrument/note
directories and appends `_w000`, `_w001`, and so on; `window_id_of` later uses the basename stem
([`noise_sweep.py` L296–297](../src/instrument_robustness/noise_sweep.py#L296-L297)).

**VERIFIED FROM METADATA.** There are 9,116 windows from 8,378 sources. All 9,116 physical WAV
headers were checked: mono, 22,050 Hz, PCM16, and exactly 66,150 frames. There are no duplicate
`window_path` values. A source contributes 1.088 windows on average (median 1, maximum 26). Of all
windows, 8,341 (91.50%) have `content_s < 3` and were tiled; 775 contain a full 3.0 seconds before
tiling.

> **Potential validity concern:** Tiling removes synthetic silence but repeats attacks and encodes
> the original segment period. The repository documents an earlier CNN analysis arguing this did
> not explain noisy recall, but that external analysis is not reproduced by current tests. Current
> per-class median `content_s` ranges from 0.580 s (tuba) to 1.486 s (clarinet), so duration-related
> shortcuts remain worth reporting.

> **Defensive-loader exception:** Canonical Step 4 never zero-pads, but
> [`featurelib.load_window` L11–17](../src/instrument_robustness/featurelib.py#L11-L17) defensively
> zero-pads a physically short file and truncates a long one. SVM/CNN Step 7, MERT extraction, and
> PANNs use this loader. AST and the noise system instead reject a wrong sample count. The current
> WAV audit found no wrong-length files, but the code paths are not identical on malformed data.

## 11. Waveform normalization

The exact clean waveform order is:

```text
decode/resample/mono -> PCM16
    -> relative-RMS edge trim -> PCM16
    -> fixed window/tile -> PCM16
    -> per-window RMS normalization with peak guard -> PCM16 in place
    -> feature extraction or pretrained processor
```

There is no explicit DC-offset removal, peak normalization to unit amplitude, dataset-level
waveform normalization, or perceptual loudness standard such as LUFS.

For a window \(x\), Step 5 calculates

\[
r=\sqrt{\frac{1}{T}\sum_{t=1}^{T}x_t^2},\qquad
g_0=\frac{0.1}{r}.
\]

If \(\max_t |g_0x_t|>0.99\), it changes the gain to

\[
g=g_0\frac{0.99}{\max_t|g_0x_t|};
\]

otherwise \(g=g_0\). It writes \(x'=gx\) back to the same WAV path as PCM16
([`step5_normalize.py` L21–33](../src/instrument_robustness/step5_normalize.py#L21-L33)). A window
with RMS below \(10^{-6}\) is left unchanged.

**VERIFIED FROM METADATA.** Median post-normalization RMS is 0.10000. Fifteen of 9,116 windows are
more than 0.001 below target because of the peak guard; the minimum post-RMS is 0.05325.

Waveform normalization and feature standardization are different:

- waveform RMS normalization changes audio samples independently per window;
- feature standardization later changes each feature coordinate using statistics fitted on train.

The order matters for SNR. The implemented mixer measures the actual final clean-window power, so
it does not simply assume \(0.1^2\). It does not normalize again after mixing
([`NOISE_PLAN.md` L82–92](../NOISE_PLAN.md#L82-L92)).

## 12. Clean-data manifests and data contracts

### Stage files

| Path | One row represents | Purpose / downstream reader |
|---|---|---|
| `all-samples/manifest.csv` | one readable source MP3 | canonical acquired index; Step 0 |
| `pipeline/manifest_labeled.csv` | one retained source | label/articulation contract; Step 1 and noise cluster join |
| `pipeline/manifest_resampled.csv` | one attempted retained source | adds resampled path/duration/status; Step 2 |
| `pipeline/manifest_trimmed.csv` | one successfully resampled source | adds trimmed path/duration/flag; Step 3 |
| `pipeline/splits.csv` | one source assignment | authoritative split; Step 4 |
| `pipeline/windows.csv` | one derived window | authoritative waveform/split table; Steps 5–7 and pretrained loaders |
| `pipeline/norm_stats.npz` | one train-statistics bundle | Step 7 and SVM noisy inference |
| `pipeline/norm_stats.json` | human-readable statistics bundle | audit/reporting |

Each important CSV has a JSON sidecar containing its SHA-256, producer stage, and complete
configuration fingerprint
([`config.py` L246–283](../src/instrument_robustness/config.py#L246-L283)). Consumers verify both
the sidecar's CSV hash and its configuration. Step 5 rewrites `windows.csv` with RMS columns and
replaces its producer stage with `step5_normalize`.

### Important fields

| Field | Meaning | Type | First source | Required downstream? |
|---|---|---|---|---|
| `path` | source ID and relative MP3 path | string | `manifest.csv` | yes through Step 3 |
| `label` | instrument name | string | manifest | yes |
| `family` | strings/woodwind/brass metadata | string | manifest | analysis; not model array input |
| `duration_s` | decoded original MP3 duration | float seconds | manifest | carried, not used by models |
| `sample_rate` | original header rate | integer Hz | manifest | audit |
| `note`, `midi` | symbolic and numeric pitch | string/int | manifest | `note` required for grouped split |
| `dynamic` | playing dynamic | string | manifest | carried through trim; analysis |
| `technique` | playing articulation/technique | string | manifest | Step-0 filter |
| `is_plain` | configured articulation indicator | integer 0/1 | manifest | audit/filter context |
| `is_phrase` | filename length was `phrase` | integer 0/1 | manifest | retained in `splits.csv` |
| `resampled_path` | relative Step-1 WAV | string | resampled manifest | Step 2 |
| `resampled_dur_s` | decoded resampled duration | float | resampled manifest | trim audit |
| `status` | resample outcome | string | resampled manifest | Step 2 filters `ok` |
| `trimmed_path` | relative Step-2 WAV | string | trimmed manifest | Step 4 |
| `trimmed_dur_s` | post-trim duration | float | trimmed manifest | audit |
| `trim_flag` | `ok` or fallback/error | string | trimmed manifest | audit |
| `source_path` | source ID copied from `path` | string | `splits.csv` | grouping and evaluation |
| `split` | `train`, `val`, or `test` | string | `splits.csv` | all model loaders |
| `window_path` | relative Step-5 waveform | string | `windows.csv` | all models/noise |
| `start_time` | onset-aligned source offset | float seconds | `windows.csv` | audit |
| `content_s` | samples present before tiling | float seconds | `windows.csv` | audit only |
| `pre_norm_rms`, `post_norm_rms` | Step-5 RMS values | float | `windows.csv` | audit/parity |

There is no explicit numeric label in these CSVs; arrays derive it from `TARGET_LABELS`. There is no
explicit `window_id` column; the noise code derives it from `Path(window_path).stem`. There are no
stored `sample_count`, `active_duration`, `active_fraction`, or activity-mask fields.

> **Potential validity concern:** CSV fingerprints hash metadata but not every clean WAV's bytes.
> The noise provenance later hashes each selected clean test WAV. The clean feature/model loaders
> generally trust the window files after checking the CSV fingerprint.

## 13. Handcrafted feature pipeline for the SVM

`featurelib.svm_vector` calculates frame-level descriptors and summarizes every row by temporal
mean and population standard deviation
([`featurelib.py` L30–61](../src/instrument_robustness/featurelib.py#L30-L61)):

\[
\mu_j=\frac{1}{T}\sum_{t=1}^{T}f_{t,j},\qquad
\sigma_j=\sqrt{\frac{1}{T}\sum_{t=1}^{T}(f_{t,j}-\mu_j)^2}.
\]

| Feature group | Frame dimensions | Saved summaries | Output dimensions | Intuition |
|---|---:|---|---:|---|
| MFCC | 20 | mean, std per coefficient | 40 | coarse spectral-envelope/timbre shape |
| Chroma STFT | 12 | mean, std per pitch class | 24 | pitch-class energy |
| Spectral centroid | 1 | mean, std | 2 | spectral “brightness” center |
| Spectral bandwidth | 1 | mean, std | 2 | spread around centroid |
| Spectral rolloff | 1 | mean, std | 2 | frequency below which default energy fraction lies |
| Spectral contrast | 7 bands | mean, std per band | 14 | peak/valley contrast across bands |
| Zero-crossing rate | 1 | mean, std | 2 | rapid sign changes/noisiness |
| RMS energy | 1 | mean, std | 2 | local amplitude energy |
| **Total** |  |  | **88** |  |

The spectral functions use `N_FFT=2048` and `HOP=512`; MFCC count is 20
([`config.py` L105–114](../src/instrument_robustness/config.py#L105-L114)). Parameters not passed
explicitly—such as rolloff percentage or MFCC DCT choices—are the installed librosa defaults and
should be version-pinned before final paper reproduction.

Step 6 fits one mean and standard deviation per feature using **train windows only**. Standard
deviations below \(10^{-8}\) are replaced by 1
([`step6_stats.py` L39–76](../src/instrument_robustness/step6_stats.py#L39-L76)). Step 7 applies:

\[
z_j=\frac{x_j-\mu_{j,\mathrm{train}}}{\sigma_{j,\mathrm{train}}}
\]

to train, validation, and test without refitting
([`step7_featurize.py` L56–74](../src/instrument_robustness/step7_featurize.py#L56-L74)).
`train_svm.py` deliberately loads these already-standardized arrays without a second scaler.

Current files:

```text
all-samples/features/svm/train.npz  X (6487, 88), y (6487,)
all-samples/features/svm/val.npz    X (1319, 88), y (1319,)
all-samples/features/svm/test.npz   X (1310, 88), y (1310,)
```

Every `X` is float32 and `y` is int64. Keys are `X`, `y`, `source_path`, `feature_names`,
`label_names`, and `config_fingerprint`. The NPZ does not store `window_path`, so ordering is tied
to the filtered order of `windows.csv`; multiple windows from one source repeat `source_path`.

For noisy SVM inference, the adapter recomputes the same raw 88 features from each noisy waveform,
loads the saved Step-6 means/stds, and applies them
([`noise_eval_svm.py` L41–82](../src/instrument_robustness/noise_eval_svm.py#L41-L82)). It never fits
statistics on noisy data.

## 14. Log-mel pipeline for CNN and CRNN

`featurelib.logmel` calls `librosa.feature.melspectrogram` with:

```text
sample rate       22,050 Hz
FFT length        2,048
hop               512 samples (23.22 ms)
mel bins          128
frequency range   0 to 11,025 Hz
power exponent    2.0 (librosa default)
window            Hann (librosa/STFT default)
centered frames   yes; boundary padding is internal to the STFT
```

([`featurelib.py` L20–27](../src/instrument_robustness/featurelib.py#L20-L27)). Conceptually:

\[
X(m,k)=\sum_{n=0}^{N-1}x[n+mH]w[n]e^{-j2\pi kn/N},
\quad
P(m,k)=|X(m,k)|^2,
\]

\[
M(m,r)=\sum_k H_r(k)P(m,k).
\]

The repository then uses `librosa.power_to_db(M, ref=1.0)`. Conceptually,

\[
L(m,r)=10\log_{10}\left(\frac{\max(M(m,r),a_{\min})}{1.0}\right),
\]

with librosa's default 80 dB floor. This is an absolute reference of 1.0, not `ref=np.max`.

Step 6 pools every frame of every train window to fit a separate mean and standard deviation for
each mel bin ([`step6_stats.py` L45–68](../src/instrument_robustness/step6_stats.py#L45-L68)). Step 7
broadcasts those 128 train-only values over time, then adds a channel axis:

```text
CNN:  (N, mel=128, frames=130, channels=1)
CRNN: load same arrays, drop channel, transpose -> (N, frames=130, mel=128)
```

Current float32 arrays have shapes:

| Split | CNN `X` | `y` |
|---|---|---|
| train | `(6487, 128, 130, 1)` | `(6487,)` |
| validation | `(1319, 128, 130, 1)` | `(1319,)` |
| test | `(1310, 128, 130, 1)` | `(1310,)` |

`crnn_data.load_crnn` performs the transpose and verifies the fingerprint
([`crnn_data.py` L13–23](../src/instrument_robustness/crnn_data.py#L13-L23)). There is no CNN or
CRNN model/training script on current `main`; only their inputs and CRNN loader exist.

Noise must be added in the linear waveform domain before this transform. Adding an arbitrary matrix
to a clean log-mel tensor would not equal a physical waveform mixture because power, mel filtering,
the logarithm, and standardization are nonlinear operations.

## 15. Raw-waveform path for pretrained models

All pretrained paths start from the same Step-5 mono 3.0-second, 22.05 kHz window. They do **not**
use Step-6 SVM/log-mel statistics
([`pretrained_extractors.py` L1–16](../src/instrument_robustness/pretrained_extractors.py#L1-L16)).

| Model | Current implementation status | Input path and rate | Representation / training |
|---|---|---|---|
| AST | Clean loader/trainer and clean artifacts implemented; no noise adapter | Resample to 16 kHz, then `ASTFeatureExtractor`; expected `(1,1024,128)` per example | Fine-tunes `MIT/ast-finetuned-audioset-10-10-0.4593`; selects best validation balanced accuracy |
| MERT | Clean extraction/probe/finalizer, artifacts, and noise adapter implemented | Resample to 24 kHz; pinned `Wav2Vec2FeatureExtractor`/MERT revision | Frozen backbone; mean over time for each of 13 hidden states, shape `(N,13,768)`; learned layer mixture + linear head |
| PANNs CNN14 | Probe/fine-tune code and noise adapter implemented; no local checkpoint/results | Resample to 32 kHz, yielding 96,000 samples; CNN14 computes 64-bin log-mel internally | 2,048-D embedding; frozen probe or full fine-tune |

AST resampling/processing is in
[`pretrained_extractors.py` L54–84](../src/instrument_robustness/pretrained_extractors.py#L54-L84).
`ASTWindowDataset` strictly checks 66,150 source samples before processing
([`ast_data.py` L100–115](../src/instrument_robustness/ast_data.py#L100-L115)). The model processor
owns padding/truncation and normalization. The repository does not hard-code its normalization
values.

MERT uses a commit-pinned `m-a-p/MERT-v1-95M` processor and backbone
([`config.py` L116–121](../src/instrument_robustness/config.py#L116-L121)). Equal-length 3 s
waveforms become 72,000 samples at 24 kHz; `padding=True` is still passed for batching
([`pretrained_extractors.py` L103–114](../src/instrument_robustness/pretrained_extractors.py#L103-L114)).
Each hidden state is mean-pooled over model time
([`extract_mert.py` L65–96](../src/instrument_robustness/extract_mert.py#L65-L96)).

PANNs resampling and internal front-end parameters are documented in
[`pretrained_extractors.py` L35–50](../src/instrument_robustness/pretrained_extractors.py#L35-L50).
The current `train_panns.py` uses a separate linear head instead of the stale plan's described
9-way replacement.

> **Potential validity concern:** AST creates its test loader before training and PANNs probe mode
> precomputes test embeddings before validation selection. Neither uses test labels for selection,
> but their test-access policy is less fail-closed than SVM/MERT. AST's default output directory is
> `$RISE_DATA_ROOT/models/ast`, while clean AST results are committed under `artifacts/ast`; the
> transfer process is not documented.

## 16. Noise benchmark design

**VERIFIED IMPLEMENTATION.**

```text
Training:   clean train only
Selection:  clean validation only
Testing:    frozen selected model on clean test, then paired noisy copies of test
```

The condition constants are
[`noise_sweep.py` L47–64](../src/instrument_robustness/noise_sweep.py#L47-L64):

```python
SNRS = [20, 10, 5, 0, -5]
NOISE_TYPES = ["white", "natural", "mechanical"]
```

This gives 15 noisy conditions plus one shared clean condition. Every test window occurs in every
condition. Train and validation windows are never noised by the generator. The clean condition is
not copied; evaluators read the canonical Step-5 WAVs.

Before noisy scoring, the common evaluator must reproduce the official clean test example count and
macro-F1 within \(10^{-3}\), or it aborts
([`noise_eval_common.py` L152–177, L273–280](../src/instrument_robustness/noise_eval_common.py#L152-L177)).
This protects against using a wrong checkpoint, label map, data build, or inference path.

**VERIFIED FROM METADATA:** No local `work/windows_noisy/noise_manifest.json` exists in the inspected
data root, and no current `artifacts/*/noise/` results are present. Thus the protocol is implemented
but the current local checkout does not contain a completed noise benchmark.

## 17. Noise sources and categories

| Concept | Current repository status | Current mapping |
|---|---|---|
| Gaussian white noise | IMPLEMENTED | `white` → generated standard normal samples |
| ESC-50 structured events | IMPLEMENTED | `natural` → targets 0–19; `mechanical` → targets 30–49 |
| ESC-50 human non-speech | EXCLUDED | targets 20–29 are omitted |
| DEMAND ambience | CONSIDERED, NOT IMPLEMENTED | explicitly dropped in `NOISE_PLAN.md` |
| MUSAN speech/music | NOT IMPLEMENTED | no code/config |
| Indoor/public/outdoor subgroups | NOT IMPLEMENTED | no project mapping |
| Competing instruments/music | NOT IMPLEMENTED | would change interpretation toward interference/multi-label recognition |

ESC-50 selection requires both `audio/` and `meta/esc50.csv`. `load_esc50_index` selects files by
integer `target`, sorts their filenames, verifies all files exist, and requires 800 clips in each
project category ([`noise_sweep.py` L132–164](../src/instrument_robustness/noise_sweep.py#L132-L164)).
The code ignores ESC-50's category-name and fold fields. Human non-speech is excluded by target
range. Repository download instructions are in [`NOISE_PLAN.md` L184–195](../NOISE_PLAN.md#L184-L195);
the repository does not record ESC-50 licensing terms, so none are asserted here.

White noise is mathematically unstructured with equal expected power per frequency. ESC-50 consists
of finite structured events, which may have transients and nonstationary spectra. DEMAND would
represent longer continuous ambience, but it is not in this protocol. Speech and competing music
would introduce semantic/acoustic sources unlike generic background noise; competing target
instruments could make the nominal single-label ground truth ambiguous.

> **Potential validity concern:** The current project labels “natural” and “mechanical” each combine
> 20 ESC-50 target classes. Only total pool size is checked. Original category/fold is not copied
> into per-mixture provenance, and no content check rejects a noise file containing a target-like
> instrument.

## 18. Noise-recording split isolation

A future noise-augmented-training study should catalog external recordings and split by original
noise source—not by cropped excerpt:

```text
noise source recording
    +-- all training excerpts      -> noise train only
    +-- all validation excerpts    -> noise validation only
    `-- all test excerpts          -> noise test only
```

Otherwise, nearly identical background excerpts can appear during augmentation and evaluation.
A useful catalog would contain:

| Field | Meaning |
|---|---|
| `noise_id` | stable catalog item ID |
| `dataset` | ESC-50, DEMAND, etc. |
| `project_category` | white/natural/mechanical/etc. |
| `original_category` | corpus's original label |
| `source_recording_id` | indivisible split group |
| `file_path` | corpus-relative path |
| `split` | noise train/validation/test |
| `sample_rate`, `duration_s` | source audio properties |

> **Unresolved / not implemented:** The current experiment has no external-noise split or catalog.
> It draws from all selected 800 ESC-50 files per category when corrupting clean **test** windows.
> This does not leak noise into clean model training because current models never train on noise,
> but it is insufficient for any future augmentation experiment. ESC-50's own fold column is not
> used.

## 19. Gaussian white-noise generation

**VERIFIED IMPLEMENTATION.** Given the deterministic per-window RNG:

```python
noise = rng.standard_normal(CLIP_LEN).astype(np.float32)
```

([`noise_sweep.py` L206–218](../src/instrument_robustness/noise_sweep.py#L206-L218)). This creates
exactly 66,150 float32 samples with theoretical mean 0 and variance 1 before SNR scaling. A finite
draw is not guaranteed to have exactly zero empirical mean or unit empirical variance; the mixer
measures its actual power.

Downloaded white-noise recordings are unnecessary because the random generator and stable seed
fully define the realization. The source provenance is `generated_gaussian`, source rate 22,050 Hz,
and crop start 0.

## 20. External noise-segment selection

For `natural` or `mechanical`, `draw_noise`:

1. chooses an ESC-50 path using the deterministic RNG and sorted category index;
2. reads float32 audio with SoundFile;
3. averages channels if stereo;
4. resamples to 22,050 Hz with `librosa.resample` if necessary;
5. tiles a source shorter than 66,150 samples rather than zero-padding;
6. draws a start uniformly from all valid resampled-sample offsets;
7. keeps exactly 66,150 samples;
8. rejects segments with RMS below \(10^{-6}\), trying at most 20 times;
9. returns source-relative path, absolute path for hashing, original source sample rate, and
   resampled crop start.

See [`noise_sweep.py` L189–241](../src/instrument_robustness/noise_sweep.py#L189-L241).

> **Verified implementation deviation from the proposed generic recipe:** No DC-offset removal is
> performed. The stored crop offset is measured in the **resampled** 22.05 kHz waveform, not in the
> original file's sample coordinates. Crop end is implicit as `start + 66150`, not stored.

The RNG seed determines both source-file selection and crop start. The same draw is reused across
all SNR levels for that clean window and noise type.

## 21. Active-region SNR

### General active-region definition

If a window contains padding or long silence, whole-window power can understate the instrument's
power during its sounding region. Let

\[
A=\{t:\text{sample }t\text{ belongs to active instrument audio}\}.
\]

An active-region definition would be:

\[
P_x^{(A)}=\frac{1}{|A|}\sum_{t\in A}x_t^2,\qquad
P_n^{(A)}=\frac{1}{|A|}\sum_{t\in A}n_t^2,
\]

\[
\operatorname{SNR}_{\mathrm{dB}}^{(A)}
=10\log_{10}\left(\frac{P_x^{(A)}}{P_n^{(A)}}\right),
\qquad
\alpha=\sqrt{\frac{P_x^{(A)}}{P_n^{(A)}10^{s/10}}},
\]

\[
y_t=x_t+\alpha n_t.
\]

Here \(x\) is the clean waveform, \(n\) is unscaled noise, \(s\) is requested SNR in dB,
\(\alpha\) is noise gain, and \(y\) is the mixture.

### What this repository actually does

> **Verified implementation: whole-window SNR, not active-region SNR.**

`mix_at_snr` uses every one of the 66,150 samples:

\[
P_x=\frac{1}{T}\sum_{t=1}^{T}x_t^2,\qquad
P_n=\frac{1}{T}\sum_{t=1}^{T}n_t^2,
\]

\[
\alpha=\sqrt{\frac{P_x}{P_n10^{s/10}}},\qquad y=x+\alpha n.
\]

The exact code is [`noise_sweep.py` L244–258](../src/instrument_robustness/noise_sweep.py#L244-L258);
measurement uses the same whole-array definition
([`noise_sweep.py` L261–268](../src/instrument_robustness/noise_sweep.py#L261-L268)).

The tiling repair is important here: canonical windows have no synthetic zero-padded tail, so
whole-window power covers repeated trimmed signal rather than a mixture of instrument and added
zeros. However, it still includes naturally quiet portions within each repeated segment.

Interpretation:

| SNR | Signal-to-noise power ratio |
|---:|---:|
| 20 dB | \(100:1\) |
| 10 dB | \(10:1\) |
| 5 dB | \(3.162:1\) |
| 0 dB | \(1:1\) |
| -5 dB | \(0.316:1\); noise power is 3.162 times signal |

> **Unresolved:** No active mask exists from which the requested alternative could be implemented
> or audited. A paper must describe the current benchmark as **whole-window power SNR**. Calling it
> active-region SNR would be incorrect.

## 22. Clipping prevention

Adding noise can make \(\max_t|y_t|>1\). Hard clipping,
\(\operatorname{clip}(y,-1,1)\), is nonlinear: it changes waveform shape and destroys the requested
signal/noise power relationship.

A possible common peak-protection alternative is

\[
\beta=\min\left(1,\frac{p_{\max}}{\max_t|y_t|}\right),\qquad y'_t=\beta y_t.
\]

Applying \(\beta\) to the complete mixture preserves SNR because signal and noise receive the same
gain.

> **Verified implementation:** The noise generator does **not** apply \(\beta\), hard clipping, or
> post-mix normalization. It writes `subtype="FLOAT"` WAV, which preserves values beyond
> \([-1,1]\), and records the reloaded peak
> ([`noise_sweep.py` L334–346, L516–553](../src/instrument_robustness/noise_sweep.py#L334-L346)).
> The manifest records `post_mix_normalization: false`.

The unit test writes and reloads a float WAV whose samples are 2.5, verifying that the shared reader
does not clamp it ([`tests/test_noise.py` L188–200](../tests/test_noise.py#L188-L200)).

> **Potential validity concern:** Float headroom preserves mathematical SNR, but values above the
> usual normalized waveform range enter model processors. The clean-parity gate cannot test this
> noisy-range behavior. The final Methods should report float32 storage and lack of post-mix scaling
> explicitly.

## 23. Stable seeds and deterministic regeneration

The current seed is:

```text
uint32(first 4 bytes of
SHA256(dataset_fingerprint + "|" + window_id + "|" + noise_type))
```

([`noise_sweep.py` L119–129](../src/instrument_robustness/noise_sweep.py#L119-L129)). The dataset
fingerprint itself hashes the configuration, actual source-manifest SHA-256, and actual Step-5
`windows.csv` SHA-256
([`noise_sweep.py` L85–116](../src/instrument_robustness/noise_sweep.py#L85-L116)).

Python's built-in `hash()` is not used because its value is not a stable persistent identifier
across interpreter processes. SHA-256 is stable.

Notably absent from the seed are:

- SNR—intentionally excluded so one realization is rescaled along the SNR curve;
- an explicit global noise seed;
- explicit noise source ID—the RNG chooses it;
- replicate number—only one realization is supported.

Identical dataset files, window ID, noise type, NumPy behavior, ESC-50 inventory, and software path
should regenerate the same sample values. The noise manifest records relevant software versions,
corpus hashes, per-output hashes, seed, source, and crop. This supports detection of drift even if a
future library version changes byte-level regeneration.

## 24. Mixture manifest

The corruption is defined centrally by:

```text
work/windows_noisy/noise_manifest.json     one completion/build/protocol record
work/windows_noisy/noise_provenance.csv    one row per noisy WAV
```

The JSON is written last and is the completion marker
([`noise_sweep.py` L560–605](../src/instrument_robustness/noise_sweep.py#L560-L605)). It records
protocol version, complete state, dataset identity, SNRs/types, test/file counts, waveform format,
seed scheme, ESC-50 corpus provenance, provenance hash, and software versions.

Actual CSV fields are:

```text
window_id, window_path, clean_sha256, noise_type, snr_db, seed,
noise_source, noise_source_sha256, noise_source_sr,
crop_start_resampled_sample, alpha, signal_power, unscaled_noise_power,
realized_snr_db, peak, output_path, output_sha256
```

Comparison with a more expansive proposed schema:

| Desired concept | Actual representation | Status |
|---|---|---|
| `mixture_id` | tuple `(window_id, noise_type, snr_db)` / unique output path | implicit |
| `clean_window_id`, `clean_file` | `window_id`, `window_path` | present |
| `clean_source_id`, instrument, clean split | join to `windows.csv` | not copied; split is always test |
| sample rate / number samples | JSON `waveform_format` | present at build level |
| noise dataset/category/source ID/file | `noise_type`, `noise_source`, source hash; corpus in JSON | partial |
| noise split | none | absent |
| noise start/end | resampled start; end implicit | partial |
| requested/achieved SNR | `snr_db`, `realized_snr_db` | present |
| seed | `seed` | present |
| replicate | none; exactly one | absent |
| active fraction | none | absent |
| noise gain | `alpha` | present |
| peak scale | none because no scaling | absent |
| output path/hash | `output_path`, `output_sha256` | present |

`validate_noise_manifest` fails on a wrong dataset/protocol, wrong count/grid, duplicate output,
out-of-tolerance SNR, changed realization fields across SNRs, missing files, or optional WAV hash
mismatch ([`noise_sweep.py` L608–739](../src/instrument_robustness/noise_sweep.py#L608-L739)).

The manifest—not a random model loader—defines corruption. This is what keeps predictions paired.

## 25. Dynamic generation versus materialized WAV files

### Dynamic generation

A loader could recreate a waveform deterministically from manifest metadata. This saves disk and is
convenient during development, but every model loader would have to share exactly the same
implementation and version. Code or dependency changes could silently change samples.

### Offline materialization

Saving each float32 noisy WAV once makes it easy to hash, audit, listen to, and share across models.
It uses substantial storage: the repository estimates about 5.2 GB for 1,310 test windows and 15
conditions ([`NOISE_PLAN.md` L23–34](../NOISE_PLAN.md#L23-L34)).

> **Verified implementation:** The official path is offline materialization under
> `$RISE_DATA_ROOT/work/windows_noisy/`. `--validate` dynamically creates only a few checks and
> writes listenable preview WAVs; `--generate` materializes the complete benchmark. Evaluators read
> the completed files and refuse a missing or partial manifest.

## 26. Model-specific noisy-input generation

```text
one materialized noisy 22.05 kHz / 3 s / float32 WAV
    |
    +-- svm_vector -> saved train mean/std -> 88-D -> frozen SVC
    |
    +-- logmel -> saved train per-bin mean/std -> (128,130,1) -> future CNN adapter
    |
    +-- same logmel transposed -> (130,128) -> future CRNN adapter
    |
    +-- 16 kHz + ASTFeatureExtractor -> future AST adapter
    |
    +-- 24 kHz + pinned MERT -> 13x768 -> frozen final probe
    |
    `-- 32 kHz + CNN14 internal log-mel -> frozen PANNs classifier
```

**Current adapters:** SVM, MERT, and PANNs.  
**Missing adapters:** CNN, CRNN, and AST.

Generating independent noise inside each model loader would change source file, crop, or random
samples across models. The result would no longer be a paired comparison, and paired cluster
statistics would be invalid. Central materialization also ensures each model starts from the same
waveform before applying its own representation.

## 27. Evaluation protocol

The common noise evaluator writes, per condition:

- one prediction CSV with window/source/pitch-group IDs, true/predicted label, correctness, and
  optional class scores;
- a JSON with accuracy, fixed-label macro-F1, per-class classification report, and confusion
  matrix;
- a tidy sweep summary.

See [`noise_eval_common.py` L242–356](../src/instrument_robustness/noise_eval_common.py#L242-L356).
The fixed primary metric is:

\[
F_{1,\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}
\frac{2\,\mathrm{precision}_k\,\mathrm{recall}_k}
{\mathrm{precision}_k+\mathrm{recall}_k},
\qquad K=12,
\]

with zero division set to zero. Accuracy is also saved. Per-class precision/recall/F1 and confusion
matrices are implemented. AST clean evaluation additionally writes family-level performance;
the shared noise evaluator does not aggregate by family.

The implemented degradation quantities are:

\[
\Delta F_1(s,c)=F_{1,\mathrm{clean}}-F_{1,c,s},
\qquad
R_{F_1}(s,c)=\frac{F_{1,c,s}}{F_{1,\mathrm{clean}}},
\]

where \(c\) is noise category and \(s\) is SNR
([`noise_eval_common.py` L351–356](../src/instrument_robustness/noise_eval_common.py#L351-L356)).
A positive \(\Delta F_1\) means performance worsened.

Current official clean macro-F1 values, included here only to identify the parity references, are:

| Model | Test examples | Macro-F1 | Status |
|---|---:|---:|---|
| SVM | 1,310 | 0.982869 | current fingerprinted summary |
| MERT | 1,310 | 0.922275 | current fingerprinted summary |
| AST | 1,310 | 0.986577 | current fingerprinted metrics |

No current PANNs, CNN, or CRNN clean result exists locally.

For uncertainty, `noise_stats.cluster_bootstrap` resamples entire pitch groups by default, keeps all
12 labels fixed in every macro-F1 calculation, uses 2,000 bootstrap replicates by default, and
reports a percentile 95% interval
([`noise_stats.py` L79–121](../src/instrument_robustness/noise_stats.py#L79-L121)). Pairing requires
identical window/source/pitch/truth columns. The primary exact test is a **cluster sign test**, not
McNemar. Ordinary exact window-level McNemar is available only as a correlation-ignoring sensitivity
analysis ([`noise_stats.py` L134–212](../src/instrument_robustness/noise_stats.py#L134-L212)).

> **Unresolved:** The noise design has one deterministic realization, no replicate axis, and no
> repeated model-training seeds in the common output contract. Cluster bootstrap quantifies
> sampling uncertainty across pitch groups, not noise-realization or training-seed uncertainty.

## 28. Data leakage and validity checklist

| Risk | Repository status | Evidence / limitation |
|---|---|---|
| Source recordings crossing clean splits | **PREVENTS IT** | one source row receives one group assignment |
| Windows from the same source crossing splits | **PREVENTS IT** | split inherited; Step-4 runtime assertion |
| Normalization fit on validation/test | **PREVENTS IT** | Step 6 selects only `split=="train"` |
| Feature-standardization leakage | **PREVENTS IT** | saved train stats reused; fingerprinted; SVM no second scaler |
| Noise recordings crossing noise splits | **DOES NOT YET ADDRESS IT** | no external-noise split/catalog |
| Different models receiving different random mixtures | **PREVENTS IT** | one centrally materialized, hashed noisy set |
| Test data used for model selection | **UNCLEAR** | SVM/MERT sealed; AST/PANNs use validation but access test inputs earlier |
| Noise category imbalance | **UNCLEAR** | equal 800-file top-level pools and full factorial mixtures; subcategories not audited/balanced |
| Pitch imbalance | **DOES NOT YET ADDRESS IT** | pitch groups are isolated, not balanced; instrument ranges differ |
| Articulation imbalance | **UNCLEAR** | one articulation/class mitigates technique count; normal vs arco-normal still family-linked |
| Dynamic imbalance | **DOES NOT YET ADDRESS IT** | retained and grouped with pitch, not stratified/balanced |
| Silence/content-fraction imbalance | **DOES NOT YET ADDRESS IT** | tiling removes zero padding; content duration still differs by class; no activity mask |
| Stale features generated from old windows | **PREVENTS IT** | config fingerprints and CSV/NPZ checks; model summaries hash inputs |
| Old label mappings | **PREVENTS IT** | label order embedded and checked; old artifacts moved to legacy |
| Duplicate windows | **TESTS FOR IT** | noise builder rejects duplicate stems; current metadata audit found no duplicate paths |
| Noise files containing target instruments | **DOES NOT YET ADDRESS IT** | target ranges selected numerically; no content screening |
| Competing music changing the task to multi-label | **DOES NOT YET ADDRESS IT** | MUSAN/music not used, but no general interference-content validator |

Additional validity observations:

- The configuration fingerprint covers labels, articulation policy, split, windowing, waveform
  target, and feature parameters, but not the Git commit or every source/WAV byte.
- `manifest.csv` and `windows.csv` hashes make the noise dataset identity build-specific.
- The strict filter deliberately removes all trill techniques, so the present classifier is not a
  general “all articulations” Philharmonia classifier.
- Window-level class counts remain imbalanced (520 trumpet versus 900 flute overall). Models use
  model-specific class weighting policies; the data pipeline does not rebalance windows.

## 29. Current dataset statistics

### Counts and split structure

**VERIFIED FROM METADATA** after validating every stage sidecar against current configuration:

| Instrument | Raw readable sources | Retained sources | Source train | Source val | Source test | Window train | Window val | Window test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bassoon | 720 | 648 | 454 | 97 | 97 | 552 | 97 | 97 |
| cello | 889 | 747 | 517 | 115 | 115 | 522 | 117 | 117 |
| clarinet | 846 | 770 | 547 | 112 | 111 | 673 | 112 | 111 |
| double-bass | 852 | 764 | 533 | 116 | 115 | 547 | 118 | 116 |
| flute | 878 | 781 | 548 | 116 | 117 | 630 | 143 | 127 |
| french-horn | 652 | 546 | 380 | 83 | 83 | 462 | 88 | 88 |
| oboe | 596 | 539 | 379 | 81 | 79 | 396 | 86 | 79 |
| trombone | 831 | 759 | 531 | 114 | 114 | 598 | 120 | 139 |
| trumpet | 485 | 433 | 303 | 65 | 65 | 370 | 76 | 74 |
| tuba | 972 | 831 | 583 | 124 | 124 | 628 | 126 | 124 |
| viola | 973 | 708 | 495 | 107 | 106 | 504 | 107 | 108 |
| violin | 1,502 | 852 | 594 | 129 | 129 | 605 | 129 | 130 |
| **Total** | **10,196** | **8,378** | **5,864** | **1,259** | **1,255** | **6,487** | **1,319** | **1,310** |

There are 10,197 physical MP3s including the one unreadable file. Total windows are 9,116.

### Pitches and source attributes

| Instrument | Unique notes / pitch groups | MIDI min–max |
|---|---:|---:|
| bassoon | 45 | 34–79 |
| cello | 49 | 36–84 |
| clarinet | 47 | 50–96 |
| double-bass | 44 | 24–67 |
| flute | 42 | 60–101 |
| french-horn | 44 | 34–77 |
| oboe | 37 | 58–94 |
| trombone | 49 | 40–88 |
| trumpet | 45 | 40–88 |
| tuba | 42 | 22–65 |
| viola | 51 | 48–98 |
| violin | 49 | 55–103 |

Total pitch groups are 544. A group contains 15.40 retained source files on average (median 16,
range 1–26).

Retained technique counts are `normal: 5,307` and `arco-normal: 3,071`; no trill survives. Retained
nominal-length counts are `025: 2,117`, `05: 2,141`, `1: 2,073`, `15: 1,656`, `long: 144`,
`very-long: 185`, and `phrase: 62`.

| Dynamic | Retained sources |
|---|---:|
| forte | 1,711 |
| fortissimo | 1,686 |
| piano | 1,497 |
| pianissimo | 1,346 |
| mezzo-forte | 1,120 |
| mezzo-piano | 840 |
| cresc-decresc | 90 |
| molto-pianissimo | 71 |
| crescendo | 13 |
| decrescendo | 4 |

### Window/content and normalization statistics

- Windows per source: mean 1.088, median 1, 99th percentile 4, maximum 26.
- Pre-tile `content_s`: mean 1.1686, median 0.9665, range 0.0784–3.0 s.
- 8,341 windows are tiled and 775 are full-length before tiling.
- Mean `content_s / 3` is 0.3895, but this is **not active fraction**; tiling fills the output.
- Post-normalization RMS: median 0.1000; 15 windows are peak-guarded more than 0.001 below target.

| Instrument | Windows | Median `content_s` | Tiled windows | Tiled % |
|---|---:|---:|---:|---:|
| bassoon | 746 | 1.207 | 646 | 86.60 |
| cello | 756 | 0.940 | 746 | 98.68 |
| clarinet | 896 | 1.486 | 767 | 85.60 |
| double-bass | 781 | 1.071 | 763 | 97.70 |
| flute | 900 | 1.045 | 774 | 86.00 |
| french-horn | 638 | 1.393 | 540 | 84.64 |
| oboe | 561 | 0.998 | 537 | 95.72 |
| trombone | 857 | 0.720 | 750 | 87.51 |
| trumpet | 520 | 1.363 | 432 | 83.08 |
| tuba | 878 | 0.580 | 829 | 94.42 |
| viola | 719 | 1.019 | 706 | 98.19 |
| violin | 864 | 0.863 | 851 | 98.50 |

> **Unresolved active-audio statistics:** No active mask or active duration is generated. The trim
> and `content_s` values above are the closest available proxies and must not be relabeled as active
> audio.

### Array shapes

| Representation | Train | Validation | Test | Dtype |
|---|---|---|---|---|
| SVM `X` | `(6487,88)` | `(1319,88)` | `(1310,88)` | float32 |
| SVM/CNN `y` | `(6487,)` | `(1319,)` | `(1310,)` | int64 |
| CNN `X` | `(6487,128,130,1)` | `(1319,128,130,1)` | `(1310,128,130,1)` | float32 |
| CRNN view | `(6487,130,128)` | `(1319,130,128)` | `(1310,130,128)` | float32 |
| MERT cached `X` | expected `(N,13,768)` | expected `(N,13,768)` | finalizer-only | absent locally |

## 30. Tests and validation

The complete safe test run on this audit checkout reported **41 passed, 4 skipped**. The skipped
tests/modules require optional PyTorch/AST dependencies.

| Area | Existing test(s) | Guarantee |
|---|---|---|
| Trimming | none | no direct Step-2 guarantee |
| Activity detection | none; no activity implementation | no active-mask guarantee |
| Grouped splitting | `test_group_assignment_is_deterministic_and_leak_free`; `test_leak_verifier_rejects...` | deterministic synthetic assignment and a group leak raises |
| Window tiling | three `WindowRegressionTests` | exact repeat pattern, short source becomes 66,150 nonzero samples, tiny final tail drops |
| Manifest integrity | fingerprint tests and prep-data mocked test | wrong stage/changed CSV rejected; canonical manifest gets sidecar |
| Feature shapes | MERT embedding-shape test; AST wrong-sample test; SVM loader tests | model-specific input validation; not a direct full Step-7 numerical regression |
| Train-only SVM preprocessing | `test_loader_does_not_standardize_features_again`; SVM-noise statistics test | no second scaler and saved train stats reused |
| Requested/achieved SNR | `test_power_snr_is_recovered` | whole-window mixer recovers each configured SNR |
| Deterministic seeds | `test_seed_is_build_scoped_and_snr_independent` | same build/window/type stable; build changes seed; SNR omitted |
| Clipping/headroom | `test_float_window_preserves_headroom...` | float WAV reader preserves values above 1 and rejects wrong length |
| Noise manifest | `test_manifest_validation_is_fail_closed`; dataset hash test | stale dataset/protocol fails; actual windows hash affects identity |
| Shared pairing/parity | runner, parity, pitch-group, pairing tests | all 16 conditions, official clean count/F1 gate, authoritative clusters |
| Fixed-label statistics | macro-F1 and cluster-statistics tests | absent labels still count; bootstrap/sign output deterministic |
| Noise split isolation | none | external noise splits do not exist |
| Waveform regeneration | none | no byte-for-byte redraw/materialize regression |

Relevant locations are
[`tests/test_preprocessing.py`](../tests/test_preprocessing.py),
[`tests/test_noise.py`](../tests/test_noise.py),
[`tests/test_svm.py`](../tests/test_svm.py),
[`tests/test_mert.py`](../tests/test_mert.py), and
[`tests/test_ast.py`](../tests/test_ast.py).

Important missing tests:

1. Step-2 trim boundaries, default frame/hop semantics, fallback, all-zero input, and attack/decay
   retention.
2. A full generated-manifest leak audit in the test suite, not only synthetic group frames.
3. Exact headers/lengths for every physical clean window as a routine preflight.
4. Numeric regression tests for the 88-feature order and `(128,130)` log-mel calculation.
5. Verification that Step 6 reads no validation/test rows and that feature statistics match train.
6. Deterministic ESC-50 file/crop selection and byte-identical regeneration.
7. DC-offset behavior and empirical Gaussian mean/power tolerances.
8. External-noise source split isolation and ESC-50 category/fold provenance.
9. A generated-file test that hashes/reloads output and checks every provenance field.
10. No-active-region test exists because no active-region algorithm exists.
11. No CNN, CRNN, or AST noise-adapter parity test exists because those adapters are absent.

## 31. Worked example

This repository-grounded clean example is the first current `windows.csv` row:

```text
source:
  bassoon/A2/bassoon_A2_025_forte_normal.mp3

after trim:
  0.3135 s

window:
  work/windows/bassoon/A2/bassoon_A2_025_forte_normal_w000.wav
  22,050 Hz × 3.0 s = 66,150 samples
  the 0.3135 s segment is repeated until 3.0 s

normalization metadata:
  pre_norm_rms  = 0.05489
  post_norm_rms = 0.10000
```

For an actual 10 dB **white-noise** condition:

1. Load the exact Step-5 PCM16 clean window.
2. Derive the seed from dataset fingerprint, `bassoon_A2_025_forte_normal_w000`, and `white`.
3. Draw 66,150 standard-normal float32 samples.
4. Measure whole-window clean and noise power.
5. Calculate \(\alpha\) for 10 dB.
6. Add \(y=x+\alpha n\).
7. Do not apply peak scaling or clipping.
8. Save float32 WAV.
9. Reload it and verify achieved whole-window SNR differs from 10 dB by less than 0.1 dB.
10. Save seed, powers, gain, peak, clean/output hashes, and output path.
11. Recompute the model's representation from this saved noisy waveform.

A small numerical illustration close to a non-peak-guarded normalized window is:

\[
P_x=0.1^2=0.01,\quad P_n=1,\quad s=10,
\]

\[
\alpha=\sqrt{\frac{0.01}{1\cdot10^{10/10}}}
=\sqrt{0.001}=0.0316228.
\]

The added noise power is \(\alpha^2P_n=0.001\), so

\[
10\log_{10}(0.01/0.001)=10\ \mathrm{dB}.
\]

This is illustrative because a finite Gaussian draw has power near, not exactly, 1, and the code
uses the measured powers. It also uses the actual clean power rather than assuming 0.01.

> **Not the current algorithm:** There is no active mask retrieval and no DEMAND cafeteria test
> recording. A worked example claiming those steps would describe a proposed experiment, not this
> repository.

## 32. From implementation to paper Methods section

| Proposed paper subsection | Facts to draw from this document |
|---|---|
| A. Dataset and Instrument Classes | Sections 5, 6, and 29: source, filtering, labels, counts, pitch/dynamic/technique |
| B. Audio Preprocessing | Sections 8–11: decode, resample, relative trim, tiling, RMS normalization |
| C. Source-Level Data Partitioning | Sections 7, 12, and 28: pitch-group assignment, ratios, leakage safeguards |
| D. Acoustic Representations | Sections 13–15: 88-D features, log-mel, AST/MERT/PANNs paths |
| E. Classification Models | Section 15 and model-specific repository files; include only completed models |
| F. Noise Conditions | Sections 16–20: categories, sources, grid, selection |
| G. Signal-to-Noise Ratio Mixing | Sections 21–23: whole-window power, gain, float storage, seeds |
| H. Evaluation Protocol | Sections 24–27: shared files, parity, metrics, paired statistics |
| I. Reproducibility and Leakage Controls | Sections 12, 23, 28, and 30: fingerprints, hashes, tests, gaps |

When converting this audit into prose, retain the distinction between source recordings and derived
windows, and between “code exists” and “experiment was run.” Do not describe active-region SNR,
external noise splits, or missing model adapters in past tense.

## 33. Methods-ready factual summary

**VERIFIED:**

- Twelve labels were used in the fixed alphabetical order shown in Section 6.
- Readable source MP3s were mono at 44.1 kHz and were resampled to mono 22.05 kHz PCM16 WAV.
- One configured articulation per instrument was retained; all trill-technique recordings were
  excluded by this policy.
- Leading/trailing regions were trimmed with `librosa.effects.trim(top_db=30)`, using a threshold
  relative to each source's maximum frame RMS.
- Splits were assigned at the `(instrument, note)` pitch-group level with seed 0 and target source
  fractions 70/15/15.
- All windows inherited the source split.
- Windows were 3.0 seconds (66,150 samples), non-overlapping, and onset-aligned at 3.0-second hops.
- Short/eligible final windows were tiled; Step 4 did not zero-pad them.
- Each window was RMS-normalized toward 0.1 with a 0.99 peak guard before feature extraction.
- SVM inputs contained 88 handcrafted temporal-summary features.
- SVM feature standardization and per-mel-bin log-mel standardization were fitted on train only and
  reused for validation/test.
- Log-mel inputs used 2,048-point FFTs, 512-sample hops, 128 mel bins, 130 frames, and 0–11,025 Hz.
- AST, MERT, and PANNs began with the same Step-5 waveform and resampled to 16, 24, and 32 kHz,
  respectively.
- The implemented noise grid was white/natural/mechanical at 20, 10, 5, 0, and -5 dB plus clean.
- Only clean test windows were corrupted; clean train/validation data and fitted models remained
  unchanged.
- SNR was calculated from mean power over the entire fixed window.
- A noise realization was deterministic per dataset build/window/category and was rescaled across
  SNRs.
- No post-mix normalization or hard clipping was used; noisy audio was stored as float32 WAV.
- The same materialized noisy WAV was intended for every model.
- SVM, MERT, and PANNs noise adapters existed; AST, CNN, and CRNN adapters did not.

## 34. Unresolved questions before paper writing

### Code ambiguity or inconsistency

1. Should `featurelib.load_window` continue silently zero-padding malformed files while AST/noise
   loaders reject them?
2. Should Step 1 enforce a between-class spectral-ceiling criterion rather than merely print it?
3. Should AST/PANNs adopt the same sealed-test access guard as SVM/MERT?
4. Which location is authoritative for AST outputs: data-root `models/ast` or repository
   `artifacts/ast`?
5. Stale 9-way text in AST/PANNs plan files must not be used in the paper.
6. The README references a missing `pipeline_report.txt`.

### Missing metadata

1. No active interval, mask, duration, or fraction is stored.
2. Canonical source metadata omit channels, bitrate, exact nominal length, and file hash.
3. Clean `windows.csv` fingerprints do not hash physical WAV content.
4. SVM/CNN NPZ files omit `window_path`.
5. Noise provenance omits original ESC-50 target/category/fold, noise split, explicit mixture ID,
   replicate, active fraction, and peak scale.

### Planned or unimplemented noise behavior

1. Active-region SNR is not implemented; decide whether whole-window SNR remains the final protocol.
2. External noise train/validation/test splitting is absent.
3. DC-offset removal is absent.
4. There are no CNN, CRNN, or AST noise adapters.
5. There is no replicate axis or noise-realization uncertainty analysis.
6. DEMAND, speech, music, reverberation, and competing-instrument conditions are not implemented.

### Teammate confirmation required

1. Confirm that the three current categories and five SNRs are the frozen paper protocol.
2. Confirm whether “natural” and “mechanical” are acceptable paper names for the selected ESC-50
   target ranges.
3. Confirm whether strict single-articulation filtering matches the intended scientific population.
4. Confirm whether the paper compares only completed models or waits for CNN/CRNN/PANNs and all
   noise adapters.
5. Confirm the desired primary uncertainty unit: pitch group or source recording.
6. Confirm corpus licensing/citation language from authoritative dataset sources.

### Experiments not yet evidenced locally

1. No completed materialized noise manifest is present.
2. No current noise-evaluation outputs are present.
3. PANNs has no local clean checkpoint/result.
4. CNN and CRNN have features but no current model result.
5. Model-to-model paired confidence intervals/tests have not been run.
6. Sensitivity to training seeds and multiple noise realizations has not been measured in the
   current shared protocol.

## 35. Glossary

| Term | Simple definition |
|---|---|
| waveform | Ordered audio sample amplitudes over time. |
| sample rate | Number of samples per second; 22,050 Hz means 22,050 values each second. |
| frame | A short, usually overlapping analysis block used to calculate a local feature. |
| window | Here, one complete 3.0-second model example; not the same as an FFT window function. |
| hop length | Samples between successive analysis frames or successive data windows. |
| FFT | Algorithm that represents a time-domain frame by its frequency components. |
| spectrogram | Time-by-frequency representation made from successive FFT frames. |
| mel scale | Frequency mapping that compresses high-frequency resolution in a perceptually motivated way. |
| MFCC | Compact coefficients describing the shape of a log-mel spectral envelope. |
| RMS | Root mean square; a power-related measure of waveform amplitude. |
| active region | Samples judged to contain the target sound. This repository does not store one. |
| SNR | Signal-to-noise power ratio, usually expressed in decibels. |
| decibel | Logarithmic ratio unit; for power, \(10\log_{10}(P_1/P_2)\). |
| noise gain | Multiplier \(\alpha\) applied to unscaled noise to reach a requested SNR. |
| macro-F1 | Unweighted average of per-class F1, giving each instrument equal importance. |
| data leakage | Information from validation/test—or a near-duplicate—improperly influencing training or selection. |
| source-level split | Assignment made before deriving multiple windows from one recording. |
| pitch-group split | Stronger grouping here: all recordings with one instrument and note stay together. |
| augmentation | Training-time transformation that creates altered examples; not used in the first robustness experiment. |
| deterministic seed | Stable number that makes a random draw reproducible. |
| manifest | Table/JSON that defines examples, identities, paths, parameters, and provenance. |
| materialization | Saving a derived waveform or representation to disk instead of recreating it on demand. |
| fingerprint | Stored configuration and/or file hash used to detect incompatible or stale artifacts. |
| tiling | Repeating a short recorded segment until it fills the required window length. |

## 36. Reproduction commands

Run commands from the repository root with the intended environment activated. `RISE_DATA_ROOT`
defaults to `all-samples`; set it explicitly on shared systems.

### Build clean data

```bash
python -m instrument_robustness.prep_data
python -m instrument_robustness.step0_filter
python -m instrument_robustness.step1_resample
python -m instrument_robustness.step2_trim
python -m instrument_robustness.step3_split
python -m instrument_robustness.step4_window
python -m instrument_robustness.step5_normalize
python -m instrument_robustness.step6_stats
python -m instrument_robustness.step7_featurize
```

> **Warning:** These commands download data and/or overwrite derived manifests, WAVs, statistics,
> and feature arrays. They are the authoritative full build, not a harmless validation command.
> Do not run them merely to inspect an existing shared dataset.

### View current manifests without modifying them

```bash
head -n 3 all-samples/manifest.csv
head -n 3 all-samples/pipeline/splits.csv
head -n 3 all-samples/pipeline/windows.csv
```

Validate the current CSV/NPZ fingerprint chain read-only:

```bash
python - <<'PY'
import numpy as np
from instrument_robustness.config import (
    MANIFEST_FINGERPRINT, MANIFEST_IN, MANIFEST_LABELED, MANIFEST_PRODUCER_STAGES,
    MANIFEST_RESAMPLED, MANIFEST_TRIMMED, SPLITS_CSV, STATS_NPZ, WINDOWS_CSV,
    assert_artifact_fingerprint, assert_serialized_fingerprint,
)
for path, stage, sidecar in [
    (MANIFEST_IN, MANIFEST_PRODUCER_STAGES, MANIFEST_FINGERPRINT),
    (MANIFEST_LABELED, "step0_filter", None),
    (MANIFEST_RESAMPLED, "step1_resample", None),
    (MANIFEST_TRIMMED, "step2_trim", None),
    (SPLITS_CSV, "step3_split", None),
    (WINDOWS_CSV, "step5_normalize", None),
]:
    assert_artifact_fingerprint(path, stage, fingerprint_path=sidecar)
with np.load(STATS_NPZ, allow_pickle=True) as data:
    assert_serialized_fingerprint(data["config_fingerprint"], str(STATS_NPZ))
print("clean provenance chain is current")
PY
```

### Check feature-array shapes

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
for family in ("svm", "cnn"):
    for split in ("train", "val", "test"):
        path = Path("all-samples/features") / family / f"{split}.npz"
        with np.load(path, allow_pickle=True) as data:
            print(path, "X", data["X"].shape, data["X"].dtype,
                  "y", data["y"].shape, data["y"].dtype)
PY
```

### Run focused tests

```bash
python -m unittest discover -s tests -p 'test_preprocessing.py' -v
python -m unittest discover -s tests -p 'test_noise.py' -v
python -m unittest discover -s tests -v
```

Optional PyTorch-dependent AST/MERT tests require the relevant extras.

### Noise preview, generation, and achieved-SNR validation

```bash
export RISE_NOISE_ROOT=/path/to/noise_sources

python -m instrument_robustness.noise_sweep --validate
python -m instrument_robustness.noise_sweep --generate
python -m instrument_robustness.noise_sweep --check-generated
python -m instrument_robustness.noise_sweep --check-generated --verify-audio-hashes
```

> **Warning:** Despite its name, `--validate` writes a small set of listenable preview WAVs under
> `work/windows_noisy/_validation_samples/`; it requires ESC-50 but does not build the full sweep.
> `--generate` is expensive, materializes about 5.2 GB for the current test set, and refuses to
> overwrite a completed or partial canonical sweep. `--check-generated` is read-only; the
> `--verify-audio-hashes` version is slower because it hashes every generated WAV.

Model evaluation commands, after a completed shared sweep, are:

```bash
python -m instrument_robustness.noise_eval_svm
python -m instrument_robustness.noise_eval_mert --device cuda
python -m instrument_robustness.noise_eval_panns
```

The PANNs command additionally requires a current clean PANNs model/result. No authoritative CNN,
CRNN, or AST noise-evaluation command exists yet.
