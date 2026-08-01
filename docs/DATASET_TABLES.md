# Dataset composition — tables and figures

**Build:** `75d81b2` ("Repair dataset and seal model evaluation workflows"), freeze state `frozen`.
**Scope:** dataset composition only. `artifacts/` was deleted in this commit, so **no model results
of any kind exist on this build.** Every accuracy, macro-F1, AUC and retention number quoted before
`75d81b2` describes the pre-repair 9,116-window dataset and must not appear in a draft.

Everything below was computed from `all-samples/pipeline/windows.csv` joined to
`all-samples/manifest.csv` on this build. Regenerate with `scripts/dataset_tables.py` (see
"Reproducing" at the end).

## Integrity gates passed on this build

| Check | Result |
|---|---|
| `windows.csv` artifact fingerprint | matches current `config_fingerprint()` |
| sources / windows | 8,374 / 8,374 — exactly one window per source |
| byte-identical file groups | 0 |
| byte-identical groups with conflicting labels | 0 (was 2 pre-repair; all 4 files removed) |
| pitch groups spanning more than one split | 0 of 544 |
| split sizes | train 5,861 · val 1,258 · test 1,255 |

The four files removed as label-contaminated are recorded in the config fingerprint under
`excluded_conflicting_label_paths`: `cello/Ds5/cello_Ds5_05_forte_arco-normal.mp3`,
`viola/G6/viola_G6_05_fortissimo_arco-normal.mp3`,
`french-horn/E2/french-horn_E2_1_fortissimo_normal.mp3`,
`oboe/E6/oboe_E6_15_mezzo-forte_normal.mp3`.

---

## Table 1 — Dataset composition by instrument

One row per recording; because `MAX_WINDOWS_PER_SOURCE = 1`, recordings and analysis windows are
in 1:1 correspondence, so these counts are simultaneously file counts and window counts.

| Instrument | Train | Val | Test | Total | Distinct notes | MIDI range |
|---|---:|---:|---:|---:|---:|---|
| bassoon | 454 | 97 | 97 | 648 | 45 | 34–79 |
| cello | 516 | 115 | 115 | 746 | 49 | 36–84 |
| clarinet | 547 | 112 | 111 | 770 | 47 | 50–96 |
| double-bass | 533 | 116 | 115 | 764 | 44 | 24–67 |
| flute | 548 | 116 | 117 | 781 | 42 | 60–101 |
| french-horn | 379 | 83 | 83 | 545 | 44 | 34–77 |
| oboe | 378 | 81 | 79 | 538 | 37 | 58–94 |
| trombone | 531 | 114 | 114 | 759 | 49 | 40–88 |
| trumpet | 303 | 65 | 65 | 433 | 45 | 40–88 |
| tuba | 583 | 124 | 124 | 831 | 42 | 22–65 |
| viola | 495 | 106 | 106 | 707 | 51 | 48–98 |
| violin | 594 | 129 | 129 | 852 | 49 | 55–103 |
| **Total** | **5,861** | **1,258** | **1,255** | **8,374** | 82 (union) | 22–103 |

Mean 697.8 ± 129.9 recordings per class; imbalance ratio **1.97:1** (violin 852 : trumpet 433).
Realised split proportions 69.99 / 15.02 / 14.99 %; per-class train share stays within
69.2–71.0 %, so the grouped split does not systematically starve any class.

**Figure 1** — `figures/fig1_class_balance.{png,pdf}`
*Recordings per instrument, stacked by split. The dataset is mildly imbalanced (1.97:1) and the
70/15/15 proportion holds within each class despite splitting on pitch groups rather than files.*

**Figure 2** — `figures/fig2_pitch_range.{png,pdf}`
*Pitch range covered per instrument in MIDI note numbers. Coverage follows the instruments'
physical registers, spanning MIDI 22 (tuba) to 103 (violin) with 82 distinct pitches overall;
ranges overlap heavily in the middle register, so pitch alone cannot separate the classes.*

---

## Table 2 — Effect of the articulation filter

Restricting each instrument to a single playing technique — `normal` for winds and brass,
`arco-normal` for strings — removes 1,822 of the 10,196 in-class recordings (17.87 %). Four of
those removals are the label-contaminated files above; the remaining 1,818 are articulation drops.

| | All articulations | After filter |
|---|---:|---:|
| Recordings | 10,196 | 8,374 |
| Distinct techniques | 47 | 2 (`normal` 5,305 · `arco-normal` 3,069) |
| Largest class | violin 1,502 | violin 852 |
| Smallest class | trumpet 485 | trumpet 433 |
| **Imbalance ratio** | **3.10:1** | **1.97:1** |

Removals are heavily concentrated in the strings, which carry most of the extended-technique
recordings: violin 650, viola 266, cello 143, tuba 141, french-horn 107, flute 97, double-bass 88,
clarinet 76, bassoon 72, trombone 72, oboe 58, trumpet 52.

**Justification for the paper.** The filter is what makes the label mean *instrument* rather than
*instrument-and-technique*. Pizzicato violin, snap-pizzicato, col legno battuto and tremolo share
almost no timbral structure with a bowed note, and the archive supplies them for strings far more
than for winds — so leaving them in both inflates the string classes (3.10:1 → 1.97:1 imbalance)
and lets a model separate classes on articulation, a cue that would not survive a change of
corpus. Holding technique fixed also makes the noise sweep interpretable: degradation can be
attributed to added noise rather than to a heterogeneous mix of excitation types. The cost is
external validity, and it should be stated: results describe sustained, conventionally-played
notes and do not license claims about pizzicato or muted playing.

**Figure 3** — `figures/fig3_articulation_filter.{png,pdf}`
*Per-instrument recording counts before and after the articulation filter. The strings shed the
most material because the archive supplies them with the widest range of extended techniques;
class imbalance falls from 3.10:1 to 1.97:1.*

---

## Table 3 — Window content before tiling

Windows are 3.0 s. Sources shorter than that are **tiled (looped)**, never zero-padded —
zero-padding was measured to collapse the noise sweep to majority-class prediction at every SNR,
because `power_to_db(ref=np.max)` clamps digital silence to the floor and injected noise then
fills a region the model never saw in training. `content_s` below is the amount of *distinct*
source audio in each window, after `librosa.effects.trim(top_db=30)`.

| Distinct content | Windows | Share |
|---|---:|---:|
| < 0.5 s | 1,774 | 21.18 % |
| < 1.0 s | 4,828 | 57.65 % |
| < 1.5 s | 6,695 | 79.95 % |
| < 2.0 s | 7,642 | 91.26 % |
| < 3.0 s | 8,148 | 97.30 % |
| = 3.0 s (untiled) | 226 | 2.70 % |

Median 0.906 s, mean 1.036 s, range 0.078–3.000 s. Per-class medians run from 0.575 s (tuba) to
1.305 s (clarinet), so the amount of repetition is not uniform across classes — worth reporting,
since a tiled window has a periodicity at the source duration that a model could in principle
exploit.

`MIN_WINDOW_CONTENT_S = 0.5` does not contradict the 21 % of windows below 0.5 s: that floor drops
only *trailing* windows, and with `MAX_WINDOWS_PER_SOURCE = 1` every source contributes exactly
its first window, so the floor never binds.

**Figure 4** — `figures/fig4_window_content.{png,pdf}`
*Distribution of distinct source audio per 3.0 s window (left) and per-instrument medians (right).
97.3 % of windows contain less than one full window of unique audio and are completed by tiling;
the spike at 3.0 s is the 226 sources long enough to fill a window outright.*

---

## Table 4 — Dynamics, pitch groups, and loudness

**Dynamic markings** (as filed in the source archive):

| forte | fortissimo | piano | pianissimo | mezzo-forte | mezzo-piano | cresc-decresc | molto-pianissimo | crescendo | decrescendo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,710 | 1,684 | 1,497 | 1,346 | 1,119 | 840 | 90 | 71 | 13 | 4 |

**Pitch groups.** The split unit is the `(instrument, note)` pair, assigned before windowing:
544 groups, mean 15.39 recordings per group (median 16, range 1–26), distributed 355 train / 95 val
/ 94 test with **zero groups spanning a split**. This matters because the same note at different
dynamics is a near-duplicate; splitting on files rather than pitch groups leaks near-copies of test
notes into training and inflates the score.

**Loudness.** Before normalisation, median RMS varies more than tenfold across instruments —
flute 0.0111, viola 0.0170, violin 0.0231, trombone 0.0246, french-horn 0.0285, oboe 0.0344,
cello 0.0347, clarinet 0.0471, trumpet 0.0479, bassoon 0.0491, double-bass 0.0864, tuba 0.1102.
Step 5 rescales every window to RMS 0.1 with a 0.99 peak guard; 8,359 windows land exactly at 0.1
and 15 (12 trombone, 3 trumpet) are attenuated by the guard, the lowest to 0.0775.

**Figure 5** — `figures/fig5_loudness_normalisation.{png,pdf}`
*Per-window RMS before (left) and after (right) loudness normalisation. Median loudness varies
about tenfold across instruments before Step 5; afterwards all but 15 windows sit exactly at
RMS 0.1, the exceptions being windows whose peak would otherwise clip the 0.99 guard.*

---

## Reproducing

Every number and figure in this file is produced by one command, which verifies the `windows.csv`
artifact fingerprint before reporting anything and refuses to run on a stale build:

```
python scripts/dataset_tables.py
```

If a number here disagrees with that command, the command wins and this file is stale. If the
manifest changes at all, the dataset fingerprint changes and everything here must be recomputed.

## What is still missing

No results exist on this build. To report anything about model behaviour, the following must be
re-run in order against the frozen dataset: feature extraction → SVM / CNN / CRNN / MERT / PANNs /
AST training and validation selection → `finalize_*` contract summaries → noise corpus generation
(the dataset fingerprint changed, so the entire noise corpus is invalidated) → `noise_eval_*` with
the clean-parity gate. Until then, only the dataset composition above is reportable.
