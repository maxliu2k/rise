# Audit checklist — 22 findings

**Verified against:** the local tree based on `main` at `7b64888`, re-checked 2026-07-30.
**How to read this:** every status below was re-derived from the current tree, not carried over from
notes. Where an item is only partly done, the remaining work is stated explicitly.

| Status | Meaning |
|---|---|
| ✅ **FIXED** | Implemented and verified in the current tree. |
| 🟡 **PARTIAL** | Some of it landed; the rest is named. |
| ⬜ **OPEN** | Not started. |
| 📝 **WONTFIX / WRITE-UP** | Cannot be fixed in code; must be disclosed in the paper. |

**Score: 9 fully fixed · 9 partial · 2 open · 2 write-up-only.**

Closed outright: **1, 5, 6, 8, 16, 17, 18, 19, 22**.

Still fully open: **9** (AST/PANNs sealed test) and **15** (external evaluation). Both need
something code alone cannot supply — a training run, or an external corpus.

Do **not** run `noise_sweep --generate` yet. The MERT validation-only SNR pilot must run first and
`N_REPLICATES` must be frozen; changing either choice after generation would invalidate the sweep.

---

## Must resolve before the official noise experiment

### ✅ 1. SNR measured across all frequencies
- **Fixed.** `noise_metrics.py` adds `band_snr_db` over `config.INSTRUMENT_BAND_HZ` (25–8000 Hz),
  a full per-octave profile, and a worst-occupied-octave summary. Recorded per mixture as
  `snr_band_db`, `snr_worst_octave_db`, `snr_worst_octave_center_hz`, `snr_octave_db`.
- **The lower edge now includes every dataset fundamental.** The previous 50 Hz edge excluded the
  lowest tuba note (MIDI 22, about 29 Hz); 25 Hz includes it while still excluding DC.
- **The headline SNR is unchanged** — this is reported *alongside* it, not instead of it.
- **Regression-demonstrated:** at the same nominal 0 dB, low rumble and HF-only noise both produce
  band SNR at least 15 dB cleaner than white noise. This is exactly the misreading this item was
  raised about, now detected automatically.
- **One subtlety, handled:** the worst-octave summary filters to bands holding ≥1% of the clean
  signal's power (`MIN_CLEAN_SHARE`). Without it, it reported ~−150 dB from a band the instrument
  does not occupy — true, meaningless, and identical for every noise type. Regression-tested.
- **Band powers are Parseval-exact**, so band SNRs are commensurable with the whole-signal number
  (tested to 10 decimal places).

### 🟡 2. SNR grid probably too harsh
- **Implemented, not yet frozen.** Grid moved to `config.SNRS` (owned there, imported by `noise_sweep`), and
  `snr_pilot.py` added — validation-only, writes no audio, reuses the real `draw_noise`/`mix_at_snr`.
- **Measured:** SVM/white, 240 validation windows, train-only checkpoint. Clean macro-F1 0.9650;
  chance 0.083. Every level of the old `[20, 10, 5, 0, -5]` was at or below chance:

  ```text
    SNR   macro-F1  retention        SNR   macro-F1  retention
     60     0.9515      0.986         20     0.0931      0.096  <- old grid, floor
     50     0.8497      0.881         10     0.0366      0.038  <- old grid, floor
     40     0.5376      0.557          0     0.0132      0.014  <- old grid, floor
     30     0.2599      0.269
  ```

- **New grid:** `[60, 50, 40, 30, 20, 10, 0]` → 22 conditions, 26,355 files, ~6.5 GiB. Spans both
  regimes deliberately rather than the SVM-optimal 55–30 band, so pretrained models are not all at
  ceiling. Rationale recorded in `config.py` and `docs/NOISE_PLAN.md` §2.
- **Still to do:** run the validation-only MERT pilot on SCC. The current MERT clean run is complete
  and the pilot now accepts its hash-verified `best_probe.pt`; no retraining or test access is
  needed. Expect to revisit the low end.

### 🟡 3. One noise sample per clip
- **The replicate axis is fixed in code.** The seed is now
  `sha256(dataset_fingerprint|window_id|noise_type|replicate)[:4]`, SNR still excluded. `config
  .N_REPLICATES` drives generation, output paths carry `r{k}` unconditionally, the manifest records
  `n_replicates`, `validate_noise_manifest` checks the full grid including replicates, and
  `noise_conditions()` emits one condition per replicate.
- **The two seed decisions are deliberate opposites:** SNR excluded so one realization is merely
  rescaled along the curve; replicate included because that is the axis where a *different* draw is
  wanted.
- **Across-replicate reporting is now defined.** `summarise_sweep` averages macro-F1 across
  replicates at each SNR and records mean/std/min/max. `paired_replicate_differences` compares two
  models only after matching `(noise_type, snr_db, replicate)` and refuses incomplete or unmatched
  grids. `noise_stats` remains the within-condition, window-clustered inference tool.
- **Verified end to end** with `N_REPLICATES = 2`: real `generate()` → real
  `validate_noise_manifest(verify_audio_hashes=True)`, all 12 windows drew different clips per
  replicate, the realization stayed constant across SNRs within a replicate, and requested SNR was
  achieved to 6.7e-7 dB.
- **Still set to 1.** Cost is exactly linear. Freeze the run decision—currently expected to be
  3—before generation. Until then, this item is partial.

### 🟡 4. Natural/mechanical categories too broad
- **Fixed — the provenance half.** `noise_provenance.csv` now carries `noise_target`,
  `noise_category`, `noise_fold` per mixture; `noise_manifest.json` records `target_ranges` and a
  full `category_composition` (which of the 20 classes, how many clips, per project category).
  `NOISE_MANIFEST_VERSION` → 5 (bumped again for active-instrument diagnostics). `validate_noise_manifest` requires the new columns and checks they
  stay constant across SNRs within one realization. White noise carries them as explicit `None`.
  Tests: `Esc50ProvenanceTests` (4 tests).
- **Why this was urgent:** unrecoverable later. Regenerating the sweep is the only way to add it
  after the fact.
- **Still open:** the two project categories still each collapse 20 ESC-50 classes. Nothing enforces
  per-subcategory balance, and no content screening rejects a clip containing a target-like
  instrument (targets 0–19 include tonal bird/insect sounds). Post-hoc analysis is now *possible*
  because the label is recorded; it has not been *done*.

### ✅ 5. No short-time or active-audio SNR check
- **Fixed.** The short-time/noise-activity measurements are recorded per mixture:
  `noise_active_fraction`, `snr_segmental_min_db`,
  `snr_segmental_{p05,p50,p95}_db`, `snr_segmental_std_db`, `snr_segmental_active_frames`.
- **Demonstrated:** a synthetic 30 ms slam mixed to nominal 0 dB reports active fraction **0.04**
  (vs 1.00 for ambience), 5 active frames of 126, and a worst frame at **−33.6 dB**. That is the
  door-slam failure mode, detectable from the CSV alone.
- **A flaw caught by testing:** measured over *all* frames, the slam's p05 came out at **+161 dB** —
  the percentiles were describing the 99% of frames containing no noise. Percentiles are now taken
  over active-noise frames, with `min` kept over all frames as the unconditional worst case.
  Regression-tested.
- **The instrument-active half is now fixed too.** `active_signal_snr_db` derives an activity mask
  from clean-frame RMS using the declared 30 dB threshold and records
  `signal_active_fraction`, `snr_signal_active_db`, and `snr_signal_active_frames`. It answers
  “how masked is the note while the note is sounding?” independently of noise activity.
- **Scope, stated precisely:** this is an energy-derived frame mask, not a human annotation, and it
  does not change mixing. The condition label remains whole-window SNR; active-instrument SNR is an
  additional diagnostic.

### ✅ 6. No model-effective SNR diagnostic
- **Fixed.** `effective_snr_db` resamples clean and added components to each model's input rate and
  remeasures. Recorded as `snr_effective_ast_16k_db`, `snr_effective_mert_24k_db`,
  `snr_effective_panns_32k_db`. Uses `resample_poly`, the same polyphase resampler the model
  adapters use, and exploits the linearity of resampling to filter the added component directly.
- **Demonstrated:** HF-only noise at nominal 0 dB arrives at **AST as +23.1 dB** (everything above
  8 kHz is gone) but at PANNs as +1.0 dB. Low-frequency noise is unaffected at every rate.
- **Why it matters for the paper:** without this column, a table row labelled "0 dB" implies the
  three pretrained models received equally corrupted input. They did not, and the difference is
  large enough to change how a robustness ranking reads.
- **Scope:** this measures the resampling stage, which is the dominant effect. It does not model
  AST's or PANNs' internal mel front-end and normalization — after those, "SNR" is no longer a
  well-defined power ratio.

### 🟡 7. Noise adapters missing for CNN, CRNN, AST
- **Fixed — the adapters exist.** `noise_eval_cnn.py`, `noise_eval_crnn.py`, `noise_eval_ast.py`.
  CRNN imports the CNN machinery rather than copying it.
- **Also fixed — the blocker.** `finalize_cnn` now emits `label_order`, `test_examples`,
  `config_fingerprint` and `test_metrics.macro_f1`; `train_ast` writes a contract-shaped
  `test_summary.json`. Without these, `load_official_summary` and the clean-parity gate could not
  run for these models at all.
- **Key correctness property:** the CNN/CRNN adapters do **not** read `features/cnn/*.npz` (the
  clean log-mel). They recompute via `featurelib.logmel` + loaded train stats.
  `test_matches_the_cached_clean_features` verifies against the real Step-7 arrays that this
  reproduces them exactly, so clean and noisy features are provably the same transform.
- **Not yet verified:** neither torch loop nor the `transformers` path has ever executed — no torch
  in the audit environment, and no current clean CNN/CRNN/AST checkpoint to run against. The
  surrounding pure logic is tested (`logmel_input`, `ensemble_scores`).

### ✅ 8. Statistics ignore reused noise files
- **Fixed.** `run_noise_evaluation` now attaches a `noise_source` column to every prediction CSV,
  joined from the provenance written by the generator, and `noise_stats --cluster noise_source`
  resamples by the ESC-50 recording drawn. `pitch_group` remains the default, because it is the
  conservative unit for comparing two conditions of one model; `noise_source` is the right unit when
  the correlation of concern runs through the **noise** instead — several windows sharing one
  destructive recording fail together.
- **Reuse rate is recorded** per condition as `noise_source_distinct_fraction` in
  `metrics_{condition}.json`: 1.0 means every window drew a distinct recording, 0.1 means ten shared
  each one.
- **Timing mattered:** the column has to exist when predictions are written. Adding it afterwards
  would mean re-joining every prediction file against provenance by hand. No evaluation has run yet,
  so it costs nothing.
- **Degrades safely:** clean rows get `"clean"`, unreadable provenance gets `"unknown"` — both
  single-valued, so clustering on them collapses to the ungrouped case rather than inventing groups.

---

## Clean-model comparison

### ⬜ 9. AST and PANNs access test too early
- **Verified open:** no `final_evaluation_status.json` under `artifacts/ast/`, and no sealed-test
  guard in `train_ast.py` or `train_panns.py`. AST builds its test loader before training; PANNs
  probe mode precomputes test embeddings before selection.
- **Contrast:** SVM, MERT, CNN and CRNN all have one-shot status files. AST and PANNs are the two
  outliers.

### 🟡 10. CNN/CRNN optimize balanced accuracy, not macro-F1
- **Improved:** `finalize_cnn` now *records* macro-F1 (added for item 7), so CNN/CRNN results are
  comparable on the project's primary metric and can pass the parity gate.
- **Still open:** `train_cnn` still *selects* on `val_balanced_accuracy`, including the combiner
  choice. Selection metric and refit procedure remain inconsistent with SVM/MERT.
- **Note the underlying tension:** `step4_window.py` argues balanced accuracy and MCC are the
  *right* metrics under imbalance, while the whole evaluation stack uses macro-F1. That
  contradiction is still unresolved project-wide, and it decides which of the two should change.

### 🟡 11. Inconsistent neural seed coverage
- **CNN/CRNN:** 5 seeds (`DEFAULT_SEEDS = (42, 43, 44, 45, 46)`) plus ensembling. Good.
- **Still open:** AST, MERT and PANNs each default to `--seed 0` and were run once. No seed sweep,
  no across-seed variance for any of them.

### 📝 12. Pretraining is a major interpretation difference
MERT, AST and PANNs have learned from large real-world audio; SVM, CNN and CRNN have not. Not
fixable — must be treated as an experimental factor and stated wherever the families are compared.

---

## Dataset limitations

### 🟡 13. Resampling may not remove codec shortcuts
- **Improved in `5c7400b`:** `step1_resample`'s Nyquist check now *asserts* instead of printing
  "INVESTIGATE" and returning 0, and it runs before the fingerprint is stamped, so a failure cannot
  leave a valid-looking manifest.
- **Still open by design:** the between-class ceiling spread is a printed diagnostic, deliberately
  not asserted (a residual spread is expected). Lower-frequency MP3 artifacts remain unmeasured.

### 🟡 14. Tiling creates repeated audio
- **Verified:** still 8,152 of 8,378 windows (**97.3%**) tiled; median `content_s` 0.906 s.
- **Materially reduced in `69df21a`:** one window per source means window count no longer varies
  with recording length, removing that shortcut channel. `MediumCNN` uses global average pooling
  specifically so it cannot read loop period; `crnn_model` documents a measured probe finding no
  excess period-matched errors.
- **Still open:** repetition itself remains, the CRNN *can* read it, and the probe was run on clean
  audio only — never under noise, which is exactly where the shortcut might matter.

### ⬜ 15. Recording-session shortcuts
Files from one instrument may share microphone, room, performer and session.
- **Verified open:** no external evaluation set. `config.py` mentions a TinySOL producer stage in
  `MANIFEST_PRODUCER_STAGES`, but `build_tinysol_manifest` **does not exist** in the tree.
- **Needed:** an external corpus (TinySOL or similar) to measure it. Scope decision, not a bug.

### ✅ 16. Clean fingerprints don't hash every WAV
- **Fixed.** `audio_inventory.py` hashes every window listed in `windows.csv` into one digest —
  `sha256` over `"<relative path>\0<file sha256>\n"`, path-sorted — and stores it in the existing
  sidecar's `metadata`. Same construction already used for the ESC-50 corpus.
- **Recorded for the current build:** 8,378 files,
  `0fe284c5a2ab86aaf037c6592b497ce5b65732c1cc57be37b50d1c2db6165ad3`. `--verify` round-trips to
  `match`, and the pre-existing `assert_artifact_fingerprint` check still passes because the CSV
  itself is untouched — so this was added to a live build without regenerating anything.
- **Demonstrated:** a test overwrites one WAV with same-length different bytes. The CSV sidecar check
  still passes (that is the gap), and `verify_window_audio` raises. A rename at identical bytes is
  also caught, because the path is part of each hashed record.
- **Opt-in by design:** hashing ~1.1 GB takes seconds, so it is a separate command rather than a hook
  in every stage's happy path. And an unrecorded build reports `not_recorded` rather than failing —
  refusing to load every artifact built before the check existed would be worse than the gap.
- **Usage:** `python -m instrument_robustness.audio_inventory --record` once after a build, then
  `--verify` whenever provenance is in question (`--required` to make absence an error).

### ✅ 17. No DC-offset removal or audit
- **Audited, and the answer is that no removal is warranted.** Measured on the current build:

  ```text
  clean windows (400 sampled):  worst DC power share 1.06e-04  ->  4.59e-04 dB SNR error
  Gaussian draws (200):         worst DC power share 1.76e-04  ->  7.64e-04 dB SNR error
  ```

  The generator's tolerance is 0.1 dB, so DC contributes ~130x less than the tolerance. Subtracting
  it would alter the corpus's actual content to correct an error far below what anything measures.
- **Instrumented rather than assumed.** ESC-50 was **not** part of that audit — the corpus is absent
  locally, and real recordings can carry a genuine offset from AC coupling. So `noise_dc_offset` and
  `noise_dc_power_share` are now recorded per mixture, and `validate_noise_manifest` warns if any
  mixture exceeds `MAX_DC_POWER_SHARE` (1%, ~50x the audited worst case). It warns rather than fails:
  the effect would have to be ~1000x larger than measured to matter, and silently rejecting a corpus
  clip is worse than reporting it.
- **Net:** resolved by measurement plus a guard, not by a transformation.

---

## Statistics and reporting

### ✅ 18. No plan for uneven SNR spacing / curve area
- **Fixed.** `robustness_curve.py` adds `robustness_auc` (trapezoidal in dB, normalised by span),
  `snr_at_retention` (the SNR where a model keeps 50%/90% of clean macro-F1 — often the more legible
  headline), `mean_retention` for comparison only, and a CLI that summarises a completed
  `noise_sweep_summary.csv`.
- **Correcting my own earlier claim:** I wrote that the new grid is "deliberately uneven". It is
  **not** — `[60, 50, 40, 30, 20, 10, 0]` has uniform 10 dB gaps. For this exact grid the weighted
  and unweighted summaries differ only by trapezoidal endpoint weighting (0.390 vs 0.406 on the SVM
  white curve).
- **The tooling still matters, because uniformity is fragile.** Adding two levels where the model
  happens to do well moves the unweighted mean by **+0.093** and the integral by **+0.0004** — same
  model, more sampling. `--snrs` and `snr_range` both produce non-uniform selections, and item 2
  explicitly expects a grid retune once a pretrained model is piloted.

### ✅ 19. No multiple-comparison correction
- **Fixed.** `robustness_curve.benjamini_hochberg` controls the false-discovery rate over a family of
  comparisons, returning per-comparison rank, critical value, monotone step-up q-value, and
  rejection flag.
- **The family label is a required argument, not optional.** Correcting 21 conditions for one model
  is a different claim from correcting 21 × 5 models, and a reader cannot check the arithmetic
  without knowing which was done — so the choice has to appear in the output.
- **BH rather than Bonferroni** on purpose: it controls the expected *proportion of false positives
  among rejections*, which is the right target here, where comparisons are positively correlated
  (the same test windows at neighbouring SNRs). Bonferroni would be far more conservative for no
  benefit.
- **Demonstrated:** on a synthetic family of 3 real effects and 30 nulls, 7 comparisons pass an
  uncorrected p < 0.05 while BH rejects exactly the 3 true ones.

### 📝 20. Only digitally added noise
No microphones, rooms, reverberation, competing instruments, or real recording change. Must be an
explicit scope limit in the paper — the benchmark cannot support a general real-world robustness
claim.

---

## Documentation drift

### 🟡 21. Test-window count: 1,255 vs 1,310
- **Root cause confirmed:** `69df21a` cropped to one window per source. Current `windows.csv` has
  8,378 windows = 8,378 sources, max 1 per source, split **5,864 / 1,259 / 1,255**. So 1,255 is
  correct and 1,310 is the pre-crop build.
- **SVM: fixed.** `artifacts/svm/` now passes the real gate —
  `load_official_summary` with the model-hash check succeeds, `test_examples=1255`,
  macro-F1 **0.991446**, status `complete`. Old runs archived under `legacy/svm_runs/`.
- **MERT: fixed.** `artifacts/mert/` now contains the current 5,864/1,259/1,255 run, with test
  macro-F1 **0.924598**, status `complete`, and a model hash that passes the shared gate.
- **Still stale:** `artifacts/ast/` has 1,310 test examples and predates the crop. It must be
  regenerated before AST enters the shared comparison.
- **Fails closed, not silently:** the stale pairs raise `StaleArtifactError` naming
  `max_windows_per_source`, and the parity gate compares example counts. No wrong number can be
  reported from them.

### ✅ 22. README references a nonexistent `pipeline_report.txt`
- **Fixed.** `all-samples/pipeline/pipeline_report.txt` is absent and no stage writes it; the README
  no longer presents it as an existing report.
- **Fixed alongside the grid change:** the README's and `docs/NOISE_PLAN.md`'s stale grid text
  (20/10/5/0/−5, "5.2 GB", "1,310 windows", "16 conditions") is now correct and cites `config.SNRS`.
- **Fixed:** the README now points at what actually exists — per-stage console output, the
  fingerprint sidecars beside every manifest, and `_step4_report_block.txt` — and states explicitly
  that `pipeline_report.txt` was never written by any stage.
- **Dead code, flagged not deleted:** `config.REPORT` still names the phantom file and has no
  readers. Left in place per the repo's own convention on unrelated dead code.

---

## Suggested order

**Before `noise_sweep --generate`:** items 1, 3, 5 and 6 are **done**. What remains before
generating is finishing **2** — pilot a pretrained model, since the grid's low end is currently a
hedge — and deciding whether to raise `N_REPLICATES` above 1, because changing it later means
regenerating.

**Before any model-comparison claim:** **9** and **11** remain (8 and 19 are done).

**Before publication:** **21** (MERT + AST), **10**, and write-ups for **12**, **14**, **15**,
**20**.

> **The ordering constraint that dominates everything:** the noise sweep's dataset fingerprint hashes
> `manifest.csv` and `windows.csv`. Any re-run of Steps 0–5 invalidates a generated sweep. Finish all
> data and provenance decisions → freeze the build → generate → evaluate. Never regenerate data after
> generating noise.
