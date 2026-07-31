# Repository Audit

Audit date: 2026-07-31

Audited branch: `main`

Audited commit: `417c57d` (`plan_noise.md: the noise methodology in plain terms`)

Audit mode: read-only inspection plus safe tests; no experiment, data, or model regeneration

> **Post-audit remediation, 2026-07-31.** This document preserves the evidence observed at the
> audited commit; its old counts/results are historical, not current. DATA-001 was resolved by
> excluding all four ambiguous files (rather than guessing labels), and the pipeline was rebuilt
> and sealed at 8,374 sources: train 5,861, validation 1,258, test 1,255; 544 pitch groups; zero
> cross-label exact-audio groups. Dataset fingerprint:
> `26f067648aa90f586299001a26b4eca3a294f277e98a669af97025005884b7d9`.
> MODEL-001/MODEL-002/MODEL-003 were repaired in code: AST/PANNs now train on train/validation
> only, select on validation macro-F1, hash their selected/base checkpoints, and use separate
> one-time finalizers. The stale active artifacts and weight copies described below were removed;
> all six models and the shared noise corpus still require rerunning on this frozen build.

## Evidence labels

The following labels have precise meanings throughout this report:

- **VERIFIED FROM CODE** — directly established by executable source.
- **VERIFIED FROM CONFIGURATION** — directly established by the active configuration.
- **VERIFIED FROM GENERATED DATA** — directly measured from local manifests, arrays, audio, or result artifacts.
- **VERIFIED BY TEST** — exercised by a test that passed during this audit.
- **INFERRED** — a reasoned consequence of verified evidence, but not directly demonstrated by an experiment.
- **PLANNED BUT NOT IMPLEMENTED** — described as future work but absent from executable code.
- **LEGACY** — belongs to an older pipeline, label set, dataset build, or experiment.
- **UNRESOLVED** — the repository does not contain enough evidence to decide.

## 1. Executive summary

The repository has a substantially engineered 12-class data and robustness pipeline, with strong provenance checks, pitch-group split isolation, training-only normalization statistics, fixed label-order validation, deterministic noise generation, and unusually broad unit coverage for noise mathematics. **VERIFIED FROM CODE; VERIFIED BY TEST.** The current clean data build contains 8,378 retained recordings and exactly one three-second window per recording, split 5,864/1,259/1,255 into train/validation/test with no `(instrument, note)` group crossing splits. **VERIFIED FROM GENERATED DATA.**

It is not yet scientifically ready for a final six-model comparison or paper submission. Three critical problems block that claim:

1. Two pairs of retained raw files are byte-identical while carrying conflicting instrument labels; all four are in the training split. This is confirmed label/data contamination, although it does not cross splits. **VERIFIED FROM GENERATED DATA.**
2. Saved PANNs noise outputs were generated under a different dataset fingerprint, window-manifest hash, and noise-manifest hash from the saved SVM, MERT, and AST outputs. Direct comparison shows that the supposedly paired ESC-50 condition selects different noise sources for essentially every test recording. Saved cross-model PANNs robustness comparisons are therefore not paired on the same mixtures. **VERIFIED FROM GENERATED DATA.**
3. The new canonical-looking `models/` bundle does not contain the weights behind the reported AST and PANNs results: its AST hash is `3133ad96…`, while the current AST summary names `25789685…`; it bundles the PANNs linear probe while `docs/RESULTS.md` and the noise outputs report the full fine-tune. **VERIFIED FROM GENERATED DATA.**

Additional high-impact gaps are that CNN and CRNN validation artifacts do not hash to the locally available feature arrays; CNN/CRNN lack final test/noise results; AST and PANNs access the test split inside their training entry points rather than through a sealed finalizer; the exact canonical AST checkpoint remains external; and current statistical-comparison outputs have not been generated. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.**

Current readiness:

- Data-pipeline implementation: mature, but the contaminated source pairs require resolution and a clean rebuild. **NOT READY for final results.**
- SVM: implementation and clean evaluation workflow are complete; final results must be rerun after resolving the data issue. **READY WITH CONDITIONS.**
- MERT: implementation and clean evaluation workflow are complete; final results must be rerun after resolving the data issue. **READY WITH CONDITIONS.**
- CNN/CRNN: validation training exists, but exact-input provenance and final evaluation are incomplete. **NOT READY.**
- AST/PANNs: implemented and evaluated, but test-isolation and artifact-provenance weaknesses remain. **NOT READY for a locked paper benchmark.**
- Noise generation: implementation is strong, but saved model outputs do not all share one realized corpus. **NOT READY for final comparison.**
- Paper Methods: mostly documentable after resolving several exact facts. **MOSTLY READY.**
- Paper Results/submission: blocked. **BLOCKED.**

Issue counts in the register: **3 CRITICAL, 7 HIGH, 12 MEDIUM, 5 LOW, and 3 INFORMATIONAL (30 total).**

## 2. Audit scope and limitations

### Inspected

- Repository guidance and documentation: `AGENTS.md`, `README.md`, `CLAUDE.md`, `docs/`, model/data plans, and SCC instructions.
- Git status, branch, recent local history, remote configuration, tracked-file inventory, and artifact sizes.
- All modules under `src/instrument_robustness/`, all model/data YAML files, SCC job scripts, tests, dependency metadata, environment template, and `.gitignore`.
- Local generated manifests, fingerprint sidecars, clean feature arrays, saved model summaries/checkpoints where present, and compact noise-evaluation outputs.
- Every current normalized window for count, existence, content identity, and a read-only frame-activity analysis.
- Saved result provenance and hashes across SVM, MERT, CNN, CRNN, AST, and PANNs.

### Commands run

The following classes of commands were read-only unless noted as tests using temporary directories:

```bash
git status --short
git branch --show-current
git log --oneline --decorate -30
git remote -v
git rev-list --left-right --count HEAD...origin/main
git ls-files
git check-ignore -v <selected paths>
git count-objects -vH
rg --files
rg -n <targeted symbols and documentation claims> ...
find <selected repository directories> ...
du -sh <selected directories>
```

Targeted Python inspection loaded CSV/JSON/NPZ files read-only to compute counts, label distributions, pitch groups, hashes, array shapes/dtypes/finite values, feature statistics, exact-audio duplicate groups, duration summaries, activity summaries, and result-provenance comparisons. The complete normalized-window inventory was also checked using:

```bash
.venv/bin/python -m instrument_robustness.audio_inventory --check
```

Safe tests were run in increasing scope:

```bash
.venv/bin/python -m unittest tests.test_preprocessing tests.test_svm -q
.venv/bin/python -m unittest tests.test_noise tests.test_noise_metrics tests.test_noise_adapters tests.test_robustness_curve -q
.venv/bin/python -m unittest tests.test_ast tests.test_mert -q
.venv/bin/python -m unittest tests.test_mert -q
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m instrument_robustness.bundle_models --check
.venv/bin/python -m instrument_robustness.bundle_weights --check
```

### Not run

No download, pipeline stage, feature extraction, noisy-corpus generation, training, test finalization, expensive model inference, dependency installation, formatter, linter, or type checker was run. No formatter/linter/type-checker configuration exists to invoke. **VERIFIED FROM CODE; VERIFIED FROM CONFIGURATION.**

### Limitations

- The full 60,240-file noisy corpus and `noise_provenance.csv` were not present locally, so their audio content, peak distribution, DC-offset distribution, and achieved SNR could not be independently rechecked in this audit. Compact per-model result files were available. **UNRESOLVED.**
- MERT and PANNs cached embedding arrays were not present locally; their saved summaries were inspected, but the arrays could not be rehashed. **UNRESOLVED.**
- The current AST runtime dependencies are not installed in the audited local environment, so the AST test module was skipped during discovery. **VERIFIED BY TEST.**
- The locally known `origin/main` reference equalled `HEAD`; no network fetch was performed, so remote freshness beyond that reference is not guaranteed. **UNRESOLVED.**
- This audit can identify exact duplicate audio but cannot determine which metadata label or upstream source file is correct without external provenance. **UNRESOLVED.**

## 3. Git and repository state

- Branch: `main`. **VERIFIED FROM GENERATED DATA.**
- HEAD at final audit verification: `417c57d`; fetched `origin/main` also pointed to `417c57d`, with ahead/behind count `0/0` before the audit commit. **VERIFIED FROM GENERATED DATA.**
- Working tree before creating this report: no tracked modifications, with two pre-existing untracked directories: `artifacts/mert/noise/` and `artifacts/svm/noise/`. **VERIFIED FROM GENERATED DATA.**
- Remote: `https://github.com/maxliu2k/rise.git`. **VERIFIED FROM GENERATED DATA.**

Recent relevant commits include:

| Commit | Relevance |
|---|---|
| `417c57d` | Added plain-language noise methodology |
| `48ff616` | Changed current CNN/CRNN selection to validation macro-F1 and marked old summaries stale |
| `7dd6d40` | Documented `models/` and weight-bundle tooling |
| `b7b2b90` | Added generated repository map and optional pre-commit hook |
| `48e9526` | Added the flat trained-weight bundle |
| `da74be8` | Added result consolidation and `docs/RESULTS.md` |
| `7f33973` | Ported project guidance and findings |
| `d5af05b` | AST noise evaluation |
| `a0b0015` | Model bundle/provenance snapshot |
| `79c129e` | Five-seed CRNN validation workflow |
| `18b7049` | Five-seed CNN validation workflow |
| `b6a7329` | PANNs noise evaluation |
| `aeaaa44` | AST clean results |
| `14238b8` | Clean-parity tolerance |
| `018f8cd` | Noise DC-offset removal |
| `a5eac9e` | Deadline-bounded reduced noise sweep |

Repository hygiene is inconsistent. Of 477 tracked paths after the fast-forward, 280 are under `artifacts/` and 18 are under the new `models/` bundle. The stale AST checkpoint is duplicated as two approximately 345 MB worktree files (one Git-LFS-managed), while SVM/MERT noise results are untracked and PANNs/AST noise results are tracked. **VERIFIED FROM GENERATED DATA.** Small metadata are intentionally re-included beneath ignored data directories, but `.gitignore` also re-includes a nonexistent `pipeline_report.txt`. **VERIFIED FROM CONFIGURATION.**

## 4. Architecture and repository map

### End-to-end workflow

```text
Philharmonia archive
  -> prep_data.py: discover/download, parse filenames, manifest.csv
  -> step0_filter.py: 12 target labels + one plain technique per class
  -> step1_resample.py: mono, 22.05 kHz, PCM16 WAV
  -> step2_trim.py: energy-based leading/trailing trim
  -> step3_split.py: grouped (label, note) 70/15/15 assignment
  -> step4_window.py: first onset-aligned 3 s window; tile short recordings
  -> step5_normalize.py: per-window RMS 0.1 with peak guard
  -> step6_stats.py: train-only log-mel and handcrafted statistics
  -> step7_featurize.py: SVM 88-D and CNN/CRNN 128 x 130 arrays
  -> clean model training/validation/finalization
  -> noise_sweep.py: shared test-only waveform corruptions
  -> model-specific noise adapters
  -> noise_stats.py / robustness_curve.py: uncertainty and comparisons
```

**VERIFIED FROM CODE:** [`run_pipeline.py`](../src/instrument_robustness/run_pipeline.py), [`prep_data.py`](../src/instrument_robustness/prep_data.py), and `step0_filter.py` through `step7_featurize.py`.

### Repository map

| Path | Purpose | Current consumer | Generated or source-controlled | Status |
|---|---|---|---|---|
| `src/instrument_robustness/` | Pipeline, models, noise, evaluation | All CLI workflows | Source-controlled | CURRENT |
| `configs/data/` | Dataset YAML documentation | Human/SCC workflows; not the primary executable config | Source-controlled | CURRENT with stale placeholder |
| `configs/models/` | Model YAML documentation/default records | Human/SCC workflows; Python CLIs own many actual defaults | Source-controlled | CURRENT but drift-prone |
| `tests/` | Unit/integration tests | `unittest discover` | Source-controlled | CURRENT |
| `docs/` | Methods, audit checklist, historical findings | Researchers/paper writing | Source-controlled | MIXED: CURRENT + LEGACY + STALE |
| `scc/` | SCC environment and job wrappers | Cluster users | Source-controlled | CURRENT but incomplete |
| `all-samples/manifest.csv` | Canonical source index | Step 0 and fingerprinting | Generated, selectively tracked | CURRENT GENERATED |
| `all-samples/pipeline/*.csv` | Labeled/resampled/trimmed/split/window contracts | Downstream pipeline and models | Generated, selectively tracked | CURRENT GENERATED |
| `all-samples/pipeline/*fingerprint.json` | Content/stage provenance | Every downstream stage | Generated, selectively tracked | CURRENT GENERATED |
| `all-samples/features/svm/` | Standardized 88-D arrays | SVM clean/noise workflows | Generated, ignored locally | CURRENT GENERATED |
| `all-samples/features/cnn/` | Standardized log-mel arrays | CNN/CRNN workflows | Generated, ignored locally | CURRENT GENERATED but provenance conflict |
| `all-samples/features/mert/` | MERT embeddings/plans | MERT workflow | Large arrays absent locally | UNCLEAR/GENERATED |
| `all-samples/features/panns/` | PANNs cache/checkpoint plan | PANNs workflow | Large files external/ignored | CURRENT GENERATED |
| `artifacts/svm/` | Selected/final SVM and clean results | SVM/noise evaluation | Tracked generated output | CURRENT |
| `artifacts/mert/` | MERT summaries/checkpoints/results | MERT/noise evaluation | Mixed tracked/untracked output | CURRENT with duplicate run history |
| `artifacts/cnn/`, `artifacts/crnn/` | Five-seed validation checkpoints/summaries | Finalizers/noise adapters | Tracked generated output | CURRENT but exact-input mismatch |
| `artifacts/new-ast-results-20260730-022036/` | Current AST clean-result record | AST noise adapter | Tracked generated output; model external | CURRENT canonical summary |
| `artifacts/ast-modified-sound-20260730-160714/` | Current AST noise-evaluation outputs | Result analysis | Tracked generated output | CURRENT GENERATED |
| `artifacts/ast/` | Older AST output and 345 MB checkpoint | No current canonical workflow | Tracked generated output | LEGACY/STALE |
| `artifacts/panns/` | PANNs results/provenance/noise outputs | PANNs comparison | Tracked generated output, checkpoints released externally | CURRENT with incomplete source provenance |
| `model_bundle/` | Snapshot of model-relevant source | `bundle_models --check` | Generated and tracked | STALE DUPLICATE |
| `models/` | Flat weight handoff assembled by `bundle_weights.py` | Human/release handoff only; production code still loads `artifacts/` | Generated copies, tracked; AST via Git LFS | CURRENT build output but scientifically inconsistent |
| `legacy/9class_file_split/`, `legacy/svm_runs/` | Retired nine-class/leaking-split metadata and prior SVM runs | Historical audit only | Source-controlled legacy snapshot | LEGACY |
| `data/noise/train/` | Empty noise placeholder | None found | Source-controlled empty directory | UNCLEAR |
| `configs/data/irmas.yaml` | Zero-byte configuration placeholder | None found | Source-controlled | STALE |
| `.venv/`, caches, raw/work audio | Local runtime/generated data | Development only | Ignored | GENERATED |

No notebooks are present. No CI configuration is present. **VERIFIED FROM GENERATED DATA.**

## 5. Current implementation status

| Component | Status | Evidence | Qualification |
|---|---|---|---|
| Philharmonia acquisition and parsing | IMPLEMENTED | `prep_data.py`; current `manifest.csv` | Internet Archive mirror; one unreadable source excluded |
| Filtering/resampling/trimming/splitting/windowing/normalization | IMPLEMENTED | Steps 0–5 and current manifests | Requires rebuild after label-contamination resolution |
| SVM features and model | IMPLEMENTED | `featurelib.py`, `train_svm.py`, `finalize_svm.py` | Clean final and noise outputs exist |
| CNN | PARTIALLY IMPLEMENTED | `cnn_model.py`, `train_cnn.py`, `finalize_cnn.py` | Validation-only results; feature hashes conflict; no official test/noise |
| CRNN | PARTIALLY IMPLEMENTED | `crnn_model.py`, `train_crnn.py`, `finalize_crnn.py` | Validation-only results; feature hashes conflict; no official test/noise |
| AST | IMPLEMENTED | `ast_data.py`, `train_ast.py`, `noise_eval_ast.py` | Test is evaluated inside training; canonical checkpoint absent locally |
| MERT | IMPLEMENTED | `extract_mert.py`, `train_mert.py`, `finalize_mert.py` | Frozen embeddings + learned layer mixture/linear probe; clean/noise outputs exist |
| PANNs probe/fine-tune | IMPLEMENTED | `train_panns.py`, `noise_eval_panns.py` | Test is loaded/evaluated inside training; checkpoint distributed externally |
| Shared waveform noise generation | IMPLEMENTED | `noise_sweep.py`, `noise_metrics.py` | Saved model outputs do not all share one realized corpus |
| White Gaussian noise | IMPLEMENTED | `noise_sweep.draw_noise` | Deterministic per build/window/type/replicate |
| ESC-50 natural/mechanical noise | IMPLEMENTED | `ESC50_TARGETS`, `load_esc50_index` | Targets 0–19 natural, 30–49 mechanical |
| DEMAND | PLANNED BUT NOT IMPLEMENTED | Historical docs only | No executable adapter/catalog found |
| MUSAN/UrbanSound8K | PLANNED BUT NOT IMPLEMENTED | Discussion/docs only | No executable implementation found |
| Speech/music/reverberation corruptions | ABSENT | Repository search | Human/non-speech ESC targets 20–29 are excluded |
| Robustness statistics | IMPLEMENTED, NOT YET APPLIED | `noise_stats.py`, `robustness_curve.py` | No final cross-model statistical output found |

## 6. Data-pipeline findings

| Stage | File and symbol | Input | Output | Active configuration | Main risk | Audit result |
|---|---|---|---|---|---|---|
| Discovery/acquisition | `prep_data.py` | Philharmonia archives | raw files + `manifest.csv` | 12 labels, archive mappings | filename parsing/source availability | **VERIFIED FROM CODE**; current manifest is structurally complete |
| Metadata parsing | `prep_data.py` parser | filenames | instrument, note, duration, dynamic, technique | strict parser/known names | metadata encoded in filenames | **VERIFIED FROM GENERATED DATA**; no retained missing fields |
| Filtering | `step0_filter.main` | manifest | `manifest_labeled.csv` | target labels + `STRICT_ARTICULATIONS` | reduced real-world technique diversity | **VERIFIED FROM CODE**; strict articulation is enforced |
| Decode/resample | `step1_resample.resample_one` | source audio | mono 22,050 Hz PCM16 WAV | `SR=22050` | library/codec version | **VERIFIED FROM CODE** at lines 28–39; Nyquist guard exists |
| Trim | `step2_trim.trim_one` | resampled WAV | trimmed PCM16 WAV | `top_db=30`, minimum 0.10 s | no explicit retained context; library defaults | **VERIFIED FROM CODE** at lines 20–35 |
| Split | `step3_split.assign_groups` | trimmed manifest | `splits.csv` | 70/15/15, seed 0, group=`(label,note)` | unequal group sizes prevent exact fractions | **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA** |
| Window | `step4_window.window_one` | trimmed WAV + source split | one 3 s PCM16 WAV/source | hop 3 s, max one, tile short clips | repeated attacks/duration cue | **VERIFIED FROM CODE** at lines 69–92 |
| Normalize | `step5_normalize.norm_one` | window WAV | in-place normalized PCM16 | RMS 0.1, peak 0.99 | quantization/rewrite; guarded windows below target | **VERIFIED FROM CODE** at lines 22–42 |
| Stats | `step6_stats.main` | train windows | `norm_stats.npz/json` | train only | positional feature contract | **VERIFIED FROM CODE; VERIFIED BY TEST** |
| Features | `step7_featurize.main` | windows + train stats | SVM/CNN NPZ per split | fixed label order | cached-array build drift | **VERIFIED FROM CODE; conflict found for CNN/CRNN artifacts** |
| Pretrained inputs | AST/MERT/PANNs data modules | normalized waveform windows | processor tensors/embeddings | 16/24/32 kHz by model | processor/checkpoint version | **VERIFIED FROM CODE** |

### Manifest and data contracts

| Contract | One row represents | Primary/source/window identifier | Key fields | Downstream consumers | Audit result |
|---|---|---|---|---|---|
| `all-samples/manifest.csv` | one decoded source recording | `path` / `path` / none | label, family, source duration/rate, note/MIDI, dynamic, technique, phrase/plain flags | Step 0, dataset identity | 10,196 unique rows/paths; no missing existing paths in the audited build |
| `pipeline/manifest_labeled.csv` | one retained plain-technique source | `path` / `path` / none | same source metadata | Step 1 | 8,378 rows, all 12 labels, no missing retained metadata |
| `pipeline/manifest_resampled.csv` | one retained source after decode/resample | `path` / `path` / none | `resampled_path`, duration, status | Step 2 | 8,378 successful rows |
| `pipeline/manifest_trimmed.csv` | one retained source after trim | `path` / `path` / none | `trimmed_path`, duration, trim flag | Step 3 | 8,378 rows; 3 fallback-untrimmed, no errors |
| `pipeline/splits.csv` | one original retained source assignment | `source_path` / `source_path` / none | trimmed path, label, note, split, phrase flag | Step 4 and split audits | 8,378 unique source assignments; split values valid |
| `pipeline/windows.csv` | one normalized model example | `window_path` / `source_path` / `window_path` stem | label, note, split, start, distinct `content_s`, pre/post RMS | features and all waveform models/noise | 8,378 unique windows; every file exists; fixed rate/count are implied by fingerprint rather than columns |
| `features/svm/{split}.npz` | one 88-D window feature vector | positional row; `source_path` metadata | `X`, `y`, feature names, label names, fingerprint | SVM | row counts align with `windows.csv`; current hashes match official SVM summaries |
| `features/cnn/{split}.npz` | one 128×130 log-mel window | positional row; `source_path` metadata | `X`, `y`, label names, fingerprint | CNN/CRNN | row counts/shape valid; saved CNN/CRNN run hashes conflict |

**VERIFIED FROM GENERATED DATA; VERIFIED FROM CODE.** No manifest stores sample count or frame-level active fraction; consumers enforce the fixed sample rate/length from configuration and fingerprint sidecars. NPZ rows are aligned positionally rather than through an explicit `window_id` array, so content hashes and row-count/order validation are load-bearing. **VERIFIED FROM CODE; INFERRED.**

### Silence repair

The prior excessive-silence approach has been replaced by energy trimming followed by a first, onset-aligned window. Short recordings are tiled repeatedly to exactly 3.0 seconds instead of being zero-padded; only one window is retained per source. **VERIFIED FROM CODE:** [`config.py` lines 79–125](../src/instrument_robustness/config.py#L79), [`step2_trim.py` lines 20–35](../src/instrument_robustness/step2_trim.py#L20), and [`step4_window.py` lines 53–92](../src/instrument_robustness/step4_window.py#L53).

Exact current behavior:

- Fixed duration: 3.0 s, 66,150 samples at 22.05 kHz. **VERIFIED FROM CONFIGURATION.**
- Activity detection: `librosa.effects.trim(..., top_db=30)` relative to the clip reference using the library's default frame/hop behavior; frame/hop are not passed explicitly. **VERIFIED FROM CODE.**
- Minimum trimmed duration: if trimming yields less than 0.10 s, the untrimmed signal is retained and flagged. **VERIFIED FROM CODE.**
- Context: no explicit pre-attack/post-decay context is restored after trimming. **VERIFIED FROM CODE.**
- Attack: the first surviving trimmed sample becomes the beginning of the sole window; actual perceptual attack preservation is not separately measured. **INFERRED; UNRESOLVED.**
- Padding: short windows are tiled, never zero-padded. **VERIFIED FROM CODE; VERIFIED BY TEST.**
- Rejection: a tiny trailing window below 0.5 s would be dropped only if it were not the first; with `MAX_WINDOWS_PER_SOURCE=1`, this branch is inactive for the current build. **VERIFIED FROM CONFIGURATION.**
- Saved activity: `content_s`, pre/post RMS, and trim duration are saved; frame-level active fraction is not. **VERIFIED FROM GENERATED DATA.**

Read-only activity analysis over all 8,378 current normalized windows used RMS frames of 2,048 samples, hop 512, active within 30 dB of each clip's loudest frame:

| Statistic | Active fraction |
|---|---:|
| Mean | 0.997367 |
| Median | 1.000000 |
| 5th percentile | 1.000000 |
| 25th percentile | 1.000000 |
| Minimum | 0.539683 |
| Below 25% active | 0.00% |
| Below 50% active | 0.00% |

**VERIFIED FROM GENERATED DATA.** Split means were train 0.997309, validation 0.997882, and test 0.997123. Instrument means ranged from 0.990296 (trombone) to 0.999989 (viola); minima ranged from 0.539683 (trombone) to 0.992063 (viola). These values describe the post-tiling waveform and therefore do **not** measure how much distinct source material remains.

The distinct pre-repeat content proxy `content_s / 3` has mean 0.3453, median 0.3019, 5th percentile 0.1084, 25th percentile 0.1829, and minimum 0.0261; 39.51% of windows contain less than 25% distinct source duration and 79.94% contain less than 50%. **VERIFIED FROM GENERATED DATA.** Thus silence was removed, but repeated waveform content is now common and must be described honestly.

Old flawed windows are not represented by a clearly versioned local legacy directory, although stale result artifacts and documentation remain. Current feature sidecars identify the corrected configuration, but not every saved model result hashes to the locally available feature files. **VERIFIED FROM GENERATED DATA; UNRESOLVED for exact old cache retention.**

## 7. Dataset composition and validity

### Counts

The current source manifest contains 10,196 indexed decodable recordings; its sidecar records one unreadable candidate excluded. Strict label/technique filtering discards 1,818 indexed recordings and retains 8,378; every retained recording survives resampling, trimming, splitting, and one-window generation. **VERIFIED FROM GENERATED DATA.** Trim status is 8,375 `ok` and 3 `kept_untrimmed`, with zero trim errors.

| Instrument | Raw indexed | Retained total | Train | Validation | Test | Unique notes | MIDI range |
|---|---:|---:|---:|---:|---:|---:|---:|
| bassoon | 720 | 648 | 454 | 97 | 97 | 45 | 34–79 |
| cello | 889 | 747 | 517 | 115 | 115 | 49 | 36–84 |
| clarinet | 846 | 770 | 547 | 112 | 111 | 47 | 50–96 |
| double-bass | 852 | 764 | 533 | 116 | 115 | 44 | 24–67 |
| flute | 878 | 781 | 548 | 116 | 117 | 42 | 60–101 |
| french-horn | 652 | 546 | 380 | 83 | 83 | 44 | 34–77 |
| oboe | 596 | 539 | 379 | 81 | 79 | 37 | 58–94 |
| trombone | 831 | 759 | 531 | 114 | 114 | 49 | 40–88 |
| trumpet | 485 | 433 | 303 | 65 | 65 | 45 | 40–88 |
| tuba | 972 | 831 | 583 | 124 | 124 | 42 | 22–65 |
| viola | 973 | 708 | 495 | 107 | 106 | 51 | 48–98 |
| violin | 1,502 | 852 | 594 | 129 | 129 | 49 | 55–103 |
| **Total** | **10,196** | **8,378** | **5,864** | **1,259** | **1,255** | **544 pitch groups** | — |

**VERIFIED FROM GENERATED DATA:** `manifest.csv`, `manifest_labeled.csv`, `splits.csv`, and `windows.csv`.

Every retained source contributes exactly one window; median/minimum/maximum windows per source are all 1. Class imbalance is 852/433 = 1.968:1. Every class occurs in every split. **VERIFIED FROM GENERATED DATA.**

### Metadata diversity

- Retained techniques: `normal` 5,307 and `arco-normal` 3,071. **VERIFIED FROM GENERATED DATA.** These are the intended single plain technique per class, not cross-class technique balance.
- Dynamics include forte 1,711; fortissimo 1,686; piano 1,497; pianissimo 1,346; mezzo-forte 1,120; mezzo-piano 840; and 178 rarer dynamic descriptors combined. **VERIFIED FROM GENERATED DATA.**
- The raw manifest contains 235 trill-named recordings; zero survive strict articulation filtering. **VERIFIED FROM GENERATED DATA.**
- Retained metadata include 8,316 note recordings and 62 phrase recordings. **VERIFIED FROM GENERATED DATA.**
- Retained source durations have mean 1.359 s, median 0.993 s, 5th percentile 0.366 s, 25th percentile 0.627 s, minimum 0.078 s, 95th percentile 2.377 s, and maximum 77.610 s. Cropping makes the final model duration fixed but does not remove repetition-period information. **VERIFIED FROM GENERATED DATA; INFERRED.**

### Authoritative label contract

The sole active source of truth is `TARGET_LABELS` in [`config.py` lines 35–42](../src/instrument_robustness/config.py#L35). The numerical order is:

| Index | Label | Index | Label |
|---:|---|---:|---|
| 0 | bassoon | 6 | oboe |
| 1 | cello | 7 | trombone |
| 2 | clarinet | 8 | trumpet |
| 3 | double-bass | 9 | tuba |
| 4 | flute | 10 | viola |
| 5 | french-horn | 11 | violin |

**VERIFIED FROM CONFIGURATION.** `prep_data`, Step 0, Step 7, every model head, clean finalizer, classification report, confusion matrix, and noise evaluator either derives from or validates against this order. Current SVM/CNN feature arrays and current saved clean summaries contain compatible 0–11 labels/order. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.** No conflicting active executable mapping was found.

The repository retains a deliberately labeled `legacy/9class_file_split/` snapshot and text describing the retired nine-class mapping. Those files are **LEGACY**, not an alternative source of truth. A “new 9-way head” comment in `pretrained_extractors.py` is **STALE** because the actual layer uses `len(TARGET_LABELS)=12`. Old nine-class checkpoints are explicitly invalid under the current index order. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.**

### Label validity problem

Hashing current retained raw audio found two exact-byte duplicate pairs with conflicting labels, all assigned to train:

- `cello/Ds5/cello_Ds5_05_forte_arco-normal.mp3` and `viola/G6/viola_G6_05_fortissimo_arco-normal.mp3` share SHA-256 `75c5dbcf…`;
- `french-horn/E2/french-horn_E2_1_fortissimo_normal.mp3` and `oboe/E6/oboe_E6_15_mezzo-forte_normal.mp3` share SHA-256 `7ae66673…`.

Their normalized window pairs also remain byte-identical (SHA-256 `8c13a4ef…` and `ed0c80de…`, respectively).

Hashing normalized windows found two duplicate groups/four rows and no duplicate group crossing splits. **VERIFIED FROM GENERATED DATA.** This is not train/test leakage, but it is confirmed label contamination: the same audio cannot truthfully be two different target instruments. The correct upstream labels are **UNRESOLVED**.

### Shortcut and validity risks

- Pitch is deliberately isolated by `(instrument, note)` group across splits, which prevents identical nominal pitches for an instrument crossing splits. It does not equalize pitch ranges between instruments; the model can still use legitimate but dataset-specific register. **VERIFIED FROM GENERATED DATA; INFERRED.**
- Technique is fixed per class (`normal` for winds/brass and `arco-normal` for strings), so technique and family are structurally confounded. This was chosen to remove within-class articulation variation, but prevents a claim that the classifier uses timbre alone. **VERIFIED FROM CONFIGURATION; INFERRED.**
- Dynamic, source duration, repetition period, and original encoding/recording-session properties may correlate with class. Resampling removes differential sample rate and caps frequency at one Nyquist, but does not prove removal of all codec or session cues. **INFERRED.**
- Absolute paths, source filenames, pitch, dynamics, and metadata are saved for provenance but are not included in SVM/log-mel feature values or pretrained waveform tensors. **VERIFIED FROM CODE.**
- This is an in-collection classification/robustness design, not evidence of general instrument recognition across recording collections. **INFERRED.**

## 8. Split integrity and leakage

The authoritative split is `all-samples/pipeline/splits.csv`. Assignment occurs before windowing, independently within each class, with `(label, note)` as the indivisible group and seed 0. Unequal group sizes mean exact 70/15/15 fractions are not guaranteed. **VERIFIED FROM CODE:** [`step3_split.py`](../src/instrument_robustness/step3_split.py), especially `assign_groups` and `verify_no_group_leak`.

Current split counts are 69.993% train, 15.027% validation, and 14.980% test. All 544 `(label,note)` groups occur in exactly one split. **VERIFIED FROM GENERATED DATA.**

| Leakage risk | Classification | Evidence |
|---|---|---|
| Same source recording crosses splits | PREVENTED + TESTED | source split precedes windows; one split/source; current metadata check |
| Overlapping windows cross splits | NOT APPLICABLE currently | exactly one window/source; no overlap |
| Same `(label,note)` group crosses splits | PREVENTED + TESTED | zero of 544 groups cross; preprocessing tests pass |
| Exact normalized waveform crosses splits | PREVENTED in current data | two duplicate groups exist, both confined to train |
| Exact waveform has conflicting labels | CONFIRMED ISSUE | two cross-label raw/window pairs in train |
| SVM scaler uses validation/test | PREVENTED + TESTED | Step 6 fits train; Step 7 reuses saved stats; loader does not re-standardize |
| Log-mel scaler uses validation/test | PREVENTED + TESTED | train-only per-mel statistics; adapter parity test |
| Hyperparameters selected on test: SVM/MERT | PREVENTED + TESTED | separate train/val selection and one-time finalizers |
| Hyperparameters selected on test: CNN/CRNN | PREVENTED so far | validation-only selection; test not finalized |
| Test loaded/evaluated by AST/PANNs training | CONFIRMED ISSUE | `train_ast.train`; `train_panns.run_probe/run_finetune` |
| Repeated historical test access | POSSIBLE | output files exist, but Git/files cannot prove human behavior |
| Class weights/sampling before split | PREVENTED | weights computed from train after split |
| Augmentation before split | NOT APPLICABLE | no training augmentation in benchmark workflows |
| Noise sources cross a noise train/test split | NOT APPLICABLE to test-only corruption | noise is not used for training; ESC-50 folds are pooled |
| Same noise realization paired across models | CONFIRMED ISSUE for saved PANNs comparison | different build/manifest seeds and different selected sources |
| Cached features use identical preprocessing | CONFIRMED ISSUE for CNN/CRNN result provenance | recorded hashes differ from current arrays and from each other |
| Filename/path enters representation | PREVENTED | paths are metadata only; feature functions receive waveform arrays |

The grouping decision is strong protection against pitch leakage, but it is not a lock: rerunning Step 3 deterministically overwrites `splits.csv`. Fingerprint checks detect downstream mismatch; they do not prevent a deliberate rerun. **VERIFIED FROM CODE.**

## 9. Feature-pipeline findings

### Handcrafted SVM features

The representation has 88 clip-level values:

- 20 MFCC coefficients summarized by mean and standard deviation: 40;
- 12 chroma bins summarized by mean and standard deviation: 24;
- spectral centroid, bandwidth, and rolloff summarized by mean and standard deviation: 6;
- 7 spectral-contrast bands summarized by mean and standard deviation: 14;
- zero-crossing rate and RMS summarized by mean and standard deviation: 4.

Total: 88. **VERIFIED FROM CODE:** [`featurelib.py` lines 76–110](../src/instrument_robustness/featurelib.py#L76).

The current NPZ contract is `X`, `y`, `source_path`, `feature_names`, `label_names`, and fingerprint metadata. Shapes are train `(5864, 88)`, validation `(1259, 88)`, and test `(1255, 88)`; `X` is `float32`, labels span 0–11, and values are finite. **VERIFIED FROM GENERATED DATA.** No constant/near-zero-variance columns, exact duplicate columns, or near-perfectly correlated feature pairs were found in the current train array. **VERIFIED FROM GENERATED DATA.**

Scaling is fitted once on train and reused. The standardized train matrix has aggregate mean approximately `4.3e-7` and standard deviation approximately `1.0`; validation/test are not forced to zero mean. No second `StandardScaler` is applied by the SVM loader. **VERIFIED FROM GENERATED DATA; VERIFIED BY TEST.**

### Log-mel features

The CNN/CRNN representation uses 22,050 Hz mono audio, 2,048-sample FFT, 512-sample hop, 128 mel bins, 130 frames, power mel spectrograms, `power_to_db(ref=1.0, top_db=None)`, Nyquist fmax, and reflect edge padding. Per-mel-bin mean and standard deviation are fitted on train and broadcast to validation/test. **VERIFIED FROM CONFIGURATION; VERIFIED FROM CODE:** [`config.py` lines 223–268](../src/instrument_robustness/config.py#L223), [`featurelib.py`](../src/instrument_robustness/featurelib.py), and [`logmel_input.py`](../src/instrument_robustness/logmel_input.py).

Saved CNN arrays have shape `(N, 128, 130, 1)` and finite `float32` values; the train array has global mean approximately zero and standard deviation approximately one. **VERIFIED FROM GENERATED DATA.** Model loaders transpose to channel-first tensors where needed. **VERIFIED FROM CODE.**

However, the current local CNN feature hashes do not equal the hashes recorded by either the CNN or CRNN validation summaries, and the CNN and CRNN recorded hashes also differ from each other. Their finalizers intentionally reject such mismatches. **VERIFIED FROM GENERATED DATA; VERIFIED FROM CODE.** Exact original arrays are not locally available, so these validation runs cannot currently be finalized reproducibly.

### Pretrained inputs

| Model | Input path | Rate/duration handling | Representation | Checkpoint control |
|---|---|---|---|---|
| AST | normalized 3 s waveform through Hugging Face extractor | resampled/processed at 16 kHz; processor pads/truncates | fine-tuned AST classifier | model name configured, revision not pinned |
| MERT | normalized 3 s waveform | resampled to 24 kHz | cached `(13,768)` hidden-layer/time-mean embeddings, learned layer mixture + linear head | exact model revision pinned |
| PANNs | normalized waveform dataset | resampled to 32 kHz | CNN14 embeddings or full fine-tuning | base checkpoint presence required, but source hash not enforced in training output |

**VERIFIED FROM CODE; VERIFIED FROM CONFIGURATION:** `ast_data.py`, `extract_mert.py`, `mert_probe.py`, `train_panns.py`, and `config.py` lines 270–275. Saved MERT summaries report expected arrays `(5864,13,768)`, `(1259,13,768)`, and `(1255,13,768)`, but those arrays were not available locally for independent verification. **VERIFIED FROM GENERATED DATA; UNRESOLVED.**

## 10. Model findings

### Inventory

| Model | Status | Input | Main implementation | Training entry point | Validation method | Saved artifact | Main concerns |
|---|---|---|---|---|---|---|---|
| SVM | IMPLEMENTED | 88-D standardized features | `svm_model.py` | `train_svm.py` | macro-F1 grid search | `artifacts/svm/final_model.joblib` | must rerun after data correction |
| CNN | PARTIALLY IMPLEMENTED | 128×130 log-mel | `cnn_model.py` | `train_cnn.py` | 5 seeds, best val loss per seed, combiner selected on macro-F1 | seed checkpoints/summary | saved summary predates metric change and has stale feature hashes; no final test/noise |
| CRNN | PARTIALLY IMPLEMENTED | 128×130 sequence | `crnn_model.py` | `train_crnn.py` | 5 seeds, best val loss per seed, combiner selected on macro-F1 | seed checkpoints/summary | same plus architecture-capacity confound |
| AST | IMPLEMENTED | 16 kHz processor tensor | `train_ast.py` | same | best validation balanced accuracy/MCC/accuracy | external canonical checkpoint + tracked summary | test evaluated inside training; version provenance |
| MERT | IMPLEMENTED | 13×768 cached embeddings | `mert_probe.py` | `train_mert.py` | val macro-F1 over LR/batch, early stopping | `final_probe.pt` | single seed; external embeddings |
| PANNs | IMPLEMENTED | 32 kHz waveform/2048-D embedding | `train_panns.py` | same | val macro-F1, early stopping | release checkpoint + results | test evaluated inside training; base checkpoint provenance |

### SVM

The active baseline is `sklearn.svm.SVC` with RBF kernel, default one-vs-one multiclass handling, `probability=False`, and `class_weight=None`. It does not re-standardize features. Candidate metrics are retained in CSV, but candidate estimators are not all saved; only the selected validation model and final refit are persisted. **VERIFIED FROM CODE; VERIFIED BY TEST.**

The saved validation search used `C ∈ {10,100,1000}` and `gamma ∈ {0.0001,0.0003,0.001,0.003}` after a documented one-time boundary extension, selecting `C=10`, `gamma=0.003` by validation macro-F1 0.959107. The finalizer refit on train+validation and evaluated test once, recording test macro-F1 0.991446 and accuracy 0.992032. **VERIFIED FROM GENERATED DATA.** These are technically complete clean results for that build, but not final paper results until DATA-001 is resolved.

### CNN

`MediumCNN` contains three convolution/pooling blocks (32/64/128 channels), global average pooling, a 128-unit dense layer, dropout 0.4, and 12 logits (110,956 parameters recorded). It uses conditionally class-weighted cross-entropy, AdamW at `1e-3`, batch 32, maximum 40 epochs, ReduceLROnPlateau, patience 8, best validation-loss weights, and seeds 42–46. Evaluation uses `eval()` and `no_grad()`. The current trainer selects soft/hard ensemble voting by validation macro-F1 and records balanced accuracy/MCC alongside it. **VERIFIED FROM CODE.** The saved summary predates that change: its historical mean balanced accuracy 0.952298 ± 0.008195 and hard-vote balanced accuracy 0.958986 are now explicitly marked `STALE` by `summarize_results`. **LEGACY; VERIFIED FROM GENERATED DATA.** No official test or noise results exist, and current feature hashes do not reproduce the saved run.

### CRNN

The CRNN applies convolutional feature extraction followed by a bidirectional GRU with hidden size 128, time-mean pooling, dropout 0.4, and a 12-way head (294,124 parameters recorded). It reuses the current CNN trainer exactly—AdamW `1e-3`, batch 32, maximum 40 epochs, patience 8, best validation loss, class weights, seeds 42–46, and macro-F1 combiner selection. **VERIFIED FROM CODE.** Its saved historical mean balanced accuracy 0.959805 ± 0.005648 and hard-vote balanced accuracy 0.965291 are pre-standardization and now marked `STALE`. **LEGACY; VERIFIED FROM GENERATED DATA.** The higher capacity and temporal recurrence differ from the CNN, so a performance difference is not attributable to recurrence alone. **INFERRED.** No official test/noise result exists, and exact feature provenance conflicts.

### AST

AST fine-tunes `MIT/ast-finetuned-audioset-10-10-0.4593` with 12 outputs, weighted cross-entropy, AdamW, default 10 epochs, batch size 8, learning rate `1e-5`, and seed 0. Checkpoint selection is lexicographic validation balanced accuracy, MCC, then accuracy—not project-primary macro-F1. **VERIFIED FROM CODE.** The canonical saved test record reports macro-F1 0.991680 and accuracy 0.992829. **VERIFIED FROM GENERATED DATA.**

The same `train()` function creates a test loader before training and evaluates test immediately after selecting the best checkpoint; there is no sealed `finalize_ast` workflow. **VERIFIED FROM CODE:** [`train_ast.py` lines 269–421](../src/instrument_robustness/train_ast.py#L269). The current result names external checkpoint SHA-256 `25789685…`. The newly added `models/ast_finetuned.safetensors` instead hashes to `3133ad96…` and is copied from the stale older-build `artifacts/ast/model.safetensors`; it cannot reproduce the reported current AST result. **VERIFIED FROM GENERATED DATA.**

### MERT

MERT uses the frozen, revision-pinned 95M encoder; hidden states are time-averaged into 13×768 examples. `MERTProbe` learns a normalized softmax mixture over layers and a linear 12-class head. Candidate training uses class-weighted cross-entropy, Adam, batch 64, maximum 100 epochs, patience 10, seed 0, validation macro-F1, and learning rates `{0.0001, 0.0005, 0.001, 0.005, 0.01}`. **VERIFIED FROM CODE; VERIFIED FROM CONFIGURATION.** The selected saved run uses learning rate 0.001 and best epoch 92; validation macro-F1 is 0.903796. Its final train+validation refit evaluated 1,255 test examples once, with macro-F1 0.924598 and accuracy 0.925896. **VERIFIED FROM GENERATED DATA.**

### PANNs

PANNs supports a frozen CNN14 probe and full fine-tuning, class-weighted cross-entropy, AdamW with weight decay `1e-4`, early stopping by validation macro-F1, patience 10, batch 16, four workers, and seed 0. Defaults are 200 epochs/`1e-3` for the linear probe and 25 epochs/`1e-4` for full fine-tuning. **VERIFIED FROM CODE.** The current full-fine-tune record reports validation macro-F1 approximately 0.989 and test macro-F1 approximately 0.9845. **VERIFIED FROM GENERATED DATA.**

Both modes load the test split at the start; both evaluate test inside the same run that trains/selects the model. The output paths are overwritten unless manually separated, and no one-time finalization guard exists. **VERIFIED FROM CODE:** [`train_panns.py` lines 181–269](../src/instrument_robustness/train_panns.py#L181). The reported clean/noise result uses the external full-fine-tune checkpoint SHA-256 `00cc195e…`, while `models/` contains only the lower-performing linear probe SHA-256 `9a754a7f…`. The original AudioSet CNN14 checkpoint identity is also not recorded/enforced in the clean result. **VERIFIED FROM GENERATED DATA; VERIFIED FROM CODE.**

## 11. Noise-robustness findings

### Implemented design

Waveform-level corruption is fully implemented; noise is not added to feature vectors or spectrograms. **VERIFIED FROM CODE.** The frozen grid is:

- noise types: Gaussian white, ESC-50 natural (targets 0–19), ESC-50 mechanical (targets 30–49);
- SNR: 60, 50, 40, 30, 20, 10, 0, and −10 dB;
- two independent realizations per test recording and noise type;
- 1,255 test windows × 3 types × 8 SNRs × 2 replicates = 60,240 mixtures.

**VERIFIED FROM CONFIGURATION:** [`config.py` lines 127–220](../src/instrument_robustness/config.py#L127).

For one clean waveform `x` and selected noise segment `n`, both are treated as length-`N` sample sequences. The actual mixer uses whole-window power:

$$
P_x=\frac{1}{N}\sum_{t=1}^{N}x_t^2,
\qquad
P_n=\frac{1}{N}\sum_{t=1}^{N}n_t^2,
$$

$$
\alpha=\sqrt{\frac{P_x}{P_n 10^{s/10}}},
\qquad
y_t=x_t+\alpha n_t.
$$

**VERIFIED FROM CODE:** [`noise_sweep.mix_at_snr` lines 400–424](../src/instrument_robustness/noise_sweep.py#L400). This differs from the attachment's conceptual active-region formula: headline mixing uses all samples, not only an active set. Active-band, instrument-band, octave, segmental, and model-effective SNRs are diagnostics recorded alongside the nominal whole-window SNR. **VERIFIED FROM CODE.**

Noise behavior:

- Gaussian samples and ESC-50 crops have their sample mean removed before scaling. **VERIFIED FROM CODE.**
- ESC-50 is decoded mono, resampled to 22.05 kHz, tiled if short, randomly cropped deterministically, and rejected if centered RMS is too low. **VERIFIED FROM CODE.**
- The stable seed is SHA-256 of dataset fingerprint, window ID, noise type, and replicate. SNR is deliberately excluded so the same realization is only rescaled along its curve. **VERIFIED FROM CODE; VERIFIED BY TEST.**
- ESC-50 source file, source hash, original category/target/fold, resampled crop offset, requested/achieved SNR, diagnostics, clean/noisy hashes, and build identity are recorded. **VERIFIED FROM CODE.**
- Generation fails if the destination contains an existing/incomplete sweep, writes atomically, reloads each FLOAT WAV, verifies SNR within 0.1 dB, and validates the completed manifest. **VERIFIED FROM CODE; VERIFIED BY TEST.**
- The mixer intentionally does not clip or renormalize after addition. FLOAT WAV permits samples outside `[-1,1]`. **VERIFIED FROM CODE.** This preserves mathematical SNR but creates a model-input amplitude/headroom condition that has not been shown equivalent across all pretrained processors.
- ESC-50 folds are pooled because noise is used only to corrupt test examples and never to train models. That avoids model-side noise leakage, although some noise sources recur across examples/conditions; statistical code supports clustering by noise source. **VERIFIED FROM CODE.**

### Critical saved-corpus mismatch

Compact noise outputs show:

| Model result | Dataset fingerprint | Windows hash relationship | Noise-manifest relationship |
|---|---|---|---|
| SVM | `09b65f…` | shared with MERT/AST | shared with MERT/AST |
| MERT | `09b65f…` | shared with SVM/AST | shared with SVM/AST |
| AST | `09b65f…` | shared with SVM/MERT | shared with SVM/MERT |
| PANNs | `89f126…` | different; matches current local `windows.csv` | different |

**VERIFIED FROM GENERATED DATA.** Window IDs match, but direct comparison for `natural_20_r0` found the same ESC-50 `noise_source` for 100% of SVM/MERT/AST rows and 0% between PANNs and any of those models. Another condition (`mechanical_0_r1`) matched only 0.159% by chance. Because dataset fingerprint participates in the seed, this necessarily changes white samples and environmental clip/crop selections. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.**

Therefore, the saved PANNs robustness curve is not paired with the other models on identical corrupted waveforms. The repository's same-mixture claim is false for the saved result set, even though the implementation would satisfy it if all adapters consumed one manifest. **CONFIRMED ISSUE.**

The SVM official clean feature hashes match the local current feature files, while its noise run records the older `09b65f…` build and recomputed clean macro-F1 differs from the official clean result by 0.001020 (accepted by the 0.002 cross-platform tolerance). This suggests cross-environment drift and means the exact waveform build used for noisy evaluation is not proven identical to the feature build used for official clean training. **VERIFIED FROM GENERATED DATA; scientific magnitude UNRESOLVED.**

DEMAND, MUSAN, UrbanSound8K, speech, music interference, impulse responses/reverberation, and adversarial noise are not implemented. ESC-50 human non-speech targets 20–29 are excluded. **VERIFIED FROM CODE.** This is a scope limitation, not an error if the paper claims only the implemented conditions.

## 12. Evaluation and statistical findings

The shared evaluation path computes full-test accuracy, macro-F1, weighted F1, fixed-label classification reports, confusion matrices, per-instrument outputs, and per-family summaries. **VERIFIED FROM CODE.** Macro-F1 calls explicitly pass all 12 label indices, preventing bootstrap samples missing a class from redefining the metric. **VERIFIED FROM CODE; VERIFIED BY TEST.**

The robustness summary records absolute metrics, degradation from clean, and clean-F1 retention:

$$
R_{m,c,s,r}=\frac{F_{1,m,c,s,r}}{F_{1,m,\mathrm{clean}}},
\qquad
\Delta F_{1,m,c,s,r}=F_{1,m,\mathrm{clean}}-F_{1,m,c,s,r}.
$$

**VERIFIED FROM CODE:** `noise_eval_common.py` and `robustness_curve.py`.

Statistical infrastructure includes cluster bootstrap macro-F1, explicit cluster units (pitch group and noise source where available), paired cluster-level sign tests, optional ordinary McNemar tests, replicate spread, SNR-curve AUC normalized by range, threshold interpolation, and Benjamini-Hochberg multiple-comparison correction. **VERIFIED FROM CODE; VERIFIED BY TEST.** The cluster-level test is a sign test, not a clustered McNemar test. **VERIFIED FROM CODE.**

Scientific limitations:

- Only two noise realizations are available per recording/type. This permits a sensitivity check but is too small for a precise noise-realization variance estimate. **VERIFIED FROM CONFIGURATION; INFERRED.**
- SVM, MERT, AST, and PANNs saved clean results are single-seed/fixed-training outcomes; CNN/CRNN use five seeds but are validation-only. **VERIFIED FROM GENERATED DATA.**
- No final confidence-interval/significance/AUC comparison outputs were found. The implementation exists but has not been applied to a valid complete six-model result set. **VERIFIED FROM GENERATED DATA.**
- Cross-model pairing is valid only when the per-example clean/noisy rows and corpus hashes match. It is invalid for the current saved PANNs-vs-other comparison. **VERIFIED FROM GENERATED DATA.**
- The 12 per-class test counts (65–129) support descriptive class metrics, but some class-level differences will have wide uncertainty; report intervals rather than rankings from point estimates alone. **INFERRED.**
- Calibration metrics are absent. This does not block a classification/robustness paper unless probability calibration is claimed. **VERIFIED FROM CODE.**
- Current SVM/MERT/PANNs/CNN/CRNN code selects with validation macro-F1, but AST still selects lexicographically by validation balanced accuracy, MCC, then accuracy. The new commit/documentation claim that “all six” models select on macro-F1 therefore disagrees with `train_ast.py`. Saved CNN/CRNN summaries also predate the current metric and are correctly rejected as stale. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.**

## 13. Reproducibility findings

### Strong controls

- Authoritative label order and load-bearing preprocessing constants are included in fingerprints. **VERIFIED FROM CODE.**
- Each pipeline stage verifies its input sidecar and stamps output content/stage provenance. **VERIFIED FROM CODE; VERIFIED BY TEST.**
- Splits are deterministic under fixed input/configuration/seed, and group isolation is asserted twice. **VERIFIED FROM CODE; VERIFIED BY TEST.**
- SVM and MERT enforce separated validation selection and one-time test finalization with completion records. **VERIFIED FROM CODE; VERIFIED BY TEST.**
- Noise seeds, source selection, crop offsets, realized audio hashes, and manifest validation are designed to be deterministic and fail closed. **VERIFIED FROM CODE; VERIFIED BY TEST.**

### Weak controls

- Python dependencies are mostly ranges or unpinned names; no lock file exists. Audio/library defaults can therefore alter decoded samples, trimming, resampling, and pretrained processing. **VERIFIED FROM CONFIGURATION.**
- AST's Hugging Face model revision is not pinned; MERT's is pinned. PANNs training does not record/enforce the original CNN14 base-checkpoint hash. **VERIFIED FROM CONFIGURATION; VERIFIED FROM CODE.**
- Result summaries inconsistently record Git commit, exact command, package versions, hardware, CUDA/cuDNN state, and deterministic-kernel settings. **VERIFIED FROM GENERATED DATA.**
- Output isolation varies: SVM/MERT finalizers guard overwrites; AST requires a fresh directory but performs test in training; PANNs writes fixed output names and may overwrite. **VERIFIED FROM CODE.**
- The tracked `model_bundle/` source snapshot is stale: `bundle_models --check` reports 42 changed source entries. **VERIFIED BY TEST.**
- The new `models/` weight-copy manifest is internally consistent with its selected source files (`bundle_weights --check` passes for 16 files), but it does not check whether those sources are the weights named by canonical result summaries. It consequently passes while bundling the wrong AST/PANNs variants and stale CNN/CRNN weights. **VERIFIED BY TEST; VERIFIED FROM GENERATED DATA.**
- CNN/CRNN summaries name absolute paths and hashes for exact arrays that are unavailable locally and differ from current data. **VERIFIED FROM GENERATED DATA.**
- The canonical AST summary points to an external SCC model path/hash; the exact checkpoint remains absent locally despite the different AST file now present under `models/`. **VERIFIED FROM GENERATED DATA.**
- Absolute personal/project paths in summaries are useful forensic provenance but are not portable commands. **VERIFIED FROM GENERATED DATA.**

Another researcher cannot currently reproduce the full reported six-model table from one documented command. The pipeline can be reproduced in stages, but required external checkpoints/data, missing SCC wrappers, conflicting result builds, and incomplete model finalization prevent end-to-end reproduction. **INFERRED.**

## 14. Testing findings

### Results

| Command | Result |
|---|---|
| `.venv/bin/python -m unittest tests.test_preprocessing tests.test_svm -q` | 14 passed |
| `.venv/bin/python -m unittest tests.test_noise tests.test_noise_metrics tests.test_noise_adapters tests.test_robustness_curve -q` | 113 run; 112 passed, 1 skipped |
| `.venv/bin/python -m unittest tests.test_ast tests.test_mert -q` | collection error because module-level AST `SkipTest` was treated as an error in this direct multi-module invocation |
| `.venv/bin/python -m unittest tests.test_mert -q` | 14 run; 11 passed, 3 skipped |
| `.venv/bin/python -m unittest discover -s tests -q` | **142 run items; 137 passed, 5 skipped** in 3.196 s after updating to `417c57d` |
| `.venv/bin/python -m instrument_robustness.audio_inventory --check` | 8,378 files verified; inventory digest matched |
| `.venv/bin/python -m instrument_robustness.bundle_models --check` | failed as designed: 42 source entries differ from the stored bundle |
| `.venv/bin/python -m instrument_robustness.bundle_weights --check` | passed: 16 copied files match their declared source files (but canonical-result identity is not checked) |

**VERIFIED BY TEST.** Discovery is the reliable aggregate test command. The direct AST+MERT invocation exposes awkward module-level skip behavior, but discovery completed successfully. AST tests and three PyTorch-dependent MERT tests were among the skips because the local environment lacks the required runtime; therefore a green discovery run does not mean AST model execution was exercised. Test audio and outputs were confined to temporary directories; no repository dataset was regenerated.

### Test inventory

Every discovered test is summarized below by file. Names are listed to make explicit what was and was not exercised.

| Test file and names | What it guarantees | Real data? | Current result | Main gap |
|---|---|---:|---|---|
| `test_preprocessing.py`: `test_group_assignment_is_deterministic_and_leak_free`, `test_leak_verifier_rejects_a_group_in_two_splits`, `test_short_signal_is_tiled_to_exact_length`, `test_window_writer_tiles_short_source`, `test_window_writer_drops_tiny_trailing_remainder`, `test_fingerprint_covers_load_bearing_pipeline_settings`, `test_sidecar_rejects_wrong_pipeline_stage`, `test_sidecar_rejects_changed_artifact_content`, `test_main_writes_manifest_and_its_fingerprint` | Split grouping, tiling, fingerprint integrity | Synthetic/temp | Pass | No trim attack/decay or cross-label duplicate test |
| `test_svm.py`: `test_loader_does_not_standardize_features_again`, `test_loader_rejects_an_unfingerprinted_split`, `test_grid_and_model_are_rbf_only`, `test_training_saves_ranked_results_without_test_split`, `test_finalization_refits_train_and_val_and_only_runs_once` | SVM contract and sealed test workflow | Synthetic/temp | Pass | No large real-array training test by design |
| `test_ast.py`: 15 tests covering stable labels, dataset indices, stale labels, missing classes/provenance/audio, one-window constraints, fresh output directory, sample count, imbalance metrics/weights, 12-way head, reports/families | AST input/model/report contract | Synthetic/temp | Skipped as module in current environment | No executed AST training/inference here; no test-once finalizer exists |
| `test_mert.py`: 14 tests covering authoritative splits, resampling, 12-label preflight, embedding shape/labels, class weights, no-test extraction, overwrite guards, one-time finalization, candidate training, logits/layer weights, dataset identity | MERT extraction/training/finalization contract | Mostly synthetic/temp | 11 pass, 3 dependency-skipped in aggregate | Cached real embeddings absent locally |
| `test_noise.py`: 22 tests covering white/ESC provenance, category/fold, centered crops, DC rejection, seed scope, power SNR, FLOAT headroom, build identity, fail-closed manifest, grid diagnostics, shared runner, pitch groups, clean parity, SVM stats, fixed labels, pairing, cluster reproducibility | Core generation/evaluation correctness | Synthetic/temp | Pass | Does not prove current external 60,240-file corpus is present/identical |
| `test_noise_metrics.py`: 30 tests covering frozen grid, band bounds, Parseval bands, tones, in/out-of-band SNR, octave profiles, active/segmental/model-effective metrics, provenance columns, replicate seeds/directories/grid, end-to-end two-replicate generation | Diagnostic math and replicate behavior | Synthetic/temp | Pass | No perceptual/model clipping-response test |
| `test_noise_adapters.py`: 26 tests covering log-mel shape/standardization/cache parity, ensemble voting, SNR pilot selection, validation-only pilot access, MERT checkpoint/schema/summary guards | Adapter contracts and pilot isolation | Synthetic plus optional local fixture | Pass with optional skip | CNN/CRNN/AST/PANNs production inference not all exercised |
| `test_robustness_curve.py`: 35 tests covering AUC, retention thresholds, BH correction, sweep schema, clean requirement, replicate averaging/spread/completeness/pairing, cluster units, provenance lookup, and audio inventory digests | Statistical summaries and integrity | Synthetic/temp | Pass | No final real-result statistical analysis |

The source defines 156 test methods. Because `test_ast.py` raises a module-level dependency skip, discovery records that entire 15-method module as one skipped module placeholder; it therefore reports 142 run items (141 non-AST methods plus that placeholder), with 5 skips in total. **VERIFIED BY TEST; VERIFIED FROM CODE.**

### Important missing coverage

- No regression test rejects exact audio reused under different labels. **VERIFIED FROM CODE.**
- No CNN/CRNN training/finalizer tests verify checkpoint-to-feature hashes end to end. **VERIFIED FROM CODE.**
- No sealed AST/PANNs test-finalization tests exist because the production workflows do not implement that boundary. **VERIFIED FROM CODE.**
- No PANNs clean-training tests or real AST/PANNs inference smoke test ran locally. **VERIFIED BY TEST.**
- No test compares dataset/noise-manifest identity across every model's saved results before a paper table is produced. **VERIFIED FROM CODE.**
- No CI runs the suite in a controlled environment. **VERIFIED FROM GENERATED DATA.**

## 15. Code-quality findings

Strengths include explicit pre/postconditions, fingerprint validation, fail-closed manifest checks, atomic noise writes, fixed label-order checks, and small reusable feature/noise-statistic functions. **VERIFIED FROM CODE.**

Risks and maintainability observations:

- `noise_sweep.py` (~1,028 lines), `snr_pilot.py` (~592), `train_ast.py` (~522), `train_mert.py` (~476), `robustness_curve.py` (~476), `noise_metrics.py` (~457), and `noise_eval_common.py` (~449) concentrate multiple responsibilities. **VERIFIED FROM CODE.** This is maintainability risk, not evidence of incorrect results.
- PANNs model construction/evaluation logic is duplicated between clean and noise workflows; AST has model-specific metric/report logic alongside common infrastructure. **VERIFIED FROM CODE.**
- Several pipeline/feature/noise modules use broad `warnings.filterwarnings("ignore")`, which can hide codec or numerical warnings relevant to reproducibility. **VERIFIED FROM CODE.**
- Configuration is split between Python constants, CLI defaults, SCC environment variables, and YAML files that are partly documentary. This makes drift possible; the saved SVM run already differs from the default YAML grid because of a documented grid extension. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.**
- Most data paths are centralized under `RISE_DATA_ROOT`; some SCC/docs/artifact paths are machine-specific. **VERIFIED FROM CODE.**
- Type annotations and docstrings are strong in newer noise modules but sparse/inconsistent in older preprocessing/training modules. **VERIFIED FROM CODE.**
- No circular-import problem, swallowed fatal data error, or secret-bearing global state was found. **VERIFIED FROM CODE/STATIC SEARCH.**

Large refactoring is not required before experiments. The high-value work is enforcing result/data identity, sealing test access, and adding narrow regression checks.

## 16. Dependency and environment findings

- `pyproject.toml` requires Python `>=3.9`. **VERIFIED FROM CONFIGURATION.**
- Core dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, `librosa`, `soundfile`, `mutagen`) are unpinned. **VERIFIED FROM CONFIGURATION.**
- MERT pins `transformers>=4.38,<4.39` and pins the model revision, but does not pin PyTorch. AST's `transformers` and model revision are unpinned. PANNs/torch/torchaudio/torchlibrosa packages are unpinned. **VERIFIED FROM CONFIGURATION.**
- No lock file or fully resolved environment export exists. **VERIFIED FROM GENERATED DATA.**
- `matplotlib` is imported by `eval_panns_probe.py` but is not declared. `joblib` is imported directly but only arrives transitively through scikit-learn. **VERIFIED FROM CODE; VERIFIED FROM CONFIGURATION.**
- MP3 decoding depends on the local audio stack used by librosa/audioread/soundfile. The test run emitted deprecation warnings about legacy `aifc`/`sunau` paths for future Python versions. **VERIFIED BY TEST.**
- GPU workflows depend on CUDA-compatible PyTorch and model downloads/caches; hardware/software versions are not captured consistently in clean result summaries. **VERIFIED FROM CODE; VERIFIED FROM GENERATED DATA.**

Changing librosa, resampling backends, decoders, NumPy, PyTorch, transformers, or pretrained weights can alter byte-level features or predictions. Current fingerprinting covers configuration/content, not the entire software environment. **INFERRED.**

## 17. Security and repository-hygiene findings

- Targeted static searches found no API key, access token, password, private key, or tracked `.env` secret. `.env.example` contains only examples/comments. **VERIFIED FROM CODE.**
- Personal absolute paths occur in SCC documentation, result summaries, and the environment example. They reduce portability and reveal usernames/project layout, but not credentials. **VERIFIED FROM GENERATED DATA.**
- `artifacts/ast/model.safetensors` is a stale ~345 MB tracked generated checkpoint and is duplicated into Git-LFS-managed `models/ast_finetuned.safetensors`; both share hash `3133ad96…`, which does not match the current AST result. Many smaller model outputs are also tracked/copied. **VERIFIED FROM GENERATED DATA.**
- Artifact policy is inconsistent: SVM/MERT local noise outputs appear as untracked and unignored, while AST/PANNs noise outputs are tracked. **VERIFIED FROM GENERATED DATA; VERIFIED FROM CONFIGURATION.**
- `.vscode/settings.json` is tracked, coupling editor preferences to the repository. **VERIFIED FROM GENERATED DATA.**
- No raw Philharmonia/ESC-50 audio dataset was found tracked in Git. **VERIFIED FROM GENERATED DATA.**
- No notebook outputs, OS metadata, or obvious temporary editor files were found tracked. **VERIFIED FROM GENERATED DATA.**

## 18. Documentation findings

| Documentation | Classification | Finding |
|---|---|---|
| `README.md` project/data pipeline and 12 labels | ACCURATE | Current one-window pipeline, authoritative counts, and main clean workflows largely match code |
| `README.md` noise/model commands | INCOMPLETE | Lists some adapters but omits CNN, CRNN, and AST workflow details; PANNs training is not documented end to end |
| `docs/DATA_PIPELINE_AND_NOISE.md` terminology/current sections | MIXED | Contains useful definitions and current material, but also old 1,310-example counts, 15-condition grid, false claims that ESC category/fold are ignored, old model scores, zero-padding behavior, and already-resolved TODOs |
| `docs/NOISE_PLAN.md` | MOSTLY ACCURATE | Correct whole-power design and diagnostic rationale; some peak examples are historical and corpus identity across saved outputs is not addressed |
| `docs/AUDIT_CHECKLIST.md` | ACCURATE/INCOMPLETE | Tracks many known risks, including test access and metric mismatch; does not include the cross-label duplicate or cross-model corpus mismatch found here |
| `docs/RESULTS.md` | ACCURATE BUT INCOMPLETE | Correctly marks pre-standardization CNN/CRNN summaries stale; currently tables PANNs noise only and does not validate the new weight bundle against result checkpoint hashes |
| `docs/REPO_MAP.md` | MOSTLY ACCURATE | Auto-generated inventory improves navigation; some descriptions overstate YAML consumption and the `models/` bundle's scientific completeness |
| `plan.md`, `plan_noise.md` | MOSTLY ACCURATE | Provide useful run order and plain-language noise protocol; they do not resolve the canonical weight/corpus mismatches found here |
| `docs/FINDINGS.md` | LEGACY | Explicitly carries cnn-ensemble history, old counts/commands/modules, and should not be used as current evidence without qualification |
| `CLAUDE.md` evaluation guidance | CONTRADICTORY | Says to report balanced accuracy/MCC and “never” accuracy or macro-F1, conflicting with the project's stated macro-F1 primary metric and current evaluators |
| `step4_window.py` generated report text | CONTRADICTORY | Lines 148–150 repeat the claim that macro-F1 should not be reported |
| `scc/README.md` | INCOMPLETE/OUTDATED | References missing `train_ast.qsub`, uses conflicting personal/shared roots, and lacks a complete reproducible six-model sequence |
| `artifacts/panns/README.md` | OUTDATED | References absent `instrument_robustness.cross_dataset_eval` and absent `scc/cross_dataset_eval.qsub` |
| `configs/data/irmas.yaml` | STALE | Zero-byte placeholder with no current consumer |
| `pretrained_extractors.py` | STALE COMMENT | Refers to a “new 9-way head” although the executable output size is 12 |

**VERIFIED FROM CODE.**

No `CONTRIBUTING.md` or CI troubleshooting guide exists. `plan.md` now provides a canonical run order, but the repository still lacks a demonstrated one-command reproduction of the final six-model comparison. **VERIFIED FROM GENERATED DATA.**

The README correctly does not claim that the missing `all-samples/pipeline/pipeline_report.txt` exists, but `.gitignore` still re-includes that old path and only `_step4_report_block.txt` is present locally. **VERIFIED FROM GENERATED DATA; VERIFIED FROM CONFIGURATION.**

Documentation needed for a reproducible paper includes: a frozen experiment/build ID shared by all model outputs; exact dependency/container information; exact canonical checkpoint locations/hashes; complete clean and noisy commands for all six models; a statement of how cross-label duplicate files were resolved; one canonical result table provenance; and a clear distinction between nominal whole-window SNR and diagnostic in-band/active/model-effective SNR. **INFERRED.**

## 19. Paper-readiness assessment

| Paper subsection | Readiness | Exact qualification |
|---|---|---|
| Dataset and instrument classes | NEEDS VALIDATION | Counts/labels are known, but two exact cross-label audio pairs must be resolved |
| Audio preprocessing | MOSTLY READY | Decode/resample/normalization are explicit; software versions and codec backend must be frozen |
| Silence handling | MOSTLY READY | Current algorithm and measured activity are clear; attack/decay retention and repeated-content implications need explicit limitation text |
| Source-level partitioning | READY FOR METHODS | Pitch-group assignment and current zero-leak result are verified |
| Windowing | READY FOR METHODS | One first 3 s window, tiling, and no overlap are explicit; repeated-content tradeoff must be stated |
| Acoustic representations | MOSTLY READY | SVM/log-mel definitions are exact; CNN/CRNN saved-input provenance conflicts |
| Models | NEEDS VALIDATION | Architectures are documentable; final CNN/CRNN results and sealed AST/PANNs workflow are incomplete |
| Noise conditions | MOSTLY READY | Three types/eight SNRs/two realizations are exact; scope exclusions must be clear |
| SNR-controlled mixing | NEEDS VALIDATION | Math/provenance are strong, but amplitude headroom and missing local full-corpus audit remain |
| Evaluation metrics | MOSTLY READY | Uniform macro-F1/report infrastructure exists; historical documentation/selection metrics conflict |
| Reproducibility | BLOCKED | Exact CNN/CRNN inputs and AST checkpoint are unavailable; dependencies/run environment not locked |
| Statistical analysis | BLOCKED | Code exists, but complete paired six-model results and final outputs do not |

### Results readiness

The repository can support preliminary, clearly qualified clean results for SVM, MERT, AST, and PANNs. It cannot support a final six-model ranking or paired robustness comparison. **INFERRED.** The final Results section is blocked by DATA-001, NOISE-001, missing CNN/CRNN final outputs, and absent final statistical products.

## 20. Prioritized issue register

| ID | Severity | Confidence | Status | Area | Finding | Evidence | Scientific impact | Recommended action |
|---|---|---|---|---|---|---|---|---|
| DATA-001 | CRITICAL | HIGH CONFIDENCE | CONFIRMED | Labels/data | Two exact audio pairs have conflicting instrument labels; four retained train rows are affected | Raw/window SHA-256 grouping; current manifests | Trains contradictory labels and corrupts the dataset contract | Trace upstream provenance, correct/exclude affected sources, rebuild all dependent artifacts |
| NOISE-001 | CRITICAL | HIGH CONFIDENCE | CONFIRMED | Noise/comparison | Saved PANNs robustness outputs use a different build/noise manifest and different realized mixtures than SVM/MERT/AST | Result hashes; direct `noise_source` comparison | Invalidates paired cross-model PANNs robustness claims | Freeze one canonical build/manifest and reevaluate every model on exactly that corpus |
| REPRO-001 | HIGH | HIGH CONFIDENCE | CONFIRMED | CNN/CRNN | Saved CNN/CRNN validation summaries do not hash to current feature arrays or each other; finalizers cannot reproduce/finalize them | Summary input hashes vs current NPZ hashes | Blocks reliable CNN/CRNN completion and comparison | Recover exact arrays or retrain both from one frozen build |
| MODEL-001 | HIGH | HIGH CONFIDENCE | CONFIRMED | AST/PANNs | Training entry points load/evaluate test; no sealed one-time finalizers | `train_ast.train`; `train_panns.run_probe/run_finetune` | Makes test-isolation history hard to enforce and invites iterative test use | Separate validation selection from hash-guarded finalization before rerun |
| REPRO-002 | CRITICAL | HIGH CONFIDENCE | CONFIRMED | Weight bundle | Canonical-looking `models/` bundles AST hash `3133ad96…` instead of reported `25789685…`, and a PANNs probe instead of the reported/noise-tested fine-tune `00cc195e…` | `models/MANIFEST.json`, result summaries, model/noise hashes | The advertised six-model weight handoff cannot reproduce the reported AST/PANNs results and can silently evaluate different models | Replace bundle entries with exact result-named weights; gate bundle hashes against canonical summaries, not only source copies |
| EVAL-001 | HIGH | HIGH CONFIDENCE | CONFIRMED | Completion | CNN and CRNN have validation-only outputs and no official clean test/noise results | Artifact inventory | Six-model benchmark and conclusions are incomplete | Resolve provenance, finalize once, evaluate same noise corpus |
| DATA-002 | HIGH | MEDIUM CONFIDENCE | LIKELY | Scientific validity | Register, technique/family, duration/repetition, codec/session cues may support in-collection shortcuts | Metadata distributions and pipeline design | High clean accuracy may overstate general instrument recognition | Narrow claims; add cross-collection/controlled analyses if available |
| NOISE-002 | HIGH | MEDIUM CONFIDENCE | POSSIBLE | Noise/model input | FLOAT mixtures can greatly exceed `[-1,1]`; processor response to that amplitude range is not validated uniformly | `_write_wav_atomic`, no clipping; historical validation peaks | Model rankings could partly reflect frontend amplitude behavior | Pre-register amplitude policy; test frontend equivalence/sensitivity before final rerun |
| STAT-001 | HIGH | HIGH CONFIDENCE | CONFIRMED | Statistics | Only two noise realizations, mostly single model seeds, and no final CI/significance outputs | Config/result inventory | Uncertainty may be too weak for fine-grained ranking claims | Use cluster-aware intervals, paired analyses, replicate spread, restrained claims |
| REPRO-003 | HIGH | HIGH CONFIDENCE | CONFIRMED | Environment | Major dependencies and AST/PANNs pretrained identities are not fully pinned | `pyproject.toml`, config, checkpoint records | Exact features/predictions may change across environments | Freeze lock/container, model revisions, base checkpoint hashes |
| MODEL-002 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Selection metric | Current code standardizes five families on macro-F1, but AST still selects on balanced accuracy/MCC/accuracy despite documentation claiming all six are standardized | `train_ast.py`, current trainers, commit/docs | Leaves an AST selection-protocol confound and a code/documentation disagreement | Either change AST before rerun or explicitly retain/report the exception; correct the “all six” claim |
| DATA-003 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Windowing | 79.94% of windows contain under 50% distinct source duration and are tiled | `content_s` analysis | Repetition period/attacks may become a cue; examples are not 3 s of unique audio | Report it; stratify/error-check by source duration |
| DATA-004 | MEDIUM | MEDIUM CONFIDENCE | POSSIBLE | Trimming | Trim frame/hop/default reference and retained context are not explicit; attack/decay preservation unmeasured | `step2_trim.trim_one` | Version or boundary changes can alter timbral transients | Pin explicit trim parameters and validate representative attacks/decays later |
| REPRO-004 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Snapshot | `model_bundle --check` reports 42 changed source entries | Command result | Bundle cannot be trusted as current reproducibility snapshot | Regenerate only after canonical code/results freeze |
| REPRO-005 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Run metadata | Git commit, command, environment, hardware, and deterministic settings are inconsistently recorded | Saved summaries | Makes forensic reproduction costly/ambiguous | Add a common run manifest to future jobs |
| TEST-001 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Testing | AST/PyTorch paths skipped locally; no CI; no duplicate-label or complete cross-model corpus-identity gate | Test inventory/results | Critical regressions can pass the current suite | Add narrow regression tests and CI with optional GPU tier |
| DOC-001 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Documentation | Current docs contain old counts/grid/behavior and conflict on primary metrics | `DATA_PIPELINE_AND_NOISE.md`, `CLAUDE.md`, step-4 report text | Researchers may describe or run the wrong protocol | Reconcile after scientific decisions are frozen |
| DOC-002 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Reproduction | SCC/PANNs docs reference missing job scripts/source and conflicting roots | `scc/README.md`, `artifacts/panns/README.md` | Documented commands cannot reproduce all results | Supply or remove missing commands; use role-based paths |
| HYGIENE-001 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Artifacts | Stale 345 MB AST checkpoint is duplicated through Git LFS, and tracked/untracked artifact policy remains mixed | Git inventory/status | Causes repository bloat and canonical-artifact ambiguity | Define artifact policy; use one canonical release/object-store copy plus hashes |
| MODEL-003 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | PANNs | Base pretrained CNN14 hash is not enforced and clean outputs can be overwritten | `train_panns.py` | Different base weights/reruns may silently change results | Hash base checkpoint and require fresh/unique output directory |
| NOISE-003 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Provenance availability | Full noisy corpus/manifest/provenance are absent locally | Local inventory | Compact results cannot independently prove waveform/SNR/peak identity | Archive canonical manifest/provenance and inventory digest with results |
| DEP-001 | MEDIUM | HIGH CONFIDENCE | CONFIRMED | Dependencies | No lock; direct imports `matplotlib`/`joblib` are undeclared | `pyproject.toml`, import search | Setup and exact reproduction can fail or drift | Declare direct deps and create a frozen environment artifact |
| CODE-001 | LOW | HIGH CONFIDENCE | CONFIRMED | Diagnostics | Broad warning suppression is used in several pipeline modules | Source search | Relevant codec/numerical warnings may be hidden | Narrow warning filters later |
| CODE-002 | LOW | HIGH CONFIDENCE | CONFIRMED | Maintainability | Several modules are large and some pretrained logic is duplicated | Line counts/source review | Raises change/regression cost | Extract only shared identity/evaluation boundaries when touched |
| DOC-003 | LOW | HIGH CONFIDENCE | CONFIRMED | Stale text | Empty `irmas.yaml`, stale “9-way” comment, old report re-include remain | Repository/config search | Confusion, limited direct scientific impact | Remove or clearly mark after audit follow-up |
| HYGIENE-002 | LOW | HIGH CONFIDENCE | CONFIRMED | Portability/privacy | Personal absolute paths and tracked editor settings remain | Static search/Git | Portability friction and username exposure | Replace with variables; reconsider editor-file policy |
| EVAL-002 | LOW | HIGH CONFIDENCE | CONFIRMED | Metrics | Calibration is not evaluated | Evaluator search | Limits probability-confidence claims only | Add only if calibration is in paper scope |
| DATA-005 | INFORMATIONAL | HIGH CONFIDENCE | CONFIRMED | Splits | Current 544 pitch groups and all exact duplicate groups are confined to one split | Manifest/hash analysis | Positive leakage control; does not cure conflicting labels | Preserve as a build gate |
| NOISE-004 | INFORMATIONAL | HIGH CONFIDENCE | CONFIRMED | Scope | Speech, music, reverb, DEMAND, MUSAN, and UrbanSound8K are absent | Source/config search | Limits external validity but is acceptable if claims are narrow | State exclusions; do not add under deadline without design |
| SEC-001 | INFORMATIONAL | HIGH CONFIDENCE | CONFIRMED | Security | No credential or private-key material was found in targeted search | Static repository scan | No identified secret exposure | Continue using `.env.example`; avoid committing credentials |

## 21. Recommended action plan

### Before any additional training

1. Resolve DATA-001 using upstream archive provenance. Decide which label/file is wrong or exclude both ambiguous copies. This changes the authoritative manifest.
2. Rebuild/fingerprint the pipeline once from that corrected manifest; rerun exact-audio duplicate and pitch-group gates.
3. Freeze one canonical dataset build ID, dependency environment, and Git commit.
4. Recover or discard the old CNN/CRNN validation artifacts; train both from the same frozen feature arrays.
5. Implement sealed AST/PANNs validation-selection and finalization boundaries before their next official runs.

Dependency: every old checkpoint/feature/noise result is downstream of the contaminated manifest and should be treated as preliminary after step 1.

### Before generating noisy data

1. Decide and document the FLOAT/headroom policy; run a focused frontend-amplitude sensitivity test.
2. Confirm all clean final models and label/fingerprint contracts refer to the same corrected build.
3. Generate one canonical shared noise corpus; archive its manifest, provenance CSV, corpus hashes, window inventory, and achieved-SNR/peak diagnostics.
4. Do not adapt the frozen SNR grid using test results.

Dependency: noise seeds include the dataset fingerprint, so changing the data build requires a new noise corpus.

### Before final model comparison

1. Complete one-time clean finalization for all six models.
2. Run all six adapters against the same validated `noise_manifest.json`.
3. Add an automated pre-table gate requiring identical dataset, windows, noise-manifest, label-order, condition, replicate, truth, and cluster identities.
4. Report architecture/capacity/pretraining differences; do not attribute all differences to a single factor such as recurrence or pretraining.

### Before statistical analysis

1. Confirm paired rows and hashes first.
2. Compute cluster bootstrap intervals with source/pitch-group-aware units and noise-source sensitivity analyses.
3. Report both replicate spread and model-seed limitations.
4. Use the implemented paired sign/McNemar outputs accurately and apply Benjamini-Hochberg within declared comparison families.
5. Avoid fine ranking where uncertainty overlaps.

### Before writing the final Methods section

1. Freeze the exact counts, hashes, commands, versions, and model hyperparameters from the canonical rerun.
2. Reconcile README, data/noise documentation, SCC instructions, metric guidance, and artifact READMEs.
3. State plainly: one first 3 s window/source; short recordings tiled; whole-window nominal SNR; in-band/active diagnostics; two noise realizations; clean-only training; exact category composition; no speech/music/reverb.
4. State limitations on register, technique/family, source collection, tiling, and single-seed pretrained results.

### Before submitting the paper

1. Produce a provenance-checked final table and figures directly from machine-readable outputs.
2. Archive code commit, environment, manifests, compact predictions, checkpoints or stable release links, and statistical outputs.
3. Verify every number in the manuscript against the canonical result hashes.
4. Run a clean checkout reproduction/smoke procedure on SCC or a documented equivalent environment.
5. Ensure license/attribution statements cover Philharmonia/Internet Archive, ESC-50, and pretrained checkpoints.

## 22. Quick wins

These are high-value future changes; none was implemented during this audit.

- Add a build gate rejecting one audio hash mapped to multiple labels.
- Add a result-aggregation gate that refuses different dataset/window/noise-manifest hashes.
- Write a common `run_manifest.json` with Git commit, command, versions, device, seeds, input/output hashes.
- Declare `matplotlib` and `joblib` directly; export a locked SCC environment.
- Pin the AST revision and record/enforce the PANNs base-checkpoint hash.
- Rename/archive stale AST artifacts so “current” is unambiguous.
- Add sealed `finalize_ast.py` and `finalize_panns.py` behavior matching SVM/MERT.
- Make `unittest discover -s tests -q` the documented aggregate command and add CI.
- Reconcile the macro-F1/balanced-accuracy documentation conflict.
- Replace personal SCC paths with `$PROJECT_ROOT`, `$RISE_DATA_ROOT`, and `$USER` examples.

## 23. Blocking questions

1. **EXTERNAL DOCUMENTATION / TEAMMATE CONFIRMATION:** What are the authoritative archive identities and correct labels for the two exact cross-label audio pairs?
2. **DESIGN DECISION:** After correcting those pairs, which new dataset fingerprint will be declared the sole paper build, and who has authority to freeze it?
3. **MISSING DATA:** Are the exact CNN and CRNN feature arrays named by their validation-summary hashes still available on any machine?
4. **MISSING DATA / TEAMMATE CONFIRMATION:** Is the canonical AST checkpoint named by the current summary still preserved on SCC, and can it be archived/distributed?
5. **DESIGN DECISION / EXPERIMENT:** Should FLOAT samples outside `[-1,1]` remain part of the formal noise protocol, or should a frontend-equivalence/headroom experiment first establish that this is comparable across models?
6. **TEAMMATE CONFIRMATION:** What source code/job produced the existing PANNs cross-dataset result referenced by `artifacts/panns/README.md`?
7. **DESIGN DECISION:** Is the paper's claim explicitly limited to within-Philharmonia clean/noisy robustness, or is cross-collection generalization required?

## 24. Final readiness verdict

| Area | Verdict | Reason |
|---|---|---|
| Clean-data validity | NOT READY | Strong split/activity pipeline, but confirmed cross-label duplicate audio requires correction |
| SVM training | READY WITH MINOR CONDITIONS | Code, validation, finalization, and tests are complete; rerun on corrected frozen data |
| CNN training | NOT READY | Saved exact feature inputs unavailable/mismatched; final test/noise absent |
| CRNN training | NOT READY | Same provenance/finalization blockers as CNN |
| Pretrained-model evaluation | NOT READY | MERT workflow is strong, but AST/PANNs test isolation and AST checkpoint provenance need correction; all must rerun after data fix |
| Noise implementation | READY WITH MINOR CONDITIONS | Mixer/provenance/tests are strong; amplitude policy and canonical artifact archiving remain |
| Noisy-test benchmark | BLOCKED | Saved PANNs mixtures differ from SVM/MERT/AST; CNN/CRNN missing; data correction changes build seed |
| Statistical comparison | BLOCKED | No valid complete paired six-model output and no final uncertainty products |
| Methods-section writing | READY WITH MINOR CONDITIONS | Most implementation facts are established; unresolved data/amplitude/provenance decisions must be marked or resolved |
| Paper submission | BLOCKED | Critical data/corpus problems and incomplete model/statistical results prevent defensible final claims |

The central conclusion is not that the repository must be redesigned. Its strongest infrastructure—grouped splitting, train-only statistics, fingerprinted stages, one-time SVM/MERT finalization, deterministic noise provenance, and cluster-aware statistical code—should be preserved. The immediate scientific task is to make every official result refer to one corrected dataset and one realized noise corpus, then complete the missing models under sealed test access.
