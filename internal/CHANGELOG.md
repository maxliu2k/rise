# Changelog: `paper.tex` to `paper_revised.tex`

> **STALE HEADER --- read this first.** `paper.tex` is no longer the original paste. On
> 2026-08-09 it was replaced with a new canonical version authored from `paper_revised.tex`,
> accepting some of the changes below, reverting others, and adding edits of its own. The
> original paste survives only in git, at commit `f48924f`:
> `git show f48924f:paper/paper.tex`.
>
> Sections 1--17 therefore describe how `paper_revised.tex` differs from a file that is no longer
> on disk. They remain accurate as a record of that revision and as the rationale for each
> change --- which is why the canonical version reverted some of them. Section 19 records what
> the canonical version did.
>
> `paper_revised.tex` is now superseded and should probably be deleted.

`paper.tex` was the pasted LaTeX, verbatim and untouched. `paper_revised.tex` is that document
with the prose-draft improvements merged and the problems in `REVIEW.md` fixed.

**Not compiled.** No TeX toolchain was available on this machine, so `paper_revised.tex` has not
been run through pdflatex. Table I gained two `\multicolumn` header rows and three longer Input
cells; check that it still fits `\textwidth` before trusting the layout.

Every change is listed. Reproduce the full diff with:

```bash
diff -u paper/paper.tex paper/paper_revised.tex
```

Legend: **[data]** verified against a committed artifact | **[prose]** taken from a pasted prose
draft | **[fix]** corrects a defect in `REVIEW.md` | **[style]** no change in meaning.

**Sections 1--17 are applied.** Section 18 is proposed and deliberately **not** applied --- it is
kept here so the file records one state of `paper_revised.tex`, not a mixture of what is in it and
what was suggested for it.

---

## 1. Preamble and front matter

| # | change | why |
|---|---|---|
| 1.1 | Added a header comment naming `paper.tex` as the baseline and `CHANGELOG.md` as the record, and stating the file is uncompiled | [style] |
| 1.2 | Removed `\usepackage{algorithmic}` and `\usepackage{xcolor}` | [style] Neither is used. `textcomp`, `amssymb`, and `amsfonts` were **kept** --- less certain they are unused, and they are harmless. |
| 1.3 | Removed the `\def\BibTeX{...}` macro | [style] Defined, never called. |
| 1.4 | Removed the comment explaining `\IEEEoverridecommandlockouts` | [style] There is no funding footnote; the comment described a decision not taken. |
| 1.5 | Added `\graphicspath{{../docs/figures/}{./}}` | [fix] Build note in REVIEW. All three `\includegraphics` used bare filenames, so the document only compiled from a directory containing the images. |
| 1.6 | `1 \textsuperscript{st}` $\to$ `1\textsuperscript{st}` (Allan Luo Yu) | [style] Stray space the other two blocks did not have. |
| 1.7 | Added a TeX comment noting that two authors are both numbered "1st" | [fix] **Not silently changed** --- co-first authorship is legitimate and only the authors can say whether it is intended. Flagged in place. |

## 2. Abstract

| # | change | why |
|---|---|---|
| 2.1 | "Instrument **detection** underpins" $\to$ "Instrument **classification** underpins" | [fix] The task is 12-way classification. The earlier LaTeX abstract had this right; the full draft regressed. |
| 2.2 | "scored on **byte-identical** mixtures, enabling paired comparison" $\to$ "scored on **identical mixtures, regenerated from a fixed seed and verified by exact reproduction of committed per-window predictions**, enabling paired comparison" | **[fix] P1-1.** Byte-identity is false and measured to be false: libsndfile stamps the write time into every float WAV's PEAK chunk, so two generations of the same audio differ at byte 60 (`tests/test_noise.py`). The replacement is the stronger true claim --- 24/24 conditions, 30,120 windows. |
| 2.3 | "within two **points**" $\to$ "within two **percentage points**" | [fix] P2-1 in the first review: the spread is exactly 0.0200, and "points" was ambiguous. |
| 2.4 | `inverted—SVM` $\to$ `inverted---SVM` | [style] Literal em dash to LaTeX `---`. |

**Unchanged and verified:** AST 0.991, PANNs 0.989, MERT 0.980, SVM 0.979, CRNN 0.974, CNN 0.971;
SVM 10.4\% vs 68.0\% at 20 dB; SVM fourth on clean to last under white. **[data]**

## 3. Introduction

| # | change | why |
|---|---|---|
| 3.1 | All hardcoded citations `[1]`--`[17]` $\to$ `\cite{b1}`--`\cite{b17}`, and `[6, 7, 8]` $\to$ `\cite{b6,b7,b8}` | [fix] P2-5. `\bibitem` labels and `\usepackage{cite}` already existed but were unused, so reordering the bibliography would silently desynchronise every number in the text. Applied throughout the document, not only here. |
| 3.2 | Added "It is therefore pertinent to establish not only which classification systems fail under noise, but which instruments they confuse and which noise categories cause it." | **[prose]** The prose draft added a motivating sentence for contribution (iii), which the LaTeX introduced with no setup. Reworded from the prose's "explore the intricacies of instruments and noise type" to name the two concrete things. |
| 3.3 | Deleted "**and whether the backbone is fine-tuned or frozen**" from the confound list | **[fix] P1-5.** No model in the paper uses a frozen backbone: MERT's checkpoint records `backbone_frozen: false`, AST and PANNs are both fine-tuned (§II.C, and PANNs' `test_summary.json` records `"mode": "finetune"`), and the other three have no backbone. The word `frozen` appeared exactly once in the document and was residue from when MERT was a linear probe. The remaining four confounds are real. |

**Deliberately not taken from the prose:** its dropped `[5]` citation (would orphan Livshin \&
Rodet), its "Instrument classification modeling ... reveals little" (the accuracy reveals little,
not the modeling), its "systems ... do not attribute" (the authors attribute), and its added
"(within the pretrained group)" (sharpens the distinction 3.3 deletes).

## 4. Methods --- Materials and Preprocessing

| # | change | why |
|---|---|---|
| 4.1 | "Articulation was filtered to prevent further class imbalance." $\to$ "Exactly one articulation was retained per instrument---\textit{normal} for winds and brass, \textit{arco-normal} for the four bowed strings---so that playing technique could not vary with class, and four recordings carrying conflicting labels were excluded." | **[fix] P2-6.** The original was too vague to reproduce. Rule read from `config.STRICT_ARTICULATIONS`; the four exclusions are `config.CONFLICTING_LABEL_PATHS`, previously unmentioned. **[data]** |
| 4.2 | Added: "This rate is not arbitrary: MP3 bitrate in the source archive varies by instrument family (64, 80, and 96 kbps), so at 44.1 kHz the codec's brick-wall cutoff falls below Nyquist and becomes a per-class shortcut available to any model. At 22.05 kHz the cutoff lies above Nyquist and is discarded; the pipeline asserts this for every resampled file." | **[fix] P2-9.** Neither draft justified `SR = 22050`, and the prose's stated reason ("to standardize input sampling rate") is weak --- every model resamples again to 24/16/32 kHz. The real reason is a controlled confound (`step1_resample.py:74-85`, which fails the build if any ceiling exceeds Nyquist). **[data]** |
| 4.3 | `recording’s` $\to$ `recording's` | [style] Curly apostrophe to ASCII. |
| 4.4 | `analyzed—bassoon` $\to$ `analyzed---bassoon` | [style] |

**Verified unchanged:** 8,374 recordings; 45.3 pitches; 433/852 = 1.97:1; splits 5,861 / 1,258 /
1,255 across 544 pitch groups; 30 dB trim; RMS 0.1; 0.99 cap. **[data]**

## 5. Methods --- Table I

| # | change | why |
|---|---|---|
| 5.1 | Added a `\multicolumn` group header: *Trained from scratch --- no external pretraining* above SVM/CNN/CRNN | **[table image]** The grouping is the clearest thing in the pasted table image and makes contribution (i) legible at a glance. |
| 5.2 | Added a `\multicolumn` group header: *Pretrained backbones --- all fine-tuned, none frozen* above MERT/AST/PANNs | **[table image] + [fix] P1-6.** The image's MERT row said "**Backbone stays frozen**", which is wrong for every number in this paper (frozen MERT scored 0.8931 clean; the reported model scores 0.9798). The header states the correct fact for all three at once. |
| 5.3 | MERT Input: "Waveform, 24 kHz" $\to$ "Waveform, 24 kHz; 13 time-averaged layer outputs \textit{(pretrained on music audio)}" | [table image] |
| 5.4 | AST Input: "**Log-Mel**, 16 kHz" $\to$ "**Waveform**, 16 kHz; official extractor to log-Mel \textit{(pretrained on AudioSet)}" | [table image] The model receives audio and its own extractor produces log-Mel; the original conflated the two. |
| 5.5 | PANNs Input: "Waveform, 32 kHz" $\to$ "Waveform, 32 kHz; CNN14 computes **64-band** log-Mel \textit{(pretrained on AudioSet)}" | [table image] **[data]** Verified: `pretrained_extractors.py:47-48`, `mel_bins=64, fmin=50, fmax=14000`. |

### 5b. Table I, second pass --- layout

The first pass (5.1--5.5) compiled badly: three Input cells wrapped to three lines each, so the
pretrained rows rendered at roughly 3$\times$ the height of the scratch rows, and MERT's Training
cell broke with "$5{\times}10^{-3}$ head" orphaned on a centered line of its own.

| # | change | why |
|---|---|---|
| 5b.1 | **Sample rate promoted to its own column.** New `Rate` column, right-aligned: 22.05 / 22.05 / 22.05 / 24 / 16 / 32 kHz | The one field that varies simply was buried mid-sentence in the widest cell and forced the wrap. Pulling it out collapses every Input cell to one line, and makes the rate comparison scannable --- which matters, since `SR = 22050` is load-bearing (see 4.2). |
| 5b.2 | Dropped the `\textit{(pretrained on ...)}` parentheticals added in 5.3--5.5 | Redundant three ways: the group header says "Pretrained backbones", and the Training column already reads "AudioSet init" / "CNN14, AudioSet init". Cost three lines for no information. MERT's distinct corpus is kept compactly as "MERT-v1-95M (music)", since music-vs-AudioSet is the contrast worth preserving. |
| 5b.3 | All text columns left-aligned (`l`, `X`); `Rate` right-aligned (`r`) | Centered text in wrapped cells was most of the visual noise. |
| 5b.4 | `\hline` $\to$ `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`, `\addlinespace`); added `\usepackage{booktabs}` | Standard for IEEE tables; the group headers now separate by space rather than by a doubled rule. |
| 5b.5 | `\tabcolsep` 5pt $\to$ 4pt | Buys width for the new fifth column. |
| 5b.6 | Sentence case in cells: "Scratch" $\to$ "scratch", "Fixed run" $\to$ "fixed run" | Consistency; these are fragments, not sentences. |

Every row is now one line: six data rows plus two group headers, against roughly thirteen lines
before. **Not verified by compilation** --- the column widths are estimates and the Training column
is the one at risk, MERT's entry being the longest.

**Verified unchanged:** MERT's loss is plain `nn.CrossEntropyLoss` (`train_mert_ft.py:154`), so
Table I's "CE" for MERT against "weighted CE" for the others is correct, not an omission. **[data]**

## 6. Methods --- Modeling prose

| # | change | why |
|---|---|---|
| 6.1 | SVM: added "An RBF kernel was selected by validation macro-F1." | **[prose]** Table I gives $C$ and $\gamma$ but the document never said how they were chosen. |
| 6.2 | PANNs: "computes log-Mel internally" $\to$ "computes **64-band** log-Mel internally" | [table image] **[data]** |
| 6.3 | CNN: "Three convolutional blocks..." $\to$ "**Standardized 128-band log-Mel spectrograms were processed by** three convolutional blocks..." | **[prose]** Matches `logmel_standardization: per_mel_bin_train` in the config fingerprint. |
| 6.4 | "5 seeds (42-46)" $\to$ "(42--46)" | [style] En dash for a range. |

## 7. Methods --- Noise Construction

| # | change | why |
|---|---|---|
| 7.1 | Moved the power definition **before** first use and made it explicit: "let $P(z)=N^{-1}\sum_{i=1}^{N}z_i^{2}$ denote mean-squared waveform power" in the setup sentence; deleted "The power $P(z)$ of any waveform $z$ was defined as its mean squared amplitude" from after (2) | **[prose]** The prose gave the formula; the LaTeX gave only words, and defined the operator after the equation that uses it. |
| 7.2 | Added: "Every model was scored on the same realizations, allowing paired comparison." | **[prose]** |
| 7.3 | Added: "; regeneration was verified by reproducing a previously scored model's committed per-window predictions exactly, for every noise category" | **[fix] P1-1 support.** This is the evidence the abstract's revised claim rests on. **[data]** `scc/verify_regen_all.qsub`, 24/24 conditions. |
| 7.4 | `20–29` $\to$ `20--29`; `25 Hz–8 kHz` $\to$ `25 Hz--8 kHz` | [style] |

**Verified unchanged:** ESC-50 targets `range(20,30)`; 18 DEMAND environments spanning D/N/O/P/S/T;
`MAX_SNR_ERROR_DB = 0.1`; `MAX_DC_POWER_SHARE = 0.01`; $1{,}255\times3\times8\times2 = 60{,}240$.
**[data]**

## 8. Methods --- Evaluation

| # | change | why |
|---|---|---|
| 8.1 | "Both models were evaluated on the same resampled groups." $\to$ "The same resampled groups were evaluated by both models, and the difference recalculated for each resample." | **[prose]** More explicit about what the bootstrap actually does --- which matters more now that §III.C has a committed artifact behind it. |
| 8.2 | `Benjamini-Hochberg(BH)` $\to$ `Benjamini--Hochberg (BH)` | **[prose] [style]** Missing space and wrong dash. Applied at all three occurrences in the document. |
| 8.3 | `noise-category-SNR` $\to$ `noise-category--SNR` | [style] |

## 9. Results --- Clean Classification Performance

| # | change | why |
|---|---|---|
| 9.1 | Added an opening sentence: "Clean audio provided little separation among the systems." | **[prose]** |
| 9.2 | "never differed by more than **0.002**" $\to$ "differed by no more than **0.0021**" | **[fix] P1-3.** The paper drew macro-F1 from the noise sweep's clean pass and accuracy from `test_summary.json`. Those disagree for SVM (0.9770 vs 0.9788) and PANNs (0.9868 vs 0.9885) --- three and two test windows flip between the two evaluation paths, same `model_sha256` on both sides. Retention denominators must come from the same path as the noisy scores, so the sweep is now the single source throughout. Under one source the largest gap is SVM's 0.00204. The prose's "0.0019" was true only of the other pairing. **[data]** |
| 9.3 | Cut the seed sentence to one clause: "Four of the six were evaluated from single fitted runs, so this ordering is not treated as an effect." 45 words $\to$ 18, and the $\pm$0.021 / $\pm$0.003 figures are dropped | [fix] The paper never ranks models on clean score --- its thesis is that clean performance does not predict robustness --- so a variance caveat defends against an inference it does not make. The facts are already in Table I ("5 seeds") and §II.C ("ensembled across 5 seeds (42--46)"), and "provided little separation" is functionally the caveat. Dropping $\pm$0.021 is a correctness gain, not just a cut: CNN's per-seed scores are [0.9251, 0.9731, 0.9727, 0.9700, 0.9693], so that spread is one outlier and the other four span 0.004. Published as-is it reads as instability that is not there. |

**Verified unchanged:** 2.0-point span, 0.9708 (CNN) to 0.9908 (AST); AST accuracy 0.9912 (identical
in both sources); CNN $0.962 \pm 0.021$; CRNN $0.973 \pm 0.003$. **[data]**

## 10. Results --- Noise Robustness

| # | change | why |
|---|---|---|
| 10.1 | Expanded the 20 dB white sentence and added the white AUC ordering: "a 61.9-point difference and the largest observed between any two models at a shared condition. Across the full white-noise curve AST reached a retention AUC of 0.636 and MERT 0.573, while SVM ranked last at 0.259." | **[prose] [data]** |
| 10.2 | Added a paragraph with the 20 dB numbers for the other two categories: AST/MERT/PANNs 85.3 / 84.6 / 83.6\% under human; AST 83.2\% and PANNs 81.2\% under environmental; CNN 46.4\% | **[prose] [data]** Measured 85.32 / 84.57 / 83.59, 83.15, 81.18, 46.41. The LaTeX gave 20 dB only for white, which made the white result look selected. |
| 10.3 | Replaced "The lowest AUC was SVM's under white and human non-speech noise and CNN's under environmental noise" with "CNN was least robust under both recorded categories" inside 10.2 | [style] Subsumed; the AUC values are in Table II. **[data]** CNN is last under human (66.33\%) and environmental (46.41\%) at 20 dB. |
| 10.4 | Added: "The separation appears well before the noise is perceptually meaningful: at 50 dB SNR, CNN has already lost 16.0\% of its clean macro-F1 and CRNN 13.9\%, while AST retains 99.1\% and MERT 97.8\%." | **[data] New finding, not in either draft.** Measured white-noise retention at 50 dB: CNN 84.0, CRNN 86.1, AST 99.1, MERT 97.8. Optional --- delete if space is short. |
| 10.5 | "reaching at most 3.6 points, far below the 61.9-point spread" $\to$ "Even the largest 20 dB range, 3.6 points, was small relative to the 61.9-point maximum separation" | **[fix] P1-2, via [prose].** The unscoped claim was wrong: over all 144 conditions the maximum realization range is **5.124 pp** (CRNN, human, $-5$ dB). 3.646 pp is the maximum *at 20 dB* (SVM, human). The prose draft already scoped it correctly. **[data]** |

**Verified unchanged:** all 24 AUC values in Table II; AST highest in all three; "only AST held its
rank under all three" (checked for all six); median realization range 0.501 pp overall and 0.640 pp
at 20 dB. **[data]**

## 11. Results --- Paired Comparisons

| # | change | why |
|---|---|---|
| 11.1 | Added: "and every window was verified to have been scored against the same noise realization by both models" | **[fix]** Now enforced in code --- `paired_model_comparison.py` refuses to compute if any window's `noise_source` differs between the two models. |
| 11.2 | Split the two differences into per-realization phrasing ("0.1122 macro-F1 in the first realization and 0.1130 in the second") | [style] **[prose]** |
| 11.3 | Added: "Four of the six intervals excluded zero, including both white-noise intervals." | **[prose] [data]** Confirmed by the regenerated artifact: white r0, white r1, human r0, environmental r0. |
| 11.4 | "None of six sign tests survived BH correction." $\to$ "None of the six exact cluster sign tests remained significant after Benjamini--Hochberg correction." | [style] Names the test. **[data]** Smallest $q = 0.465$. |

**All of §III.C was previously unreproducible (P1-4).** It now regenerates from
`artifacts/failure_analysis/paired_model_comparison.{csv,json}` via

```bash
python -m instrument_robustness.paired_model_comparison
```

Every number in the section was **confirmed correct** --- 0.112203, 0.112979, $[0.056044,
0.162518]$, $[0.058924, 0.160092]$, 4 of 6, 0 of 6. No text changed because a number was wrong.

## 12. Results --- Instrument-Specific Failures

| # | change | why |
|---|---|---|
| 12.1 | "(Fig.2, AUC 0.599)" $\to$ "(Fig.~\ref{fig:recall_loss}, AUC 0.599)" | **[fix] P2-5.** Hardcoded figure number. |
| 12.2 | Added "(Fig.~\ref{fig:distance_confusion})" | **[fix] P2-5.** That float was defined with a label and never referenced anywhere in the text. |
| 12.3 | Added: "the association was negative in **17 of the 18** predeclared model-by-noise tests, and seven were significant after..." | **[fix] P1-7. [data]** Only the significance count was reported. 17 of 18 Spearman correlations are negative; the lone exception is SVM/white at $\rho = +0.0039$, zero to three decimals. Median $\rho = -0.298$. The direction is near-unanimous and the paper was reporting only the weaker half. |
| 12.4 | Added: "Six of the seven significant tests involved recorded rather than synthetic noise." | **[data]** Four human, two environmental, one white. This was in the Conclusion ("most consistently under recorded noise") but never in Results. |
| 12.5 | `1.0 \times  10^{-5}` $\to$ `1.0 \times 10^{-5}` | [style] Double space. |

**Verified unchanged:** tuba 0.5991 / violin 0.2147; tuba worst for 5 of 6 under human (PANNs'
worst is french-horn); flute--oboe SVM white 0.342; AST/human $\rho = -0.6066$, $p = 1.0\times
10^{-5}$, $q = 1.8\times10^{-4}$. **[data]**

## 13. Figures

| # | change | why |
|---|---|---|
| 13.1 | Fig. 1 caption: "**Shaded bands span the two noise realizations.**" $\to$ "Each point is the mean of the two noise realizations." | **[fix] P2-11.** There are no shaded bands in `fig6d_retention_compact.png` --- it is plain lines with circular markers. The bands existed in the earlier `fig6`/`fig6b` renders and the caption did not follow the compact variant. |
| 13.2 | Deleted the stale float-placement comment: "`[!b]` + stfloats is deliberate: with `[!t]` the figure lands above two paragraphs of Methods" | **[fix] P2-7.** The figure it precedes is declared `[!t]`. The comment described the opposite of the code. |

## 14. Discussion

| # | change | why |
|---|---|---|
| 14.1 | Deleted `% TO BE WRITTEN (~800 words). Results above is deliberately free of % interpretation...` | **[fix] P2-2.** Stale, and malformed --- the second `%` was mid-line, swallowing the sentence into one comment. |
| 14.2 | "clean **accuracy** is not indicative" $\to$ "clean **performance** is not indicative" | [fix] The headline metric is macro-F1 everywhere else. Accuracy is reported, so the original was defensible, but the mismatch is avoidable. |
| 14.3 | "There were significant variations in model performance ranking **for the rest of the models**: changes in model ranking were observed across noise types and noise levels." $\to$ "model ranking changed across both noise categories and noise levels" | **[fix] P2-4.** "The rest of the models" had no antecedent, the sentence restated itself, and *significant* was used colloquially in a paper that reports BH-corrected significance three subsections earlier. |
| 14.4 | "**SVM and CNN degraded the quickest**" $\to$ "SVM lost the most performance under white noise and CNN under environmental noise" | **[fix] P2-3. [data]** Under white the order worst-first is SVM (0.259), **CRNN** (0.330), CNN (0.357) --- CNN is third-worst. The Conclusion already had this right. |
| 14.5 | "results generally tended towards AST being the most accurate and robust throughout" $\to$ "AST was the most accurate and the most robust throughout" | [style] Hedge removed; the claim is exactly true of Table II. |
| 14.6 | "noise aware training" $\to$ "noise-aware training" | [style] |
| 14.7 | `models’` $\to$ `models'` | [style] |

## 15. Limitations

| # | change | why |
|---|---|---|
| 15.1 | "This issue was explored by evaluating model performance when learning repetition, but only achieved $\sim$20\% accuracy on clean audio. Given this finding, this confound seemed unlikely but its possibility could not be ruled out." $\to$ "This was probed directly: loop period alone predicts instrument at 0.184 balanced accuracy against a chance level of 0.083, and the energy envelope alone at 0.419, so the shortcut is available, but a trained model's misclassifications favour period-matched classes no more often than a five-seed baseline (0.4538 against 0.4592, range 0.3844--0.5254). The confound therefore appears unused rather than absent, and that test rests on 26 errors, so it rules out only a large effect." | **[fix] P1-8. [data]** Three problems. (a) The number is **0.184**, the test score; 0.20 was the *train* figure. (b) "only achieved" inverts the meaning --- chance is 0.083, so 0.184 is 2.2$\times$ chance and the period *does* carry class information. (c) The paper cited the weakest of three probes and omitted that the energy envelope alone reaches **0.419**. The reassurance actually rests on a probe the paper never mentioned. Sources: `outputs/probes/envelope_probe.json` and `period_error_probe.json`, both **restored from git history** (`3229b5f`, `c4c33d5`) --- they had been deleted from the working tree, so the claim had no live source. |
| 15.1b | Compressed 15.1 from 99 words to 66: dropped the envelope probe's 0.419, dropped the baseline range 0.3844--0.5254, and replaced them with the metric's null value (0.5) so the two remaining scores are interpretable without it | [style] The claim is unchanged --- available, unused, underpowered --- and every surviving number is still sourced. `period_rank`'s docstring (`period_error_probe.py:89`) states 0.5 is the no-effect value, which "0.4538 against 0.4592" alone did not convey. **[data]** |
| 15.2 | "Third, only two noise realizations were generated" $\to$ "Third, **because of time and compute constraints** only two noise realizations were generated" | **[prose]** Adds the reason. The prose's second half ("reliable information about variability or uncertainty is inadequate") was **not** taken --- garbled; the LaTeX's "giving a sensitivity check but not a reliable variance estimate" is kept. |
| 15.3 | "making **reliable,** direct comparison of noise loudness unattainable" $\to$ "making direct comparison..." | [style] Redundant with "unattainable". |
| 15.4 | Removed a double space after "could not be ruled out." | [style] |

**Deliberately not taken from the prose:** its ">90\%" in place of the measured **97.3\%** (8,148
of 8,374 --- exact, and vagueness here reads as an estimate), and its wholesale deletion of the
tiling-probe evidence.

## 16. Conclusion

| # | change | why |
|---|---|---|
| 16.1 | "**Our** research" $\to$ "**This** research" | **[prose]** Better for blind review. |
| 16.2 | "white, human, and environmental" $\to$ "white, **human non-speech**, and environmental" | **[prose]** Matches the term used everywhere else. |
| 16.3 | "SVM degraded **earliest** under white noise" $\to$ "degraded **fastest**" | **[fix] P2-12. [data]** At 50 dB the **CNN** has degraded most (84.0\% retention) while SVM is still at 92.2\%. SVM is lowest only from 40 dB down. It degrades fastest and furthest, not earliest. |
| 16.4 | "showed the **most** recall loss" $\to$ "the **greatest** recall loss" | [style] |
| 16.5 | "Acoustically similar instrument pairs became more confused under noise in 7 of 18 model-noise tests after correction, most consistently under recorded noise." $\to$ merged into the previous sentence as "...tended to be confused more heavily under noise---the association was negative in 17 of 18 model-by-noise tests and significant in seven after correction, six of them under recorded noise." | **[fix] P1-7. [data]** Same correction as 12.3. Also fills the `[insert]` / `[INSTRUMENTS]` placeholders from the revised Conclusion draft: AST, SVM/CNN, and tuba/oboe/trumpet. |
| 16.6 | "clean **accuracy** alone is insufficient" $\to$ "clean **performance** alone" | [fix] As 14.2. |
| 16.7 | "...for noisy conditions. Robustness must also be evaluated..." $\to$ "...for noisy conditions; robustness must also be evaluated..." | [style] **[prose]** |

**Note on the pasted Conclusion draft:** it said acoustically similar pairs "were **not
consistently** more likely to become confused." That reads the significance column and ignores the
sign column, and was **not** adopted --- see 16.5.

## 17. Bibliography

| # | change | why |
|---|---|---|
| 17.1 | All straight `"..."` quotes $\to$ LaTeX `` ``...'' `` | [style] The file mixed straight quotes (b1--b11, b13, b14) with curly (b12, b15--b17); straight quotes render as `"` in both open and close positions. |
| 17.2 | All en dashes in page ranges $\to$ `--` (b1, b4, b7, b13) | [style] |
| 17.3 | b5 `notes — indexing` $\to$ `notes --- indexing`; b8 `Kolmogorov–Arnold` $\to$ `Kolmogorov--Arnold` | [style] |
| 17.4 | b15, b16: added the missing space after `\bibitem{b15}` / `\bibitem{b16}` | [style] |
| 17.5 | b15, b17: added the missing terminal period | [style] |

No bibliography entry's content was altered --- authors, titles, venues, years, and DOIs are
untouched. Note that 3.1 makes this list order-independent for the first time.

---

## 18. PROPOSED --- not applied to `paper_revised.tex`

Everything in sections 1--17 is **in** the file. Everything below is **not**. Nothing here has
been written to `paper_revised.tex`; the abstract there still reads as documented in section 2.

### 18a. Abstract, second revision

Four substantive proposals and three tightenings, raised against the compiled abstract.

| # | change | why |
|---|---|---|
| 18a.1 | Add a gap statement: "...yet robustness is rarely compared across model families." | **Substantive.** The abstract never says why the work was needed. The earlier LaTeX draft carried this sentence and the full draft dropped it; it is what makes the paper a contribution rather than a measurement. $\approx$8 words. |
| 18a.2 | Replace the six enumerated clean scores with a range: "Clean macro-F1 spanned only 0.971 to 0.991 across the six." | **Substantive.** The abstract spends $\approx$30 words listing AST 0.991 / PANNs 0.989 / MERT 0.980 / SVM 0.979 / CRNN 0.974 / CNN 0.971, then says in the next clause that all six fall within two points, and concludes that clean macro-F1 predicts nothing. Foregrounding numbers the paper declares uninformative works against its own thesis. Saves $\approx$25 words. |
| 18a.3 | Add the aggregate result: "the models retained 26--64\% of their clean macro-F1 under white noise and 65--79\% under human non-speech; the ranges do not overlap, so no choice of model compensated for the change in noise category." | **Substantive. [data]** Currently the abstract's only evidence is one model at one SNR (SVM at 20 dB). The retention-AUC ranges are an integral over all eight SNRs and are the strongest quantitative claim in the paper. Verified: white 0.259--0.636, human 0.649--0.791, non-overlapping. $\approx$35 words. |
| 18a.4 | Add: "AST was the most robust under all three categories and the only model to hold its rank across them." | **Substantive. [data]** A reader currently finishes the abstract knowing that clean scores do not predict robustness but not which model won. Both halves verified --- AST is highest in all three AUC columns, and rank-holding was checked for all six. $\approx$18 words. |
| 18a.5 | "may **fail or lose** substantial accuracy" $\to$ "may lose substantial accuracy" | [style] Failing is losing accuracy; the pair is redundant. |
| 18a.6 | "scored on identical mixtures, **regenerated from a fixed seed and verified by exact reproduction of committed per-window predictions**, enabling paired comparison" $\to$ "scored on identical mixtures, verified by exact reproduction of committed predictions, so every comparison is paired" | [style] 27 words $\to$ 18 for a mechanism that now lives in §II.D (change 7.3). The claim is unchanged; only the abstract's share of the explanation shrinks. |
| 18a.7 | "model ranking **inverted**" $\to$ "reordered" | [fix] One model moving fourth to sixth is a reorder. "Inverted" overstates it, and the sentence that follows describes exactly one model's move. |

Net effect is close to word-neutral: 18a.2 and 18a.6 pay for 18a.1, 18a.3, and 18a.4.

Full proposed text:

> Instrument classification underpins Music Information Retrieval (MIR) infrastructure and
> supports applications including musicology, audio editing, and transcription. Deep learning
> models that perform well on clean recordings may lose substantial accuracy under artificial and
> real-world noise, yet robustness is rarely compared across model families. This work benchmarks
> six instrument classification models---three trained from scratch (SVM, CNN, CRNN) and three
> pretrained (AST, MERT, and PANNs)---on 12 orchestral instruments from the Philharmonia Orchestra
> Sound Samples dataset, under white (Gaussian), human non-speech (ESC-50), and environmental
> (DEMAND) noise at SNRs from 50 dB to $-$10 dB. All six were scored on identical mixtures,
> verified by exact reproduction of committed predictions, so every comparison is paired.
>
> Clean macro-F1 spanned only 0.971 to 0.991 across the six, but robustness diverged sharply.
> Averaged over the SNR range, the models retained 26--64\% of their clean macro-F1 under white
> noise and 65--79\% under human non-speech; the ranges do not overlap, so no choice of model
> compensated for the change in noise category. At 20 dB the SVM retained 10.4\% under white noise
> against 68.0\% under human non-speech, falling from fourth on clean audio to last. AST was the
> most robust under all three categories and the only model to hold its rank across them. Clean
> macro-F1 is therefore not indicative of noise robustness, and degradation depends on noise
> category, not severity alone.

### 18b. Limitations, tiling paragraph --- readability alternative

Change 15.1b compressed this to 66 words and the compressed form proved hard to read: "period-matched
errors score 0.4538" carries no meaning to a reader who does not already know what the probe was.
The alternative spends words on the mechanism instead of the numbers, at $\approx$78 words:

> 97.3\% of recordings were under three seconds and were tiled to fill the window, so the loop rate
> encodes note length, which itself correlates with instrument. That cue is real: a classifier given
> only the loop period reaches 0.184 balanced accuracy against 0.083 for chance. The models do not
> appear to use it, however --- a CRNN that can read timing confuses similar-length instruments no
> more often than a CNN that cannot. The check rests on 26 errors, so it rules out only a large
> effect.

Recommended over the 66-word form. Costs $\approx$12 words against what is in the file now, and the
50 dB observation (10.4) and the bitrate paragraph (4.2) are both easier places to find them.

---

## 19. AUDIT --- which of sections 1--17 survive in the canonical `paper.tex`

Checked by marker string against `paper/paper.tex` as of 2026-08-09. Method: 56 of the numbered
changes carry a distinctive phrase that can be searched for; those were checked mechanically. The
remainder are §17 bibliography style items (quotes, dashes, terminal periods), confirmed present by
inspection. **41 survived, 15 were reverted.**

### Reverted --- these entries describe `paper_revised.tex` only

| # | change | note |
|---|---|---|
| 1.1 | header comment block | Canonical starts at `\documentclass`. |
| 3.2 | "It is therefore pertinent to establish..." | The prose draft's motivating sentence for contribution (iii); dropped again. |
| 5b.1--5b.6 | **entire Table I layout redesign** | Rate column, booktabs, left alignment, sentence case --- all reverted to the `\hline` + centered + `\newline` form. The wrapping and row-height problem this fixed is back. Possibly an unintended paste from an older state rather than a decision. |
| 6.1 | "An RBF kernel was selected by validation macro-F1." | Lost with the per-model paragraph (19.2 below). |
| 6.2 | PANNs 64-band **in prose** | Still present in Table I (5.5 survived). |
| 6.3 | CNN "Standardized 128-band log-Mel" | Lost with the per-model paragraph. |
| 8.2 | `Benjamini--Hochberg` en dash | The added space survived; the dash reverted to a hyphen. |
| 9.1 | "Clean audio provided little separation among the systems" | |
| 10.1 | white-noise AUC ordering (0.636 / 0.573 / 0.259) | |
| 10.2 | 20 dB human and environmental numbers | |
| 10.4 | 50 dB observation | Was flagged as optional. |
| 11.1 | "every window was verified to have been scored against the same noise realization" | |
| 11.4 | sign tests reported in Results | Removed from Methods too, so the paper stays internally consistent --- see 19.4. |
| 15.1c | "That test rests on 26 errors, so it rules out only a large effect." | **The one revert with a cost.** Limitations now asserts the tiling shortcut "appears unused" with no acknowledgement that the test is underpowered. Recommend restoring. |

### Survived

1.2--1.7, 2.1--2.4, 3.1, 3.3, 4.1--4.4, 5.1--5.5, 6.4, 7.1--7.4, 8.1, 8.3, 9.2, 9.3, 10.3,
10.5, 11.2, 11.3, 12.1--12.5, 13.1, 13.2, 14.1--14.7, 15.1, 15.1b, 15.2--15.4, 16.1--16.7,
17.1--17.5.

Every **[fix]** tagged as correcting a wrong claim survived: byte-identical (2.2), the frozen
confound (3.3), the 3.6-point scoping (10.5), the 0.0021 gap (9.2), 17-of-18 (12.3, 16.5),
"degraded fastest" (16.3), and the Fig. 1 caption (13.1).

### 19b. Changes the canonical version made that sections 1--17 do not cover

| # | change | note |
|---|---|---|
| 19.1 | **Reference b11 (Gonzales) deleted**, with the sentence "Elsewhere, comparisons similarly span only a single model family". b12--b17 renumbered to b11--b16 | Renumbering verified consistent: 16 `\cite` keys, 16 `\bibitem`s, all resolve. "Model span is narrow" now rests on Deng et al. alone. |
| 19.2 | **Per-model paragraph replaced** by three summary sentences | Drops CNN $\approx$111k / CRNN $\approx$294k parameters, GRU width, the 2048-d PANNs embedding, MERT's learned layer weighting, and the AST checkpoint ID. Nothing in the paper now states the scratch models' size or which AST checkpoint was used. |
| 19.3 | Intro paragraphs 1--2 merged and compressed | |
| 19.4 | Sign tests and the exploratory acoustic-summaries analysis removed from Methods; "No multiplicity correction was applied to the bootstrap intervals" added; equation (6) deleted | Internally consistent --- both are now absent from Methods *and* Results. The added sentence is an honest replacement. |
| 19.5 | **Table II restructured** with a stacked `\multicolumn{3}{c}{Retention AUC}` header and `\cline{3-5}`, and moved into §III.A | Better than the flat header it replaced. |
| 19.6 | §III.C reports "only two of four intervals excluded zero" (recorded noise) instead of "four of the six" (all) | **[data]** Verified correct: human r0 yes, r1 no; environmental r0 yes, r1 no. |
| 19.7 | Figure captions rewritten; `fig:recall_loss` scaled to `0.93\columnwidth`; `\FloatBarrier` removed; `\ref{fig:distance_confusion}` split into A and B | |
| 19.8 | "pre-registered test" $\to$ "pre-defined test" | Arguably more accurate. |

### 19c. Defects introduced by the canonical version --- not yet fixed

1. **Duplicated sentence, §II.C:** "CNN and CRNN predictions were ensembled across five seeds."
   then "...across 5 seeds (42--46)." Same paragraph.
2. **Duplicated claim, §II.E2:** the Spearman correlation is described twice.
3. **"Spearmean"** --- typo.
4. **"The difficulties of real-world recordings was also acknowledged"** --- should be *were*;
   also a missing space in `time\cite{b5}`.
5. **"($\sim$Early 2000s)"** renders as "~Early 2000s", and `b3` is dated 1999.
6. **"these mean vectors"** --- antecedent is now "centroid".
7. **"Table I summarizes"** --- hardcoded; everything else uses `\ref{tab:model_configs}`.
8. Double space before "Statistical significance"; curly apostrophe in "model's"; literal em
   dashes in Limitations.

---

## 20. Edits to the canonical `paper.tex` after it became canonical

| # | change | why |
|---|---|---|
| 20.1 | Author block: dropped the `1\textsuperscript{st}` / `2\textsuperscript{nd}` ordinals; added `\textsuperscript{*}` to **Allan Luo Yu and Max Liu only**; added `\thanks{\textsuperscript{*}These authors contributed equally to this work.}` | Resolves the open question flagged in 1.7 and in "Deliberately NOT changed". The ordinals are IEEE **template placeholders** --- the conference template ships author blocks reading "1st Given Name Surname" --- not authorship notation, so two authors marked "1st" reads as a typo rather than as co-first authorship. Equal contribution is conventionally marked with a shared symbol and a footnote; IEEE has no dedicated markup for it. `\thanks` is usable in conference mode here because `\IEEEoverridecommandlockouts` is already set (the line whose explanatory comment 1.4 removed). Gavin Hu is unmarked, per the authors. |

| 20.2 | Affiliation lines: "Boulder, USA" $\to$ "Boulder, CO, USA"; "Sugar Land, USA" $\to$ "Sugar Land, TX, USA"; "Southborough, USA" $\to$ "Southborough, MA, USA". Also "St. Marks School" $\to$ "St. Mark's School" | The IEEE template placeholder reads "City, Country", which is where the original form came from, but published IEEE papers conventionally give City, State, Country for US affiliations. Southborough is the case that needs it --- there is a Southborough in Massachusetts and one in Kent, England. **States inferred from the school names and not confirmed by the authors:** Fairview High School (Boulder, Colorado), Clements High School (Sugar Land, Texas), St. Mark's School (Southborough, Massachusetts). Applied to both `paper.tex` and `paper_revised.tex`. |

**Unverified:** `\thanks` inside a three-block `\author` has not been compiled. If it renders in the
wrong place, the fallback is a manual `\footnotetext` after `\maketitle`. Also worth confirming
against URTC's own author-formatting instructions, which override general IEEE convention.

---

## 21. `paper_revised.tex` rebuilt from canonical, with all proposed changes applied

`paper_revised.tex` was overwritten with a byte-identical copy of the canonical `paper.tex`
(including 20.1), then every outstanding proposal in this file was applied to it. It is once
again a *candidate* revision of the canonical paper, not a stale ancestor of it.

**Applied:**

| source | what |
|---|---|
| **18a.1--18a.7** | Abstract, second revision --- gap statement added; six enumerated clean scores replaced by the 0.971--0.991 range; the non-overlapping 26--64\% / 65--79\% retention ranges added; AST named as most robust and sole rank-holder; "fail or lose" tightened; the verification clause shortened; "inverted" $\to$ "falling". |
| **18b** | Limitations tiling paragraph replaced with the 78-word mechanism-first version, which also restores the 26-errors caveat lost at 19a/15.1c. |
| **19c.1--19c.8** | All eight defects: the duplicated ensemble sentence, the duplicated Spearman description, "Spearmean", "difficulties\ldots was" $\to$ "were" plus the missing space before `\cite{b5}`, "($\sim$Early 2000s)" $\to$ "(from the late 1990s)" (b3 is dated 1999), the dangling "these mean vectors", hardcoded "Table I" $\to$ `\ref{tab:model_configs}`, and the stray double space, curly apostrophe, and literal em dashes. |
| **5b.1--5b.6** | Table I layout redesign restored --- `Rate` column, booktabs rules, left alignment, sentence case, `\tabcolsep` 4pt --- plus `\usepackage{booktabs}`. Restored because 19a flagged the revert as probably an unintended paste rather than a decision, and because the wrapping it fixes is what prompted the request in the first place. |
| **8.2** | `Benjamini--Hochberg` en dash, folded into the §II.E2 rewrite. |

**Verified after applying:** 16 citations, 16 `\bibitem`s, none uncited and none undefined; no
dangling or broken `\ref`; none of the eight defect markers still present. Body length 3158 $\to$
**3153 words**, a net change of $-5$.

**Deliberately not restored** --- these were length decisions, not defects: 3.2 (the "It is
therefore pertinent" sentence), 9.1 ("provided little separation"), 10.1 (white AUC ordering),
10.2 (20 dB human and environmental numbers), 10.4 (the 50 dB observation), 11.1 (the
noise-realization verification clause), 11.4 (sign tests), and 6.1--6.3, which cannot return
without also restoring the per-model paragraph deleted at 19.2.

**Still not compiled.** Table I is the risk: the new fifth column plus MERT's long Training cell
is the combination most likely to overrun `\textwidth`. `\thanks` inside a three-block `\author`
(20.1) is also unverified.

---

## Deliberately NOT changed

| item | why |
|---|---|
| ~~Two authors both numbered "1st"~~ | **Resolved --- see 20.1.** Confirmed as co-first authorship for the first two authors and re-marked with the conventional symbol-plus-footnote. |
| `\begin{table}[H]` and `\begin{figure}[H]` | `[H]` cannot break across columns in a two-column layout and will overfull rather than move. Changing it reflows the whole document, and I cannot compile to check. Still pinned in `REVIEW.md`. |
| "double bass" in §II.A prose vs `double-bass` as the class label | The prose spelling is correct English; the hyphen is a label convention. Left alone. |
| Limitations gains no training-seed sentence | The five-seed ensembles are the intended design, disclosed in Table I and §III.A. P2-1 was withdrawn. |
| The SVM/PANNs three-window and two-window clean discrepancy | 9.2 applies a single-source convention so the paper is self-consistent, but the *root cause* --- why the sweep's clean pass and the one-time test evaluation disagree for exactly two models --- is not diagnosed. Still open in `REVIEW.md` as P1-3. |
| `fig6d_retention_compact.png` has no generator in the repo | P1-9 is a repository problem, not a `.tex` problem. The figure's contents were checked against the data and are current and correct (MERT's curve sits at $\approx$0.63 at 20 dB white, matching 0.6271). Fix belongs in `scripts/fig6b_retention_row.py`. |
| Everything in the prose §III drafts marked `OLD` | Explicitly superseded by their own author. |
| The prose's duplicated paragraphs and typos | "standard **divisions**", "88 handcrafted" with no noun, PANNs cited as `[18]` (out of range --- the bibliography ends at 17), the twice-pasted distance-confusion paragraph, the twice-pasted Limitations paragraph, the unclosed `($\sim$294{,}000 parameters.` --- all listed in `REVIEW.md` §"Prose vs LaTeX differences". |

## Repository changes made alongside this revision

| file | change |
|---|---|
| `src/instrument_robustness/paired_model_comparison.py` | **New.** Regenerates §III.C. Untested --- it reproduced the known numbers exactly, which is evidence but not a test suite. |
| `artifacts/failure_analysis/paired_model_comparison.{csv,json}` | **New.** The artifact §III.C now cites. |
| `src/instrument_robustness/single/envelope_probe.py` | **Restored** from `3229b5f`. |
| `src/instrument_robustness/single/period_error_probe.py` | **Restored** from `c4c33d5`. |
| `outputs/probes/envelope_probe.json` | **Restored** from `3229b5f`. Backs Limitations 15.1. |
| `outputs/probes/period_error_probe.json` | **Restored** from `c4c33d5`. Backs Limitations 15.1. |

Nothing is committed.
