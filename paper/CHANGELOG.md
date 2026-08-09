# Changelog: `paper.tex` to `paper_revised.tex`

`paper.tex` is the pasted LaTeX, verbatim and untouched. `paper_revised.tex` is that document
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
| 9.3 | Reordered the seed sentence to lead with the caveat: "Because SVM, AST, MERT, and PANNs were evaluated from single fitted runs, these small clean-score differences were not tested against training-seed variance; CNN and CRNN are five-seed soft-vote ensembles..." | **[prose]** The prose supplied the caveat the LaTeX stated the facts for but never drew. |

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

## Deliberately NOT changed

| item | why |
|---|---|
| Two authors both numbered "1st" | Could be intended co-first authorship. Flagged with a TeX comment (1.7); only the authors can decide. |
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
