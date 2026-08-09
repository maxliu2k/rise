# Pinned problems in the paper

Checked against the repo at `b519fde`: `artifacts/<model>/noise/noise_sweep_summary.csv`,
`artifacts/<model>/noise/metrics_clean.json`, `artifacts/<model>/test_summary.json`,
`artifacts/failure_analysis/*.csv`, `all-samples/pipeline/windows.csv`, and `config.py`.

Severity: **P1** = wrong, or contradicted by our own artifacts. **P2** = unsupported as
phrased, internally inconsistent, or unreproducible. **P3** = cosmetic.

Nothing here is fixed yet.

---

## P1 --- wrong or contradicted by the data

### P1-1 --- "byte-identical mixtures" is false
> Abstract: "All six models were scored on byte-identical mixtures, enabling paired comparison."

MERT was scored on a **regenerated** corpus, and float WAVs from libsndfile are not
byte-reproducible: the PEAK chunk carries the write timestamp, so two generations of the same
audio differ at byte 60. Measured, not suspected (`tests/test_noise.py`).

The true claim is stronger than the false one: the frozen probe reproduced its committed
per-window predictions on 24/24 conditions, 30,120 windows, across all three noise categories
(`scc/verify_regen_all.qsub`).

Proposed: "each scored on identical mixtures, regenerated from a fixed seed and verified by
exact reproduction of committed per-window predictions, enabling paired comparison."

Note §II.D's "All models were evaluated on the same mixtures" is fine as written --- only the
word *byte-identical* is wrong.

### P1-2 --- "at most 3.6 points" understates the realization spread
> §III.B: "median retention range between draws was 0.5 percentage points across all conditions
> and 0.6 points at 20 dB, reaching at most 3.6 points"

Measured over all 144 model x category x SNR conditions:

| statistic | paper | measured |
|---|---|---|
| median range, all conditions | 0.5 pp | **0.501 pp** OK |
| median range at 20 dB | 0.6 pp | **0.640 pp** OK |
| maximum range | 3.6 pp | **5.124 pp** (CRNN, human, $-5$ dB) |

3.646 pp is the maximum *at 20 dB* (SVM, human), not the maximum overall. The argument survives
--- 5.1 pp is still far below the 61.9-point between-model spread --- but the number is wrong.

**The prose draft already fixes this**: "Even the largest **20 dB** realization range, 3.6
points, was small relative to the 61.9-point maximum separation between models." Scoping it to
20 dB makes the number correct. Take the prose sentence.

### P1-3 --- Table II's clean column and the models' own test evaluations disagree for SVM and PANNs
Two independent clean evaluations exist per model, and the paper quotes both in one paragraph.

| model | `test_summary.json` macro-F1 | sweep `metrics_clean.json` | paper Table II | correct windows: official vs sweep |
|---|---|---|---|---|
| SVM | 0.9770 | 0.9788 | **0.9788** | 1228 vs 1231 |
| CNN | 0.9708 | 0.9708 | 0.9708 | --- |
| CRNN | 0.9738 | 0.9738 | 0.9738 | --- |
| MERT | 0.9798 | 0.9798 | 0.9798 | --- |
| PANNs | 0.9868 | 0.9885 | **0.9885** | 1240 vs 1242 |
| AST | 0.9908 | 0.9908 | 0.9908 | --- |

Same `model_sha256` on both sides, so it is not a stale checkpoint --- three test windows flip
for SVM and two for PANNs between the two evaluation paths. Four models agree exactly; two
do not, which is what makes this worth resolving rather than waving through.

Two consequences:
1. §III.A quotes macro-F1 from the sweep ("0.9908 (AST)") and accuracy from `test_summary`
   ("0.9912") **in the same sentence**. Mixed provenance.
2. The earlier prose draft of the abstract said "PANNs (0.987)" and "SVM (0.977)" --- those are
   the `test_summary` numbers. The current LaTeX says 0.989 and 0.979 --- the sweep numbers.
   The drafts were silently reading different sources.

**Recommended resolution:** cite the sweep (`metrics_clean.json`) for everything. Retention
denominators must come from the same evaluation path as the noisy scores, so Table II is already
correct; it is §III.A's *accuracy* that should switch source. `metrics_clean.json` carries
accuracy too.

**But that changes a sentence.** Accuracy--macro-F1 gaps computed entirely from the sweep:

| model | macro-F1 | accuracy | gap |
|---|---|---|---|
| SVM | 0.97883 | 0.98088 | **0.00204** |
| CNN | 0.97076 | 0.97211 | 0.00135 |
| CRNN | 0.97380 | 0.97530 | 0.00150 |
| MERT | 0.97978 | 0.98167 | 0.00189 |
| PANNs | 0.98853 | 0.98964 | 0.00111 |
| AST | 0.99083 | 0.99124 | 0.00041 |

So "never differed by more than **0.002**" (LaTeX) and "no more than **0.0019**" (prose) are both
true only of the `test_summary` pairings. Under the single-source fix the bound is **0.0021**.
AST's "highest clean accuracy (0.9912)" holds either way.

### P1-4 --- RESOLVED. Every §III.C number now regenerates, and all of them were correct
`src/instrument_robustness/paired_model_comparison.py` writes
`artifacts/failure_analysis/paired_model_comparison.{csv,json}`:

```bash
python -m instrument_robustness.paired_model_comparison
```

| noise | rep | $\Delta$ macro-F1 | 95\% CI | excludes 0 | sign $p$ | sign $q$ |
|---|---|---|---|---|---|---|
| white | 0 | 0.112203 | [0.056044, 0.162518] | yes | 0.560 | 0.840 |
| white | 1 | 0.112979 | [0.058924, 0.160092] | yes | 0.888 | 0.888 |
| human | 0 | $-$0.035579 | [$-$0.065732, $-$0.002668] | yes | 0.289 | 0.578 |
| human | 1 | $-$0.003293 | [$-$0.024620, 0.023201] | no | 0.788 | 0.888 |
| environ | 0 | $-$0.046485 | [$-$0.077184, $-$0.013759] | yes | 0.155 | 0.465 |
| environ | 1 | $-$0.025709 | [$-$0.049276, 0.001531] | no | 0.130 | 0.465 |

Mean white difference 0.1126. **4 of 6 intervals exclude zero, 0 of 6 sign tests survive BH**
(smallest $q = 0.465$). Both drafts' claims are confirmed, including the prose's "four of the six
... including both white-noise intervals" and the LaTeX's "excluding zero in one realization but
not the other" for the two recorded categories.

The module asserts, before computing anything: each condition's metrics fingerprint matches the
current config; the two frames are window-paired; **both models saw the same `noise_source` on
every window**; and all six conditions across both models carry one dataset identity. The family
of six is declared up front, so a BH correction cannot silently run over fewer.

*Original finding, for the record:*

### P1-4 (superseded) --- the paired-comparison numbers cannot be reproduced from anything committed
> §III.C: "CNN exceeded CRNN by 0.1122 and 0.1130 macro-F1; 95\% pitch-group bootstrap
> intervals $[0.0560, 0.1625]$ and $[0.0589, 0.1601]$ both excluded zero. ... None of six sign
> tests survived BH correction."

`noise_stats.py` implements `cluster_bootstrap` and `cluster_sign_test` and is covered by
`tests/test_noise.py`, but **nothing in `scripts/` or the package calls it**, and there is no
committed artifact holding these results --- no CSV, no JSON, nothing under
`artifacts/failure_analysis/`. The intervals survive only as prose in
`docs/POSTER_REVIEW.md:74`, which quotes the same two intervals against a mean of 0.1126
(consistent with the paper's 0.1122 / 0.1130).

The six sign tests have no recorded source at all.

This is the CLAUDE.md rule directly: *"If a result cannot be reproduced by the documented
command, it is not a result."* Every other number in the paper regenerates from a committed
artifact; this subsection does not. Needs a script that writes the comparison to
`artifacts/failure_analysis/`, and the numbers re-read from it.

### P1-8 --- RESOLVED, but the number is 0.184 and the framing is backwards
**Found.** `outputs/probes/envelope_probe.json`, restored from `3229b5f` (it and
`src/instrument_robustness/single/envelope_probe.py` had been deleted from the working tree).
The paper's "$\sim$20\%" is the `period_only` probe --- a classifier given nothing but the
estimated loop period:

| probe | test | train |
|---|---|---|
| `envelope` (energy envelope, no timbre) | **0.4192** | 0.4788 |
| `autocorr` | 0.3781 | 0.4331 |
| `period_only` | **0.1842** | 0.2017 |
| chance | 0.0833 | --- |

Three corrections follow:

1. **The number is 0.184, not $\sim$0.20.** 0.2017 is the *train* figure. Quote the test one.
2. **"only achieved" inverts the meaning.** Chance is 0.0833, so 0.184 is **2.2$\times$ chance** ---
   loop period alone *does* carry class information. The defensible claim is not that the
   shortcut is absent, but that it is far too weak to account for 0.97+ macro-F1.
3. **The stronger availability result is omitted.** The energy envelope alone --- no timbre at
   all --- reaches **0.419**, five times chance, and the estimated period recovers true note
   length at $r = 0.914$. The paper cites the weakest of the three probes and calls the confound
   "unlikely".

What actually licenses the reassurance is a *different* probe the paper never mentions
(`outputs/probes/period_error_probe.json`, restored from `c4c33d5`): a trained CRNN's
misclassifications favour period-matched classes at **0.4538**, inside the five-seed CNN baseline
range of 0.3844--0.5254 (mean 0.4592). No excess --- the shortcut is available but not used.
`crnn_model.py:33` states the caveat plainly: 26 errors, SE $\approx$ 0.057, so it rules out a
large effect rather than proving none.

Proposed replacement: "Loop period alone predicts instrument at 0.184 balanced accuracy against a
chance level of 0.083, and the energy envelope at 0.419 --- the shortcut is available. A trained
CRNN's misclassifications nonetheless favour period-matched classes no more often than a
five-seed CNN baseline (0.4538 against 0.4592, range 0.3844--0.5254), so it does not appear to be
used, though that test rests on 26 errors and rules out only a large effect."

### P1-8 (original finding, superseded above) --- the "$\sim$20\%" had no locatable source
> Limitations: "This issue was explored by evaluating model performance when learning repetition,
> but only achieved $\sim$20\% accuracy on clean audio."

The repo records **two** tiling probes, and this matches neither:

1. `docs/FINDINGS.md:337-352` --- pre-registered test that if the model reads loop period, classes
   at the extremes of the source-length distribution should be the most noise-robust. Result: both
   extremes floor at 0.000 recall at 0 dB, no monotone relationship. Corroborating figure: 0 dB
   balanced accuracy under tiling **0.1071** vs **0.1207** for the old variable-length design.
2. `docs/AUDIT_CHECKLIST.md:252-255` --- a CRNN probe finding "no excess period-matched errors",
   run on clean audio only.

Neither produces $\sim$20\%. Probe 1's numbers are 10.7\% and 12.1\% and are *noise* results, not
clean. Either the paper is describing a third probe whose output was never committed, or the
number is misremembered from probe 1.

Two further scope problems if it is probe 1: `FINDINGS.md` states the result "validates tiling
*for this CNN*" and flags explicitly that "an SVM with handcrafted onset-rate features could still
pick up the loop period" --- and the SVM is the model whose white-noise robustness is the paper's
headline failure. `AUDIT_CHECKLIST.md` adds that the probe "was run on clean audio only --- never
under noise, which is exactly where the shortcut might matter."

Same class as **P1-4**: a number in the paper with no artifact behind it.

### P1-9 --- Figure 1 has no generator in the repository
`fig6d_retention_compact.png` is committed as a binary (`a2c7c9e`, whose message says "committing
it so it can be uploaded to Overleaf") but **nothing in the tree produces it**.
`scripts/fig6b_retention_row.py` writes only `fig6b_retention_row`; there is no `fig6c` or `fig6d`
code anywhere. Same for `fig6c_retention_col`.

The other two figures are fine --- `fig_recall_loss_heat.py` and `fig_distance_confusion.py` both
carry a `"column"` variant that produces exactly the files the paper includes.

I checked the figure's contents against the data by eye and it is **current and correct**: the red
MERT curve sits at $\approx$0.63 at 20 dB white, matching fine-tuned MERT's 0.6271, and all six
curves match Table II at every gridpoint I could read. So this is a reproducibility gap, not a
wrong figure --- but `REPOSITORY_AUDIT.md` lists "a plot regenerated from a different config than
the numbers printed beside it" as one of this repo's known historical bugs, and an uncheckable
figure is how that happens.

Fix: add the `fig6d` variant to `fig6b_retention_row.py` and regenerate.

### P1-5 --- the frozen-vs-fine-tuned confound no longer exists in this study
> §I: "pretraining corpora, input representation, sample rate, architecture, and whether the
> backbone is **fine-tuned or frozen** all differ, so family-level differences cannot be
> attributed to a single factor."

All three pretrained backbones are fine-tuned: MERT (`train_mert_ft.py`, checkpoint carries
`backbone_frozen: false`), AST ("backbone and head fine-tuned jointly", §II.C), PANNs
(`test_summary.json` records `"mode": "finetune"`). The other three models have no backbone at
all. **No model in the paper uses a frozen backbone.**

The word `frozen` appears exactly once in the whole document --- here --- and it is residue from
when MERT was a frozen linear probe. It now names a source of variation the design does not
contain, which is worse than a stale number: a reader who checks Table I against this sentence
finds the paper contradicting itself about its own design. Delete that clause; the other four
confounds are real and carry the argument on their own.

The prose draft makes this worse, not better --- it adds "(within the pretrained group)",
which sharpens a distinction that no longer exists.

### P1-6 --- the proposed replacement table says MERT's backbone stays frozen
The pasted table image, under **Pretrained backbones**, gives MERT:

> What's learned on this dataset: "Layer weights + linear head" / "**Backbone stays frozen**"

That is the frozen-probe description, and it is wrong for every number in this paper. MERT's
backbone was fine-tuned; the checkpoint records `backbone_frozen: false`; its clean macro-F1
went 0.8931 (frozen) to 0.9798 (fine-tuned), and its white-noise AUC 0.573 does not exist for
the frozen model. Using this table as drawn would reintroduce exactly the error **P1-5** is
about, in the one place a reader looks first.

Correct cell: "Backbone + layer weights + linear head, all fine-tuned."

The rest of the image checks out: MERT 13 time-averaged layer outputs; AST 16 kHz via the
official extractor; PANNs 32 kHz waveform with CNN14 computing **64**-band log-Mel internally
(`pretrained_extractors.py:47-48`, `mel_bins=64, fmin=50, fmax=14000`) --- a detail Table I
currently omits and is worth keeping.

### P1-7 --- "not consistently more likely to become confused" understates the result
The revised Conclusion draft says acoustically similar pairs "were **not consistently** more
likely to become confused under noise."

That reads the significance column and ignores the sign column. Across all 18 model-by-noise
cells:

- **17 of 18 Spearman correlations are negative** --- closer pairs, more noise-induced confusion.
- The single exception is SVM under white noise at $\rho = +0.0039$, which is zero to three
  decimal places, not a counterexample.
- Median $\rho = -0.298$.
- Seven clear BH correction; **six of those seven are recorded noise** (four human, two
  environmental).

So the *direction* is near-unanimous and the *significance* is patchy --- those are different
statements, and the draft asserts the weaker one about the wrong column. 17/18 by sign alone is
$p < 0.001$ under a two-sided sign test.

This also loses the "most consistently under recorded noise" finding that the current
`paper.tex` Conclusion has and that the data supports.

Proposed: "acoustically similar instrument pairs tended to be confused more heavily under noise
--- the association was negative in 17 of 18 model-by-noise tests and significant in seven after
correction, six of them under recorded noise."

---

## P2 --- unsupported as phrased, or unreproducible

### P2-1 --- WITHDRAWN. The ensembles are the intended design
CNN and CRNN are meant to be five-seed ensembles; that is what those systems are, not an
accident of how many seeds were run. The paper already discloses it in Table I ("5 seeds") and
states it outright in §III.A ("CNN and CRNN are five-seed soft-vote ensembles"), and the prose
draft adds the one caveat sentence that was missing. Nothing further is owed here.

Reference numbers, kept because they are cheap to have and awkward to recompute:

| model | training seeds | reported system | single-seed spread |
|---|---|---|---|
| SVM | none (deterministic RBF fit) | the fit | --- |
| CNN | 5 (42--46) | **5-seed soft-vote ensemble** | 0.9620 $\pm$ 0.0207 |
| CRNN | 5 (42--46) | **5-seed soft-vote ensemble** | 0.9733 $\pm$ 0.0025 |
| MERT | 1 (42) | the run | --- |
| AST | 1 | the run | --- |
| PANNs | 1 | the run | --- |

The complaint is usually phrased as unequal *variance reporting*. That is the smaller half. The
larger half is that **two of the six reported systems are ensembles and four are single runs**,
and the noise sweep scored the ensembles --- `metrics_clean.json` for CNN is 0.97076, matching the
ensemble exactly, not the 0.9620 single-seed mean. So the asymmetry propagates into every
robustness number, not just the clean column.

Size of the effect:

- **CNN gains +0.0088 macro-F1 from ensembling** (0.9620 $\to$ 0.9708). That is larger than the
  MERT--SVM clean gap (0.0010) and larger than the CRNN--CNN clean gap (0.0030).
- CRNN gains +0.0005. Negligible.

One thing worth knowing before quoting $\pm$0.021 anywhere: CNN's per-seed scores are
[0.9251, 0.9731, 0.9727, 0.9700, 0.9693]. **Seed 42 is an outlier**; the other four span 0.004.
So $\pm$0.021 is one bad run, not typical run-to-run variation, and should not be cited as a
general instability figure for the CNN.

### P2-1b --- Limitations omits the training-seed asymmetry
§III.A states the fact (four single runs, two five-seed ensembles) but never draws the caveat,
and Limitations covers noise realizations while saying nothing about training seeds. This is the
one a reviewer will raise, and it bites where the paper's margins are thinnest: MERT 0.9798 vs
SVM 0.9788 is a 0.001 gap, against a CNN single-seed spread of $\pm$0.021.

Proposed addition: "AST, PANNs, and MERT were each evaluated from a single training seed,
whereas CNN and CRNN used five-seed ensembles. Between-seed variation was therefore not
estimated uniformly, and small differences between model families should be read descriptively
rather than as established orderings."

**Partly resolved by the prose Results**, which adds: "Because SVM, AST, MERT, and PANNs were
evaluated using single training seeds, these small clean-score differences were not tested
against training-seed variance." Take that sentence --- but it covers only the *clean* scores.
The same caveat governs the AUC orderings in §III.B, so Limitations still needs the general
form. Both drafts of Limitations remain silent on training seeds.

### P2-2 --- the Discussion is a placeholder and still says so
The source carries `% TO BE WRITTEN (~800 words)` above roughly 150 words of text. The comment
is also malformed --- the second `%` sits mid-line, so the sentence "Results above is
deliberately free of interpretation" is swallowed into one comment line.

### P2-3 --- "SVM and CNN degraded the quickest" is imprecise
Under white noise the order worst-first is SVM (0.259), **CRNN** (0.330), CNN (0.357) --- CNN
is third-worst, not second. The Conclusion gets this right ("SVM degraded earliest under white
noise and CNN under environmental"); the Discussion generalizes it into a ranking the table does
not support.

### P2-4 --- "significant variations in model performance ranking for the rest of the models"
"The rest of the models" has no antecedent --- no subset was named beforehand. And *significant*
is used colloquially in a paper that reports BH-corrected significance three subsections
earlier. Reword to avoid the word entirely.

### P2-11 --- Figure 1's caption describes shaded bands the figure does not have
> "Shaded bands span the two noise realizations."

There are no shaded bands in `fig6d_retention_compact.png` --- it is plain lines with circular
markers, six per panel. The band was in the earlier `fig6`/`fig6b` renders; the compact variant
dropped it and the caption did not follow.

Either restore the bands (they are the visual evidence for §III.B's realization-agreement
argument, and at a median 0.5 pp they would be nearly invisible anyway) or delete the sentence.
Deleting is honest and costs nothing; claiming a band that is not drawn is a reviewer noticing
you did not look at your own figure.

### P2-12 --- "SVM degraded earliest under white noise" is the wrong word
White-noise retention at the top of the SNR grid:

| dB | SVM | CNN | CRNN | MERT | PANNs | AST |
|---|---|---|---|---|---|---|
| 50 | 92.2 | **84.0** | 86.1 | 97.8 | 90.2 | 99.1 |
| 40 | **59.6** | 70.5 | 70.1 | 87.4 | 73.7 | 95.7 |
| 30 | **28.2** | 48.9 | 46.6 | 72.6 | 56.1 | 87.1 |

At 50 dB the **CNN** has degraded most (84.0\%); SVM is still at 92.2\%. SVM is lowest only from
40 dB down. So SVM degrades *fastest*, and *furthest*, but not *earliest*. "Degraded fastest" or
"lost the most" is accurate; "earliest" is checkable and wrong.

### Observation (not a defect) --- everything except AST and MERT is already damaged at 50 dB
At 50 dB SNR the added noise is essentially inaudible, yet CNN has lost 16\% of its clean
macro-F1, CRNN 14\%, PANNs 10\%, SVM 8\% --- while AST retains 99.1\% and MERT 97.8\%. The paper
never remarks on this, and it is arguably a cleaner statement of the pretrained/scratch split than
anything in §III.B: the gap opens before the noise is perceptually meaningful at all. Worth one
sentence if there is room. Flagging it as an opportunity, not an error.

### P2-9 --- the paper never says why the sample rate is 22.05 kHz, and the real reason is a strength
Neither draft justifies `SR = 22050`. The prose adds "to standardize input sampling rate across
recordings", which is a weak rationale and arguably a wrong one --- every model resamples again
to its own rate (24, 16, 32 kHz), so standardizing at 22.05 kHz is not what makes the inputs
comparable.

The actual reason is a confound the study defused and does not claim credit for. Per-instrument
MP3 bitrate in the Philharmonia archive varies by family (64/80/96 kbps). At 44.1 kHz the codec
brick wall falls *below* Nyquist and becomes a free per-class shortcut --- a model could identify
the instrument from the encoder, not the timbre. At 22.05 kHz the brick wall is *above* Nyquist
and is discarded. `step1_resample.py:74-85` asserts every resampled ceiling is at or below
Nyquist and fails the build otherwise.

A reviewer who knows the Philharmonia archive will ask about bitrate. Two sentences in §II.B
turn an unexplained constant into a controlled confound.

### P2-10 --- PANNs is cited as [18] in the prose, and [18] does not exist
The bibliography ends at `b17`. PANNs is [17] in the LaTeX, which is correct. Do not carry the
prose's `[18]` across.

### P2-5 --- every cross-reference in the body is hardcoded
- Citations are literal `[1]`--`[17]` while `\bibitem{b1}`--`{b17}` define labels and
  `\usepackage{cite}` is loaded. Reordering the bibliography silently desynchronises the text.
- §III.D says "(Fig.2, AUC 0.599)" instead of `\ref{fig:recall_loss}`.
- `\label{fig:recall_loss}` and `\label{fig:distance_confusion}` are **never** `\ref`'d.
  `fig:distance_confusion` is not referenced anywhere in the text --- an unreferenced float.

### P2-6 --- "Articulation was filtered to prevent further class imbalance" is too vague to reproduce
The actual rule is one articulation per instrument: `normal`, except the four bowed strings
(cello, double-bass, viola, violin) which use `arco-normal`. One clause would make §II.A
reproducible. Four recordings with conflicting labels were also excluded --- currently unstated.

### P2-7 --- the float-placement comment describes the opposite of the code
The comment block before `fig:curves` says "`[!b]` + stfloats is deliberate: with `[!t]` the
figure lands above two paragraphs of Methods". The figure that follows is declared `[!t]`.

### P2-8 --- author block
Two authors are both numbered "1st" (co-first authorship, or a typo?). `1 \textsuperscript{st}`
has a stray space the other two lack. Allan's block gives an ORCID where the other two give
email addresses.

---

## P3 --- cosmetic

- `Benjamini-Hochberg(BH)` --- missing space.
- Mixed dashes: `---` in the abstract, literal em dashes elsewhere (`inverted—SVM`,
  `analyzed—bassoon`, `notes — indexing`).
- Mixed quote characters: curly (`recording’s`, `“Sound samples,”`) and straight (`"..."` in
  most `\bibitem`s) in the same file.
- Unused: `\usepackage{algorithmic}`, `xcolor`, `textcomp`, and the `\BibTeX` macro.
- `b15` and `b17` lack terminal periods; `b15`--`b17` lack a space after `\bibitem{}`.
- §II.A writes "double bass"; the class label everywhere else is `double-bass`.
- `\begin{table}[H]` and `\begin{figure}[H]` force placement in a two-column layout --- `[H]`
  cannot break across columns and will overfull rather than move. Fragile next to
  `\FloatBarrier`.
- Repo, not paper: `fig_recall_loss_heat.py` and `fig_distance_confusion.py` both have docstrings
  saying they write `{,_compact}` variants. Each has three, including the `column` variant the
  paper actually includes. The docstring undersells the file the paper depends on.

---

## Prose vs LaTeX differences

Prose drafts are pasted separately from the LaTeX. Only the LaTeX lives in `paper.tex`; this
section records what each prose version says differently, so nothing is lost by not merging it.

### §I Introduction

Paragraphs 1 and 4 are identical. The differences are all in 2 and 3:

| # | LaTeX | prose | consequence |
|---|---|---|---|
| 1 | "difficulties of real-world recordings **[5]**" | citation dropped | **Orphans reference [5]** (Livshin \& Rodet) --- it is then cited nowhere, and the text jumps [4] to [6]. Keep the LaTeX. |
| 2 | --- | adds "It is thus pertinent to recognize how and which instrument classification models fail, and to explore the intricacies of instruments and noise type." | Motivates contribution (iii), which the LaTeX introduces without setup. Worth keeping, but "intricacies" is vague --- say "which instruments and which noise categories". |
| 3 | "whether the backbone is fine-tuned or frozen" | adds "**(within the pretrained group)**" | Sharpens a distinction that no longer exists --- see **P1-5**. Delete the clause in both. |
| 4 | "High accuracy on clean instrument audio reveals little about performance under perturbation" | "Instrument classification **modeling** has achieved high accuracy ... but **reveals little** about classification performance" | Grammar regression: the modeling is not what reveals little, the high accuracy is. LaTeX is correct. |
| 5 | "Complete classification systems are therefore compared, and performance differences are not attributed to architecture alone." | "Therefore, complete classification systems are compared and **do not attribute** performance differences to architecture alone." | Grammar regression: systems do not attribute anything; the authors do. LaTeX is correct. |
| 6 | "additive noise ... **was** not covered" | "**is** not covered" | Tense only. LaTeX is consistent with the surrounding past tense. |
| 7 | "comparisons similarly span only a single model family" | "comparisons similarly only span a single model family" | Word order only. |

**Net:** the LaTeX Introduction is better on every substantive point. Take only item 2 from the
prose, and fix item 3 in both.

Worth noting in the other direction: contribution (ii) says "every model is scored on the **same
generated noisy mixtures**", which is exactly right. The Introduction states the pairing claim
correctly while the Abstract overstates it as byte-identical (**P1-1**) --- so the fix already
exists in the paper's own words.

Unverifiable from our artifacts, flagged only so it is not mistaken for a checked number: "works
pushed accuracy and precision past 95\% on clean, isolated-note datasets [6, 7, 8]" is a claim
about prior work. Also, IEEE style would set that as [6]--[8].

### §III Materials and Methods

The prose version is a half-converted draft: plain headings (`A.`, `B.`, `C.`) sit alongside raw
`\subsection{}` / `\subsubsection{}` commands, and it ends with a block explicitly marked
`OLD` --- do not merge that block. It also carries several defects the LaTeX does not.

**Prose changes worth keeping:**

| # | change | why |
|---|---|---|
| 1 | defines the power operator explicitly: \(P(z)=N^{-1}\sum_{i=1}^{N}z_i^2\) | The LaTeX says only "mean squared amplitude" in words. The formula is better --- but the prose places it *after* first use in (2); move it before. |
| 2 | adds "An RBF SVM was selected by validation macro-F1." | The LaTeX gives $C$ and $\gamma$ in Table I but never says how they were chosen. |
| 3 | adds "Models were evaluated on the same realizations, allowing paired comparison." | Correct phrasing of the pairing claim --- the third place the paper states it right while the Abstract overstates it (**P1-1**). |
| 4 | "Standardized 128-band log-Mel" for CNN | Matches `logmel_standardization: per_mel_bin_train`; the LaTeX omits it. |
| 5 | splits the bootstrap into per-step detail ("The same resampled groups were evaluated by both models...") | Clearer, and the explicitness matters given **P1-4**. |
| 6 | "Benjamini--Hochberg correction" | Fixes the LaTeX's `Benjamini-Hochberg(BH)` (**P3**). |

**Prose defects --- do not carry across:**

| # | defect | note |
|---|---|---|
| 7 | PANNs cited as **[18]** | Out-of-range; see **P2-10**. |
| 8 | **the entire "To examine whether acoustically similar instruments..." paragraph appears twice**, back to back | Copy-paste duplication. |
| 9 | "The two realization scores were averaged for each noise-category--SNR point" appears twice --- once in E.1 para 1, once after (5) | Same sentence, two places. |
| 10 | "temporal means and standard **divisions**" | Typo for *deviations*. |
| 11 | "producing 88 handcrafted per window" | Missing the noun *features*. |
| 12 | "fine-tuned jointly with the head" --- no terminal period | Also "...0.4593 [16] Each window was resampled" runs two sentences together. |
| 13 | "($\sim$294{,}000 parameters." | Unclosed parenthesis. |
| 14 | "(50, 40, 30, 20, 10, 0, -5 and $-$10 dB)" | Mixes a hyphen-minus with a real minus, and drops the serial comma. The LaTeX sets both as `$-5$` and `$-10$`. |
| 15 | "Low SNR can exceed the [$-$1, 1] range" not in math mode | LaTeX has `$[-1, 1]$`. |
| 16 | "PANNs framework uses AudioSet-pretrained CNN14 [18] received the waveforms" | Broken grammar --- two clauses fused. |
| 17 | "comprises thousands of samples **and** over 20 instruments" | LaTeX's "across over 20 instruments" is correct. |
| 18 | "CNN and CRNN predictions ensembled (seeds 42-46) across 5 seeds." | Says *seeds* twice. |
| 19 | carries `\label{AA}` on the Modeling subsection | Leftover from the IEEE template; unreferenced. |
| 20 | OLD block: "we calculated the centroid" | First person, inconsistent with the passive voice used throughout. Marked OLD --- discard entirely. |

**Neutral rewordings** (no action): "Data are split by instrument and note" restates the next
sentence; "making this model time-sensitive" on the CRNN is interpretive but fair; the ESC-50 /
DEMAND selection order is reorganized without changing meaning.

### §IV Results

**This is the one section where the prose is clearly better than the LaTeX.** It fixes a P1,
supplies a missing caveat, and adds seven verified numbers the LaTeX leaves out. Every new
number checks out:

| prose claim | measured |
|---|---|
| accuracy--macro-F1 gap $\le$ 0.0019 | 0.00189 (MERT) --- tighter and truer than the LaTeX's 0.002 |
| MERT 62.7\% at 20 dB white | 0.6271 |
| AST / MERT / PANNs = 85.3 / 84.6 / 83.6\% at 20 dB human | 85.32 / 84.57 / 83.59 |
| AST 83.2\%, PANNs 81.2\% at 20 dB environmental | 83.15 / 81.18 |
| CNN 46.4\% at 20 dB environmental | 46.41 |
| "CNN was least robust under these conditions" | CNN is last under *both* human (66.33\%) and environmental (46.41\%) |
| "remaining 11 tests were not significant" | $18-7=11$ |
| "Four of the six ... intervals excluded zero, including both white-noise intervals" | consistent with the LaTeX's "one realization but not the other" for the other two categories --- but still unreproducible, **P1-4** |

**Take from the prose:**
1. The corrected realization-spread sentence (**P1-2**).
2. The single-seed caveat in §A (**P2-1**).
3. The 20 dB human and environmental numbers --- the LaTeX reports 20 dB only for white, which
   makes the white result look cherry-picked. Giving all three kills that impression.
4. "Four of the six ... excluded zero" --- more informative than "in one realization but not
   the other", and it states the count plainly.

**Two problems in the prose:**

| # | defect | note |
|---|---|---|
| 1 | "the 61.9\\% gap" | It is 61.9 **percentage points**, not 61.9\%. The LaTeX says "61.9-point difference", which is right. The same paragraph later says "61.9-point maximum separation" --- so the prose contradicts itself within three sentences. |
| 2 | "AST retained the best performance across every noise category, **but ordering of the remaining systems changed**" | True, but it drops the LaTeX's sharper "only AST held its rank under all three", which is the actual finding and is verified. Keep the LaTeX phrasing. |

### Limitations

**The paragraph is pasted twice**, identically, once with LaTeX escapes and once plain.

| # | LaTeX | prose | consequence |
|---|---|---|---|
| 1 | "97.3\% of recordings were shorter than three seconds" | "The vast majority ($>$90\%)" | **Loss.** 97.3\% is measured and exact (8,148 of 8,374). ">90\%" is true but vaguer, and vagueness here reads as an estimate rather than a count. Keep 97.3\%. |
| 2 | "This issue was explored by evaluating model performance when learning repetition, but only achieved $\sim$20\% accuracy on clean audio. Given this finding, this confound seemed unlikely but its possibility could not be ruled out." | dropped entirely | **Serious loss.** This is the only evidence in the paper that the tiling confound was *tested* rather than merely acknowledged. Without it the limitation is a bare concession; with it, it is a controlled check. Restore. |
| 3 | "Third, only two noise realizations were generated, giving a sensitivity check but not a reliable variance estimate." | "because of time and compute constraints, there were only two noise realizations. Thus, only sensitivity analysis is provided and reliable information about variability or uncertainty is inadequate." | The prose adds the *reason* (worth keeping) but "reliable information ... is inadequate" is garbled. Merge: keep the LaTeX's second half. |
| 4 | ordinal markers First/Second/Third/Fourth/Lastly | prose loses "Third"/"Fourth", using "Additionally" and an unmarked sentence | The LaTeX's enumeration is easier to follow in a dense paragraph. |
| 5 | neither | neither | **Training-seed asymmetry still absent from both** --- see **P2-1**. |

### §VI Conclusion --- placeholders filled

The revised draft arrived with `[insert]` / `[INSTRUMENTS]` slots. Filled from the artifacts,
with the **P1-7** correction applied:

> This research compares and ranks six musical-instrument classification systems under
> controlled white, human non-speech, and environmental noise degradation. Despite similar
> performance on clean recordings, **AST** retained the most clean macro-F1 across the tested
> conditions, while **SVM degraded earliest under white noise and CNN under environmental
> noise**. Failures were also uneven across instruments: **tuba, oboe, and trumpet** showed the
> greatest recall loss, and **acoustically similar instrument pairs tended to be confused more
> heavily under noise --- the association was negative in 17 of 18 model-by-noise tests and
> significant in seven after correction, six of them under recorded noise**. These findings
> demonstrate that clean accuracy alone is insufficient for selecting an instrument classifier
> for noisy conditions; robustness must also be evaluated across multiple noise categories and
> severities. Future work should extend this benchmark to multiple datasets, polyphonic music,
> independently recorded environmental noise, additional noise realizations, and noise-aware
> training.

Sources: AST is highest in all three AUC columns (Table II). SVM is lowest under white (0.259)
and CNN lowest under environmental (0.505) --- note this is *not* "SVM and CNN degraded the
quickest" generally, which is **P2-3**. tuba 0.5991 / oboe 0.5800 / trumpet 0.5123 are the top
three by mean recall-loss AUC.

Two smaller notes on this draft:
- It changes "Our research" to "This research" --- better for a blind submission.
- "clean accuracy alone" is defensible here (accuracy *is* reported, and tracks macro-F1 within
  0.0019), but the paper's headline metric is macro-F1 everywhere else. "Clean performance"
  avoids the mismatch without asserting either metric.

---

## The proposed replacement table

**It cannot replace Table I, and it should not be discarded either.** They are different tables:

- **Table I is the reproducibility table** --- learning rates, batch sizes, epoch caps,
  early-stopping patience, seed counts, loss weighting. A reviewer needs every one of those to
  judge or repeat the work. The image drops all of them.
- **The image is the orientation table** --- what each model consumes, what is actually learned,
  and the scratch-versus-pretrained split. It does that job better than Table I does, and the
  grouped headers make the study's span legible at a glance, which is the paper's contribution (i).

Recommended: keep Table I, and fold the image's two genuine additions into it --- group the rows
under **Trained from scratch** / **Pretrained backbones** headers, and put the pretraining corpus
(AudioSet, AudioSet, music audio) in the Input column. That gets the orientation benefit without
surrendering the hyperparameters. Both as separate tables is not worth the column-inches in a
4-page format.

If any version of the image ships, the MERT row must be fixed first (**P1-6**).

---

## Verified correct --- do not re-check

**Dataset (§II.A--B).** 8,374 recordings; splits 5,861 / 1,258 / 1,255 across 544 pitch groups;
45.333 distinct pitches per instrument (paper: 45.3); violin 852 / trumpet 433 = 1.968 (paper:
1.97:1); 97.30\% of recordings under 3 s (paper: 97.3\%). `TRIM_TOP_DB=30`, `WINDOW_S=3.0`,
`MAX_WINDOWS_PER_SOURCE=1`, `SR=22050`, `TARGET_RMS=0.1`, peak cap 0.99 --- all match.

**Noise construction (§II.D).** ESC-50 targets `range(20,30)` = 20--29; DEMAND is exactly 18
environments spanning D/N/O/P/S/T = domestic, nature, office, public, street, transport;
`MAX_SNR_ERROR_DB=0.1`; `MAX_DC_POWER_SHARE=0.01`; near-silent rejection present;
`INSTRUMENT_BAND_HZ=(25, 8000)` matches "25 Hz--8 kHz"; $1{,}255\times3\times8\times2 = 60{,}240$.

**Model configs (Table I).** MERT fine-tune uses unweighted `nn.CrossEntropyLoss` --- Table I's
"CE" (not "weighted CE") for MERT alone is correct. LR 3e-5 backbone / 5e-3 head, B16, E45,
ES10, 1 seed all match `train_mert_ft.py`.

**Clean results (§III.A).** AST accuracy 0.99124 (paper: 0.9912); max accuracy--macro-F1 gap
0.00189 (paper: "never differed by more than 0.002"); CNN single-seed 0.96201 $\pm$ 0.02072
(paper: 0.962 $\pm$ 0.021); CRNN 0.97325 $\pm$ 0.00251 (paper: 0.973 $\pm$ 0.003).

**Robustness at 20 dB (prose §IV.B).** human: AST 85.32, MERT 84.57, PANNs 83.59, CRNN 68.12,
SVM 67.97, CNN 66.33. environmental: AST 83.15, PANNs 81.18, MERT 76.92, SVM 62.28, CRNN 49.97,
CNN 46.41. white: AST 72.28, MERT 62.71, PANNs 38.64, CNN 33.40, CRNN 21.74, SVM 10.38.

**Robustness (§III.B, Table II).** All 24 AUC values match. AST highest in all three categories.
Lowest: SVM under white and human, CNN under environmental. AST 0.7228 and SVM 0.1038 at 20 dB
white (paper: 72.3\%, 10.4\%, 61.9-point difference). SVM 0.6797 under human at 20 dB (paper:
68.0\%). SVM 4th on clean, last under white --- verified by both AUC and the 20 dB point. Only
AST holds its rank across all three --- checked for all six.

**Failure analysis (§III.D, Conclusion).** tuba 0.5991 highest / violin 0.2147 lowest (paper:
0.599, 0.215); top three tuba, oboe, trumpet; tuba worst for 5 of 6 models under human noise
(PANNs' worst is french-horn); flute--oboe SVM white 0.342; 7 of 18 BH-significant; AST/human
$\rho=-0.6066$, $p=1.0\times10^{-5}$, $q=1.8\times10^{-4}$, the strongest. "Most consistently
under recorded noise" --- 6 of the 7 significant cells are human or environmental.

**Figures.** `fig6d_retention_compact.png`, `fig_recall_loss_heat_column.pdf`, and
`fig_distance_confusion_column.pdf` all exist in `docs/figures/`.

---

## Build note

All three `\includegraphics` calls use bare filenames, so the document only compiles from a
directory containing the images. Either add `\graphicspath{{../docs/figures/}}` to the preamble
or copy the three files next to `paper.tex`.
