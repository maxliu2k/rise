# Poster review — 2026 RISE symposium

Reviewed against the rendered PDF and the committed results at `eb3f1b8`. Items are ordered by
how much damage they do if left, not by how hard they are to fix.

---

## 1. CRITICAL — AUC is defined as retention but computed on macro-F1

**Where:** Figure 5 caption, and Eq. 9.

> Figure 5. "Area under the **retention**-vs-SNR curve (1.0 = no degradation)"
> Eq. 9. "Area Under Robustness Curve — Average **retention** under all eight noise levels"

The numbers in that table come from `scripts/noise_figures.py`, which computes

```python
grouped = curve(frames[name], noise_type, "macro_f1")      # absolute macro-F1
robustness_auc(grouped["snr_db"], grouped["mean"])
```

and labels its own axis `robustness AUC (dB-weighted mean macro-F1)`. **The plotted quantity is
absolute macro-F1, not retention.** A retention AUC would divide by the clean score and give
different numbers.

This matters because a reader who takes the definition literally will think AST's 0.6298 means
"AST retained 63% of its clean score under white noise". It actually means "AST's mean macro-F1
across the SNR range was 0.63", and since AST's clean score is 0.9908 those happen to be close —
which makes the error harder to catch, not less serious.

There is a second copy of this confusion: `robustness_curve.robustness_auc` in the library **is**
retention-based, so the poster's definition matches a function that was not used.

**Fix:** change both to "Area under the **macro-F1**-vs-SNR curve, dB-weighted (1.0 = perfect
classification at every SNR)". No number changes.

**Related trap:** the poster uses "AUC" for two quantities running in opposite directions —
Figure 5's AUC (higher = better) and the recall-loss AUC behind Figure 8 (higher = worse).
Figure 8 has been relabelled to say "recall lost" and avoid the word AUC entirely. Keep it that
way in the body text.

---

## 2. CRITICAL — the poster contradicts its own figure on what DEMAND is

Figure 6 column header and the Objectives text call it **"studio room-tone (DEMAND)"**.
Allan's Figure 9a labels the same corpus **"DEMAND environmental ambience"**.

The `studio` condition draws from **all 18 DEMAND environments**, which include `STRAFFIC`,
`TMETRO`, `TBUS`, `NPARK`, `NRIVER` and `PCAFETER`. Those are traffic, a metro carriage, a bus,
a park, a river and a café — recorded *inside* those spaces. Calling the condition "studio room
tone" describes maybe four of the eighteen.

Allan's own analysis plan flags this: *"paper text should use the descriptive corpus/category
names rather than implying … every DEMAND environment was a studio."*

**Fix, no rerun needed:** keep the short name `studio` as a label but define it once, e.g.

> studio: real-world ambience from all 18 DEMAND environments (domestic, nature, office,
> public, street, transport)

Then the name is a handle and the parenthetical is the definition, and the two figures agree.

---

## 3. MAJOR — model-vs-model claims have no intervals behind them

The Discussion asserts:

> "AST was the most robust model under all 3 noise types"
> "MERT swung from 2nd under white to last under audience, and PANNs fell to 3rd under white"

These are rank claims between point estimates. The only computed interval anywhere is CNN vs
CRNN (white, 20 dB): 0.1126 macro-F1, 95% pitch-group bootstrap [0.0560, 0.1625] and
[0.0589, 0.1601], both excluding zero — which is correctly reported in Statistical Analysis.

Meanwhile the Limitations section already concedes the problem:

> "SVM, AST, MERT and PANNs are single-seed runs … so small differences are not treated as
> effects"

So the poster states the caveat and then makes the claims anyway. A reviewer who reads both will
notice. The AST-vs-PANNs gap on audience is 0.7834 vs 0.7689 — 0.0145 — which is almost
certainly inside the seed spread.

**Fix (wording only):** "AST and PANNs led on every noise type, with AST highest in each" states
the same observation without asserting a tested difference. Reserve "significantly" for the
CNN/CRNN comparison that actually has an interval.

---

## 4. MODERATE — two overclaims in the Conclusion

> "**Universal benchmark** for instrument classification models, tested on six **frontier
> architectures**"

Both halves overreach, and the poster refutes them itself three inches to the right:

- *Universal* — one dataset (Philharmonia), isolated notes, 12 classes, one recording setup.
  The Limitations box says exactly this.
- *Frontier architectures* — an RBF SVM on 88 handcrafted features and a custom CNN are
  baselines, not frontier. AST/MERT/PANNs are the pretrained ones.

**Fix:** "A reproducible robustness benchmark spanning six model families, from handcrafted
baselines to audio-pretrained transformers."

Keeps the real contribution — the *span* is the interesting part — without a claim the
Limitations box contradicts.

---

## 5. Wording — quick passes

| where | current | suggested |
|---|---|---|
| Limitations | "tiled … which confound learning and repetition" | "tiled …, which **confounds content with repetition**" (subject-verb, and the original doesn't quite parse) |
| Discussion headers | "Clean Macro-F1 ≠ robustness" / "Pretrained Model ≠ robustness" | "Clean macro-F1 **does not predict** robustness" / "Pretraining **is not sufficient for** robustness" — `≠` between a metric and a property reads as a type error, and the second one is arguing something subtler than inequality |
| Conclusion | "Noise type equally important as noise strength (SNR ratio)" | "Noise **type matters as much as** SNR" — "SNR ratio" expands to "signal-to-noise-ratio ratio", and "equally" implies a test you did not run |
| Figure 5 | labelled "Figure" | it is a table — "Table 1" if the venue cares |
| Acknowledgments | "created in affiliation with Boston University RISE program" | "created **as part of** the Boston University RISE program" |

---

## 6. Verified correct — no action

Spot-checked against the committed results; these all hold:

- "5/6 models sat within 2 points on clean audio" — 0.9708–0.9908 excluding MERT ✓
- "SVM fell from 3rd to last (AUC 0.25)" — 3rd clean at 0.9788, last on white at 0.2531 ✓
- "MERT rose from last to 2nd (0.43)" — 6th clean at 0.8931, 2nd on white at 0.4321 ✓
- "AST … 72% vs 85%" at 20 dB — audience 0.845 / clean 0.9908 = 85.3% ✓
- "Degradation concentrates in tuba, oboe, and trumpet" — mean recall loss 0.62 / 0.58 / 0.50,
  the top three ✓ (french-horn is 4th at 0.47)
- "5 of 18 model-noise tests significant after Benjamini–Hochberg; strongest for AST under human
  non-speech (ρ = −0.607, q = 0.00018)" ✓
- Figure 8 caption numbers — tuba 0.62 lost, violin 43 clean errors / 0.19 lost, ρ = −0.16,
  p = 0.62 ✓

---

## 7. Typography — the most visible fix available

The PDF embeds **six faces**: Arial, Arial-Bold, Calibri, Calibri-Bold, Times New Roman,
Times New Roman-Bold. Every matplotlib figure adds a seventh, DejaVu Sans.

Four families on one board is the thing a passer-by registers before reading a word, and it is
invisible while making each piece because each looks fine alone.

`scripts/poster_style.py` now sets one style for every figure in the repo — Arial (already the
title face, and a sans holds up at poster distance where Times does not), grey frames instead of
black, 300 dpi instead of 200, and embedded TrueType so nothing is re-rasterised at print.

Re-render with:

```bash
python scripts/noise_figures.py
python scripts/confusion_grid.py --errors-only --require-all
python scripts/rank_slope.py
```

The body text still mixes Calibri and Times New Roman; picking one (Arial for headings, one
serif or sans for body — not both) would finish the job on the PowerPoint side.
