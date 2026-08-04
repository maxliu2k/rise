# Frozen failure-analysis plan

This plan must be read with `NOISE_PLAN.md`. It is fixed before the corrected six-model rerun is
complete. The implementation is `failure_analysis.py`; its outputs are derived from saved
predictions and training features, so it does not retrain models, rerun inference, or alter the
noise corpus.

Run it only after all six noise adapters have completed:

```bash
python -m instrument_robustness.failure_analysis
```

The command fails unless SVM, CNN, CRNN, MERT, AST, and PANNs all contain:

- the clean condition and all 48 noisy conditions;
- the current configuration fingerprint and the exact sealed SCC dataset identity
  `97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf`;
- one shared noise-manifest hash;
- the same window, truth, pitch-group, and noise-draw identities; and
- one unchanged model identity within each sweep.

The three frozen noise categories are generated white noise, ESC-50 human non-speech events
(`audience`), and DEMAND environmental ambience (`studio`). The code tags are retained in output
filenames; paper text should use the descriptive corpus/category names rather than implying that
every ESC-50 event occurred in an audience or every DEMAND environment was a studio.

The active data root must independently reconstruct that same complete identity. The SVM training
feature file must also match the exact SHA-256 recorded by the final SVM run. These gates make the
command reject the retired local build and all earlier invalid result folders even when their row
counts happen to match.

## Analysis 1: which instruments lose recall?

**Status: exploratory.** Macro-F1 remains the headline measure of model robustness. This analysis
uses recall only to describe which instruments account for that degradation.

For instrument $c$, noise category $n$, SNR $s$, and replicate $r$, recall loss is

$$
L_{c,n,s,r}=R_{c,\mathrm{clean}}-R_{c,n,s,r}.
$$

A positive value means recall became worse; a negative value means it happened to improve. Values
are not clipped. Each complete eight-SNR curve is summarized by

$$
A^{\mathrm{recall}}_{c,n,r}
=\frac{1}{s_{\max}-s_{\min}}
\int_{s_{\min}}^{s_{\max}}L_{c,n,s,r}\,ds,
$$

using trapezoidal integration in dB. The two replicate-specific areas, their mean, and their range
are retained. No SNR is selected after looking at the results.

Five acoustic summaries are computed from **training recordings only**:

1. mean spectral centroid, in Hz;
2. mean spectral bandwidth, in Hz;
3. mean spectral rolloff, in Hz;
4. spectral contrast, averaged over the seven contrast bands, in dB; and
5. MFCC-profile distance: the Euclidean distance between an instrument's training centroid and the
   overall training centroid over the 40 standardized MFCC mean/standard-deviation coordinates.

For each model and noise category, each summary is related to mean recall-loss area using Spearman
correlation. Two-sided $p$-values use 100,000 deterministic permutations of the 12 complete
instrument identities (seed 0). These 90 associations are exploratory and are reported as such;
their uncorrected $p$-values are not used to declare confirmatory discoveries. There are only 12
instrument observations, so effect direction and size matter more than a threshold crossing.

## Analysis 2: do acoustically similar instruments become confused?

**Status: primary failure-mechanism analysis.** This analysis uses all

$$
\binom{12}{2}=66
$$

unordered instrument pairs.

### Acoustic distance

The already standardized 88-dimensional SVM features from the **training split only** are used. For
instrument $a$, its centroid is

$$
\boldsymbol{\mu}_a=\frac{1}{N_a}\sum_{i:y_i=a}\mathbf{x}_i.
$$

The distance between instruments $a$ and $b$ is

$$
D_{a,b}=\left\|\boldsymbol{\mu}_a-\boldsymbol{\mu}_b\right\|_2.
$$

This is distance in the project's handcrafted representation. It is not claimed to be human
perceptual distance.

### Noise-induced pair confusion

Confusions are row-normalized so unequal class counts do not dominate. For pair $a,b$,

$$
C_{a,b}(n,s,r)=\frac{1}{2}
\left[
P(\hat y=b\mid y=a)+P(\hat y=a\mid y=b)
\right].
$$

The model's clean confusion is removed:

$$
\Delta C_{a,b}(n,s,r)=C_{a,b}(n,s,r)-C_{a,b}(\mathrm{clean}).
$$

The full fixed curve is summarized as

$$
A^{\mathrm{conf}}_{a,b,n,r}
=\frac{1}{s_{\max}-s_{\min}}
\int_{s_{\min}}^{s_{\max}}\Delta C_{a,b}(n,s,r)\,ds.
$$

Replicate-specific areas and their range are retained. Their predeclared mean is the value used by
the primary test.

### Primary statistical test

For every model and noise category, Spearman correlation relates $D_{a,b}$ to mean
$A^{\mathrm{conf}}_{a,b,n}$. The alternative is two-sided. The null distribution uses 100,000
deterministic permutations (seed 0) of complete instrument identities: each permutation moves the
same label on the matrix row and column together. The 66 pair values are never shuffled
independently because pairs that share an instrument are dependent.

The declared correction family contains exactly

$$
6\ \text{models}\times3\ \text{noise categories}=18\ \text{tests}.
$$

Benjamini-Hochberg correction controls false-discovery rate over those 18 two-sided permutation
tests at $q=0.05$. Correlation coefficients, raw permutation $p$-values, corrected $q$-values, and
the complete pair tables are all retained. A correlation does not establish that acoustic distance
caused the failures.

## Outputs

Outputs are written under `artifacts/failure_analysis/`:

- training class centroids and the five acoustic summaries;
- all 66 acoustic distances;
- Analysis 1 condition-level recall losses, replicate curve areas, summaries, and exploratory
  associations;
- Analysis 2 condition-level pair confusions, replicate curve areas, summaries, and 18 corrected
  primary tests; and
- `analysis_manifest.json`, which records the protocol, hashes, identities, software versions, and
  output hashes.

The command refuses to overwrite an existing result directory unless `--overwrite` is explicitly
given. A rerun should normally go to a new archived directory instead of overwriting the paper's
analysis.
