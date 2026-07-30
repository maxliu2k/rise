# Data Pipeline and Noise Benchmark: Implementation Audit and Methods Reference

## Start here: plain-language glossary

You do not need an audio-engineering or machine-learning background to read this document. The
shortest explanation of the project is:

1. A sound recording is stored as a long list of numbers describing how the air pressure changes.
2. The pipeline turns every retained recording into one example of exactly three seconds. That
   example is called a **window**.
3. Each model converts the same window into the kind of numbers it knows how to use.
4. The model predicts which instrument produced the window.
5. The noise experiment adds controlled noise to copies of the test windows and measures how much
   each model's predictions worsen.

In the current build, **one source recording produces exactly one window**. If the useful recording
is shorter than three seconds, the pipeline repeats it until the window is full. For example:

```text
original trimmed note:  [violin sound------]             1.2 seconds
three-second window:    [violin sound------][repeat][repeat part]
```

### Audio and window terms

| Term | Simple definition |
|---|---|
| source recording / source file | One original Philharmonia MP3. It is the starting audio before the pipeline changes it. |
| clip / example | One piece of audio presented to a model. In this project, the model example is normally one three-second window. |
| audio sample | One number measuring the waveform at one instant. It is not a complete recording or a dataset example. |
| waveform | The complete ordered list of audio samples. Plotting those numbers against time gives the familiar wavy audio shape. |
| amplitude | The height of a waveform sample. Larger absolute values generally mean a stronger instantaneous movement. |
| sample rate | How many waveform samples are stored per second. At 22,050 Hz, one second contains 22,050 numbers. |
| Hz / kHz | Hertz means “times per second.” `22.05 kHz` means 22,050 Hz. |
| mono / stereo | Mono has one audio channel. Stereo has separate left and right channels. This pipeline converts everything to mono. |
| channel | One stream of audio samples, such as the left side of a stereo recording. |
| duration | How long a recording lasts, measured here in seconds. |
| frequency | How quickly a sound vibration repeats. Low frequencies sound lower; high frequencies sound higher. |
| pitch / note | The perceived musical highness or lowness, represented here by names such as `A4` or `Cs5`. |
| fundamental frequency | The main repetition rate associated with the perceived note. Harmonics occur above it. |
| harmonic | A frequency related to the note's fundamental frequency. The pattern of harmonics helps distinguish instruments playing the same note. |
| timbre | The sound quality that makes, for example, a violin and flute sound different while playing the same pitch. |
| onset | The beginning of a note—the moment the instrument sound starts. |
| dynamic | A musical loudness marking such as piano, mezzo-forte, or fortissimo. |
| articulation / playing technique | How the note is produced, such as normal bowing, pizzicato, or a trill. The pipeline retains one configured technique per instrument. |
| phrase | A recording containing a musical sequence rather than one isolated note. |
| nominal length | The duration named in source metadata or the filename. It may differ from the exact decoded duration. |
| instrument family | A broad group such as strings, woodwinds, or brass. |
| window | One complete fixed-length model example. Here it is exactly 3.0 seconds or 66,150 samples at 22,050 Hz. This is different from a short analysis frame. |
| window function | A mathematical weighting, such as a Hann window, applied inside frequency analysis. It is unrelated to this document's three-second model example called a window. |
| frame | A much shorter piece of a window used to measure how its sound changes over time. Feature extraction uses many overlapping frames inside one window. |
| hop / hop length | How far the analysis moves before starting the next frame. Overlapping frames have a hop shorter than the frame itself. |
| crop | Select a fixed-length piece from a longer waveform. |
| trim | Remove quiet audio from the beginning and end of a recording. It does not remove quiet gaps in the middle. |
| silence threshold | The rule deciding whether an audio frame is quiet enough to trim. Here it is relative to each recording's loudest frame. |
| padding | Add made-up values, often zeros, to reach a required length. The current window pipeline does **not** zero-pad short notes. |
| tiling | Repeat real recorded audio until it reaches the required length. This is how the project fills a short three-second window. |
| active region | The part of a waveform that contains the target instrument sound. Noise diagnostics estimate it from clean-frame RMS within 30 dB of the loudest frame; the source pipeline does not store a human-annotated mask. |
| mask | A list of yes/no values marking which samples belong to a region of interest, such as active instrument audio. |
| resampling | Convert audio from one sample rate to another. The common pipeline converts 44.1 kHz audio to 22.05 kHz. |
| normalization | Rescale values to a chosen reference. Step 5 adjusts each window toward the same RMS loudness. |
| RMS | “Root mean square,” a measure of the typical waveform size. Squaring prevents positive and negative samples from cancelling each other. |
| power | The mean of the squared waveform samples. RMS is the square root of this power value. |
| peak | The largest absolute sample value in a waveform. |
| peak guard | A rule that stops loud samples from exceeding a chosen limit while trying to reach the target RMS. |
| clipping | Damage caused when a waveform exceeds the format's allowed range and its peaks are cut flat. |
| DC offset | A waveform shifted above or below zero even when it should be centered. It can incorrectly increase measured power. |
| decibel / dB | A logarithmic way to describe a ratio. For power, a 10 dB increase means a ten-times-larger ratio; 0 dB means equal power. |
| Nyquist frequency | The highest frequency a sample rate can represent: half the sample rate. At 22.05 kHz, it is 11.025 kHz. |
| codec | A method for encoding and decoding audio. MP3 is compressed and can introduce small artifacts; PCM WAV is uncompressed. |
| bitrate | Roughly how much encoded data an audio file uses per second. Different MP3 bitrates can leave different artifacts. |
| MP3 | A compressed audio format used by the original Philharmonia files. Compression removes information to reduce file size. |
| WAV | A container commonly used for uncompressed audio. The pipeline writes its processed audio as WAV files. |
| PCM16 | WAV samples stored as 16-bit signed integers. Its normal numeric range is limited, so excessive values clip. |
| float32 audio | Audio samples stored as 32-bit floating-point numbers. It can safely store noisy mixtures above the usual `-1` to `1` range without clipping. |

### Dataset and pipeline terms

| Term | Simple definition |
|---|---|
| dataset | The complete collection of examples and labels used in an experiment. |
| corpus | Another word for an organized collection of recordings, often referring to an external dataset such as ESC-50. |
| data pipeline | The ordered set of steps that turns downloaded recordings into model-ready data. |
| stage / step | One part of the pipeline, such as resampling, trimming, splitting, or feature extraction. |
| class | One answer category. This project has 12 instrument classes. |
| label / target | The correct class attached to an example, such as `violin`. |
| label index | The integer used in arrays instead of the class name. For example, index `0` always means `bassoon` in the current mapping. |
| metadata | Information describing audio rather than the audio samples themselves, such as path, instrument, note, duration, and split. |
| manifest | A CSV or JSON inventory saying which examples exist and where they are located. |
| split | One non-overlapping portion of the dataset: train, validation, or test. |
| training set / train split | The examples used to fit the model's learned values. |
| validation set / validation split | Separate examples used to choose settings or decide when training should stop. They are not used for the final reported score. |
| test set / held-out test | Examples kept hidden from model fitting and selection. They are used once at the end for the official clean result. |
| group | Several related files that must remain in the same split. |
| pitch group | All files with the same `(instrument, note)`, such as every retained `violin A4` recording. |
| stratification | Assigning splits separately within each instrument so every split contains every class in roughly similar proportions. |
| source-level assignment | Choosing a source recording's split before deriving model inputs from it. |
| near-duplicate | Two recordings that are not byte-for-byte identical but are so similar that testing on one after training on the other makes the task artificially easy. |
| data leakage | Training or model selection improperly gains information about validation or test examples. Near-duplicates crossing splits are one kind of leakage. |
| class imbalance | Some instruments have more examples than others. Plain accuracy can then be dominated by the larger classes. |
| class weight | A training multiplier that makes mistakes on less common classes count more. |
| random seed | A starting number for a pseudo-random process. Reusing the seed makes the same random choices when everything else is unchanged. |
| deterministic | Expected to produce the same output from the same input, configuration, software behavior, and seed. |
| configuration / config | The recorded settings controlling an experiment, such as labels, sample rate, split fractions, and window length. |
| hash / SHA-256 | A short digital identifier calculated from file contents. If the contents change, the hash should change. |
| fingerprint | Stored configuration and file hashes used to reject data built under incompatible settings. |
| provenance | The history of where an artifact came from: source files, settings, software, and processing steps. |
| schema | The required structure of a file, such as its column names and value types. |
| stale artifact | A generated file that belongs to an older dataset, configuration, or code path and should not be reused. |
| artifact | Any saved experimental output, including a feature array, model, summary, prediction table, or confusion matrix. |
| materialization | Saving derived data to disk rather than recreating it every time. The noise sweep materializes noisy WAVs. |
| cache | Saved intermediate results intended to avoid repeating expensive computation, such as MERT embeddings. |

### Features and model terms

| Term | Simple definition |
|---|---|
| feature | A number describing some property of audio. Examples include average spectral brightness or how often the waveform crosses zero. |
| feature vector | The complete list of features for one example. The SVM receives 88 numbers per window. |
| representation | The numeric form of audio given to a model. A handcrafted vector, log-mel image, or MERT embedding is a representation. |
| feature extraction / featurization | Convert a waveform into a representation a model can use. |
| standardization | Subtract the training mean and divide by the training standard deviation so feature scales are comparable. |
| mean | The arithmetic average. |
| standard deviation / std | A measure of how spread out values are around their mean. |
| train-only statistics | Means and standard deviations calculated only from training data, then reused unchanged for validation, test, and noise data. |
| time domain | Describing audio as amplitude changing over time—the ordinary waveform. |
| frequency domain | Describing how much energy occurs at different frequencies. |
| FFT | “Fast Fourier transform,” an algorithm that converts one short time-domain frame into frequency amounts. |
| STFT | “Short-time Fourier transform,” repeated FFTs on successive frames so frequency content can be followed over time. |
| spectrogram | A time-by-frequency table or image produced from successive frequency analyses. |
| mel scale | A frequency scale designed to give more detail to lower frequencies and less to very high frequencies. |
| mel bin | One frequency band on the mel scale. The CNN/CRNN representation uses 128 bins. |
| log-mel spectrogram | A mel spectrogram whose power values are put on a logarithmic/decibel-like scale. It is the image-like input for CNN and CRNN. |
| MFCC | “Mel-frequency cepstral coefficient,” a compact summary of the broad shape of a log-mel spectrum. |
| chroma | Twelve values describing energy associated with the twelve musical pitch classes, largely ignoring octave. |
| zero-crossing rate | How often the waveform changes sign. Noisy or bright sounds often cross zero more frequently. |
| spectral centroid | A rough frequency “center of mass,” often interpreted as brightness. |
| spectral bandwidth | How widely spectral energy is spread around its centroid. |
| spectral rolloff | The frequency below which a chosen percentage of spectral energy lies. |
| spectral contrast | A description of the difference between spectral peaks and valleys in frequency bands. |
| dimension | One numeric axis. An 88-dimensional feature vector contains 88 values. |
| shape | The size of every array axis. `(5864, 88)` means 5,864 examples with 88 features each. |
| tensor | A multi-axis numeric array—the general term neural-network libraries use for vectors, matrices, and higher-dimensional arrays. |
| model / classifier | A learned rule that turns an input representation into a predicted instrument. |
| baseline | A clearly defined reference system used for comparison. It does not mean unimportant or intentionally weak. |
| parameter / weight | A number learned by a model during training. |
| hyperparameter | A setting chosen outside ordinary training, such as SVM `C`, SVM `gamma`, or neural-network learning rate. |
| SVM | “Support vector machine,” a classifier that separates classes using selected training examples and a mathematical similarity function. |
| support vector | A training example that helps define an SVM's decision boundary. |
| kernel | The SVM's similarity function. This project uses the nonlinear RBF kernel. |
| RBF | “Radial basis function,” a kernel that treats examples as more similar when their feature vectors are close. |
| C | An SVM setting controlling the tradeoff between fitting training examples and allowing a smoother boundary. Larger is stricter about training errors. |
| gamma | An RBF setting controlling how local each example's influence is. Larger gamma creates narrower, more detailed decision regions. |
| neural network | A model made from layers of learned numerical transformations. |
| layer | One stage inside a neural network. Its output becomes input to the next stage. |
| CNN | “Convolutional neural network,” a network that learns local patterns. Here it analyzes log-mel time-frequency patterns. |
| CRNN | “Convolutional recurrent neural network,” combining local pattern detection with a component that follows changes over time. |
| transformer | A neural-network design that learns which parts of an input should pay attention to one another. AST and MERT use transformer-based components. |
| AST | “Audio Spectrogram Transformer,” a pretrained transformer that classifies an image-like spectrogram representation of audio. |
| MERT | A pretrained music-audio transformer. This project freezes it, combines representations from its layers, and trains a small instrument classifier on top. |
| PANNs | “Pretrained Audio Neural Networks,” a family trained on AudioSet. This project uses the CNN14 version. |
| CNN14 | A particular 14-layer-style PANNs convolutional architecture used to create or classify audio embeddings. |
| AudioSet | A large collection of labeled real-world audio clips used to pretrain AST and PANNs. It is not this project's Philharmonia dataset. |
| pretrained model | A model that already learned from a large external dataset before this project uses it. |
| backbone / encoder | The large part of a pretrained model that converts input audio into learned representations. |
| processor / feature extractor | Model-library code that converts a waveform to the sample rate, scale, and array layout expected by a pretrained model. |
| frozen | Kept unchanged during this project's training. A frozen MERT backbone does not update its pretrained weights. |
| embedding | A learned numeric summary of an example. Similar sounds may receive similar embeddings. |
| hidden state | An intermediate representation produced inside a neural-network layer. |
| pooling | Combine many time-step representations into one fixed-size summary, such as by taking their mean. |
| linear probe / classification head | A small classifier trained on top of frozen pretrained embeddings. It tests how useful those embeddings already are. |
| fine-tuning | Continue updating some or all weights of a pretrained model on this project's training data. |
| batch | A small group of examples processed together during one training update. |
| batch size | The number of examples in one batch. |
| epoch | One complete pass through the training set. |
| learning rate | The size of each neural-network weight update. Too large can be unstable; too small can learn very slowly. |
| early stopping | Stop training when validation performance stops improving, helping avoid overfitting. |
| checkpoint | Saved model weights from a particular point in training. |
| logit / score | A model's raw value for a class before conversion to a probability. |
| probability | A normalized confidence value. Not every model score is a trustworthy calibrated probability. |
| calibrated probability | A confidence value whose stated frequency matches reality reasonably well—for example, predictions near 80% confidence are correct about 80% of the time. |
| inference | Running a trained model to make predictions without updating it. |
| prediction | The class selected by the model for an example. |

### Noise and robustness terms

| Term | Simple definition |
|---|---|
| clean audio | The processed instrument window before experimental noise is added. “Clean” does not mean the original recording contains literally no background sound. |
| noisy / corrupted audio | A copy of a clean window after the experiment adds a controlled disturbance. |
| additive noise | Noise combined by sample-by-sample addition: `noisy = clean + scaled noise`. |
| corruption | A controlled alteration of an input. Additive noise is the corruption tested here. |
| robustness | How well a model keeps working when its input is corrupted. |
| white Gaussian noise | Random sample values drawn from a bell-shaped distribution, with roughly equal expected power across frequencies. |
| ESC-50 | An external dataset of environmental sound recordings used as the source of natural and mechanical noise. |
| natural noise | This project's broad group of selected ESC-50 animal, soundscape, and water recordings. |
| mechanical noise | This project's broad group of selected ESC-50 domestic, interior, exterior, and urban recordings. |
| noise segment / crop | The exact piece selected from a longer environmental-noise recording. |
| noise realization | The particular random waveform, source file, and crop used for one example. Another realization would sound different even at the same SNR. |
| noise gain | The multiplier applied to a noise waveform to reach a requested SNR. |
| SNR | “Signal-to-noise ratio,” the instrument power divided by added-noise power, usually reported in dB. Higher SNR means cleaner audio. |
| 20 dB SNR | Instrument power is 100 times the added-noise power under the measured definition. |
| 10 dB SNR | Instrument power is 10 times the added-noise power. |
| 0 dB SNR | Instrument and added noise have equal measured power. |
| negative SNR | Added noise has more measured power than the instrument. At -5 dB, noise power is about 3.16 times signal power. |
| whole-window SNR | Compute power across all three seconds, including quieter parts. This is what the current implementation uses. |
| active-region SNR | Compute power only where the instrument is considered active. The condition is still mixed using whole-window SNR, but `snr_signal_active_db` reports this diagnostic using an energy-derived frame mask. |
| in-band SNR | Compute SNR only within a chosen frequency range. It can reveal that low-frequency-heavy noise barely overlaps the useful instrument frequencies. |
| frequency band | A chosen range of frequencies, such as 200 Hz to 8 kHz. |
| band-limited measurement | A measurement that ignores frequencies outside a declared band. |
| spectral weighting | Give different frequencies different importance when measuring a sound. |
| noise colour | A description of how noise power is distributed across frequency. White, pink, and brown noise have different frequency balances. |
| pink noise | Random noise with less power at high frequencies than white noise. |
| brown noise | Random noise even more concentrated at low frequencies than pink noise. |
| broadband | Covering a wide range of frequencies. |
| transient | A short, sudden sound such as a knock, click, or impact. |
| model-effective SNR | SNR measured after a model's own input-rate resampling, representing what that model effectively receives. It is recorded for AST, MERT, and PANNs; it does not model their later internal feature transforms. |
| condition | One exact evaluation setting, such as `mechanical noise at 10 dB`. |
| grid | The complete list of conditions tested, such as all noise types crossed with all SNR levels. |
| full factorial design | Test every listed combination—for example, every noise type at every SNR—rather than testing only selected pairs. |
| replicate / repeat | Another independently drawn noise realization under the same condition. The frozen protocol uses three so noise-draw variability can be measured. |
| paired examples | Comparisons that use the same underlying test window and noisy realization for every model or condition. |
| augmentation | Altering training examples to teach a model about variation. The first robustness experiment does not train with noise. |
| clean-parity gate | A safety check requiring a noise evaluator to reproduce the official clean result before it may score noisy data. |
| adapter | Small model-specific code that connects a saved model to the shared noise-evaluation system. |

### Evaluation and statistics terms

| Term | Simple definition |
|---|---|
| evaluation | Measure a trained model's predictions against known correct labels. |
| metric | A number summarizing performance. |
| confusion matrix | A table counting each true instrument against each predicted instrument. Off-diagonal cells show which instruments were confused. |
| accuracy | Correct predictions divided by all predictions. A large class can dominate it. |
| precision for one class | Of everything predicted as that instrument, the fraction that truly was that instrument. |
| recall for one class | Of all real examples of that instrument, the fraction the model found correctly. |
| F1 for one class | A combined precision-and-recall score. It is high only when both are high. |
| macro-F1 | Calculate F1 separately for every instrument, then average the 12 values equally. This is the project's primary metric. |
| weighted F1 | Average class F1 scores while giving classes with more examples more weight. |
| balanced accuracy | Calculate recall separately for every class and average those recalls equally. |
| per-class result | A metric reported separately for each instrument rather than one overall average. |
| support | The number of true examples contributing to a class's reported metric. |
| MCC | “Matthews correlation coefficient,” a single classification score that considers all confusion-matrix cells. `1` is perfect, `0` is chance-like, and `-1` is completely opposed. |
| model selection | Use validation results to choose a model or hyperparameter setting. Test results must not influence this choice. |
| finalization | Freeze the validation choice, optionally refit according to the declared protocol, and perform the single official test evaluation. |
| sealed test / test-access guard | Code and procedure designed to stop people from repeatedly checking test results and changing the model in response. |
| overfitting | Learning details that work unusually well on seen data but do not generalize to new examples. |
| generalization | Performance on genuinely unseen examples rather than memorized training details. |
| confound | A second difference that changes with the factor being studied, making the true cause of a result unclear. |
| shortcut | An unintended easy clue, such as codec artifacts, that lets a model predict labels without learning instrument sound properly. |
| distribution shift | Test data differ from training data in some meaningful way, such as another recording collection, room, or microphone. |
| uncertainty | How much a reported result might vary if examples, noise draws, or training randomness changed. |
| cluster | A set of related examples treated as one unit in statistical resampling. Pitch group is the default cluster here. |
| bootstrap | Repeatedly resample the observed data to estimate how much a metric could vary. |
| confidence interval / CI | A range summarizing uncertainty from a stated procedure. A 95% bootstrap interval is not a guarantee that every future experiment lands inside it. |
| paired comparison | Compare two models or conditions on exactly the same examples, making the difference easier to interpret. |
| hypothesis test | A calculation asking whether an observed difference would be surprising under a stated no-difference assumption. |
| sign test | A paired test that counts how many clusters favor each side while ignoring the size of each win. |
| McNemar test | A paired test based on examples where one system is correct and the other is wrong. Ordinary window-level McNemar ignores related-example clustering. |
| statistical significance | Evidence that a difference is difficult to explain by the test's assumed random variation. It does not automatically mean the difference is large or useful. |
| multiple comparisons | Running many tests creates more chances for a false positive. The analysis should account for how many claims are tested. |
| false-discovery-rate control / FDR | A method for limiting the expected proportion of false positives among results called significant. |
| AUC / area under the curve | One number summarizing performance across an SNR range. Unequally spaced SNR values must be weighted by their actual spacing. |

### Repository and reproducibility terms

| Term | Simple definition |
|---|---|
| repository / repo | The Git-managed project containing code, configuration, documentation, and small artifacts. |
| dependency | An outside software package required by the project, such as scikit-learn or PyTorch. |
| virtual environment / venv | An isolated Python installation for one project, preventing its package versions from conflicting with other projects. |
| data root / `RISE_DATA_ROOT` | The top directory where this project reads and writes large datasets and generated data. |
| output directory | The folder where a run writes its model and result artifacts. |
| model revision | The exact saved version of a pretrained model. Pinning it prevents an online update from silently changing the experiment. |
| finalizer | A command that performs the declared one-time final fit/test procedure after validation selection is frozen. |
| fail closed | Stop with an error when required evidence is missing, instead of guessing or continuing with a warning. |
| regression test | An automated check designed to ensure a previously fixed bug does not return. |
| reproducibility | The ability to rebuild or verify the experiment from recorded data identities, code, settings, seeds, and software. |

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
12. The frozen grid is clean plus white, ESC-50 natural, and ESC-50 mechanical noise at
    60, 50, 40, 30, 20, 10, 0, and -10 dB.
13. The mixer uses whole-window power and writes float32 WAV to avoid clipping. It also records
    frequency-, time-, instrument-activity-, and model-rate-specific SNR diagnostics.
14. `N_REPLICATES` deterministic realizations per window and category are each rescaled across SNRs,
    and every adapter reads the same materialized files. The frozen value is 2.
15. The central mixer and SVM, MERT, PANNs, CNN, CRNN, and AST noise adapters exist.

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
  ([`NOISE_PLAN.md` L1–8](NOISE_PLAN.md#L1-L8)).
- **Noise-aware training/augmentation** would add noise during model fitting and require retraining.
  It is explicitly outside the current protocol
  ([`NOISE_PLAN.md` L212–215](NOISE_PLAN.md#L212-L215)).

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

In compact mathematical form, source recording $r_i$ first becomes a trimmed common-rate signal

$$
u_i
=
\operatorname{Trim}_{30\mathrm{dB}}
\left(
\operatorname{Resample}_{22.05\mathrm{kHz}}
\left(
\operatorname{MonoDecode}(r_i)
\right)
\right).
$$

Its retained window $k$ becomes the one canonical model example

$$
x_{i,k}
=
\operatorname{RMSNorm}_{0.1,\ \mathrm{peak}\le0.99}
\left(
\operatorname{CropOrTile}_{3\mathrm{s}}(u_i,k)
\right)
\in\mathbb R^{66{,}150}.
$$

Every clean model dataset is then the same labels and splits paired with a different representation
function:

$$
\mathcal D_m^{(q)}
=
\left\{
\left(g_m(x_{i,k}),\,y_i\right):
\operatorname{split}(i)=q
\right\},
\qquad
q\in\{\mathrm{train},\mathrm{val},\mathrm{test}\}.
$$

Sections 13–15 define the actual $g_m$ for SVM, CNN, CRNN, AST, MERT, and PANNs. This is why the
models are comparable: they do not receive independently split audio; they receive different
mathematical views of the same canonical windows.

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
| `docs/NOISE_PLAN.md` | Fixed noise protocol and commands | Sections 1–14 | CURRENT |
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
- `midi`: $12(o+1)+p$, so A4 is 69
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

**VERIFIED IMPLEMENTATION.** The source of truth is the ordered `TARGET_LABELS` list in
[`config.py` L32–42](../src/instrument_robustness/config.py#L32-L42). If $\mathcal L_k$ denotes its
$k$th entry, the integer target for a label $\ell$ is

$$
y(\ell)=k\quad\Longleftrightarrow\quad \mathcal L_k=\ell.
$$

The resulting numerical mapping is:

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

$$
g_i=(\mathrm{label}_i,\mathrm{note}_i).
$$

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
([`step3_split.py` L80–92](../src/instrument_robustness/step3_split.py#L80-L92)). In the current
configuration, Step 4 produces exactly one window per source, and that window inherits its source's
split tag. Step 4 then rebuilds the `(label,note)` grouping from the window metadata and asserts
that no pitch group crosses splits
([`step4_window.py` L93–139](../src/instrument_robustness/step4_window.py#L93-L139)).

Randomly splitting the current windows would still be invalid even though there is only one window
per source. Different source recordings of the same instrument and note—including recordings at
different dynamics or nominal lengths—can be near-duplicates. A file- or window-level random split
could therefore place one version in training and a closely related version in validation or test.
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

- 3.0 seconds equal exactly $3(22050)=66{,}150$ samples;
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

**VERIFIED IMPLEMENTATION.** Step 2 applies librosa's edge trimmer with a 30 dB threshold
([`step2_trim.py` L18–29](../src/instrument_robustness/step2_trim.py#L18-L29)). With the installed
librosa 0.11 defaults, this uses 2,048-sample frames, a 512-sample hop, maximum frame RMS as the
reference, and an RMS frame calculation. For frame $k$,

$$
\operatorname{RMS}(k)=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{k,i}^{2}},\qquad N=2048.
$$

Librosa converts RMS amplitude to decibels relative to the recording's maximum frame RMS,
conceptually

$$
D(k)=20\log_{10}
\left(
\frac{\max(\operatorname{RMS}(k),a_{\min})}
{\max(\max_j \operatorname{RMS}(j),a_{\min})}
\right),
$$

and treats frames with $D(k)>-30$ dB as non-silent. The output is the contiguous interval between
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

> **Verified implementation limitation:** The clean preprocessing metadata stores no human-derived
> frame activity mask or active interval. At noise-generation time, `active_signal_snr_db` derives
> clean activity from frame RMS and records the active fraction, active-frame count, and SNR in
> provenance. Empty decoded files are rejected in Step 1, but an effectively silent nonempty
> recording is not explicitly rejected.
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

Let a trimmed source waveform be $u_i\in\mathbb{R}^{N_i}$. The window and hop lengths are

$$
W=H=3(22050)=66{,}150\text{ samples}.
$$

Candidate window $k$ begins at

$$
a_k=kH
$$

and contains

$$
c_{i,k}=\min(W,N_i-a_k)
$$

recorded samples. The first candidate is always retained. A later candidate is retained only when
$c_{i,k}\ge 0.5(22050)=11{,}025$. When $c_{i,k}<W$, the retained segment is tiled:

$$
x_{i,k}[t]=u_i\!\left[a_k+(t\bmod c_{i,k})\right],
\qquad 0\le t<W.
$$

This equation is the precise meaning of “repeat the short note until the window is full.” It adds
no zero-valued padding; it reuses recorded samples cyclically. When $c_{i,k}=W$, the modulo has no
effect and the window is the ordinary source slice $u_i[a_k:a_k+W]$.

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

For a window $x$, Step 5 calculates

$$
r=\sqrt{\frac{1}{T}\sum_{t=1}^{T}x_t^2},\qquad
g_0=\frac{0.1}{r}.
$$

If $\max_t |g_0x_t|>0.99$, it changes the gain to

$$
g=g_0\frac{0.99}{\max_t|g_0x_t|};
$$

otherwise $g=g_0$. It writes $x'=gx$ back to the same WAV path as PCM16
([`step5_normalize.py` L21–33](../src/instrument_robustness/step5_normalize.py#L21-L33)). A window
with RMS below $10^{-6}$ is left unchanged.

**VERIFIED FROM METADATA.** Median post-normalization RMS is 0.10000. Fifteen of 9,116 windows are
more than 0.001 below target because of the peak guard; the minimum post-RMS is 0.05325.

Waveform normalization and feature standardization are different:

- waveform RMS normalization changes audio samples independently per window;
- feature standardization later changes each feature coordinate using statistics fitted on train.

The order matters for SNR. The implemented mixer measures the actual final clean-window power, so
it does not simply assume $0.1^2$. It does not normalize again after mixing
([`NOISE_PLAN.md` L82–92](NOISE_PLAN.md#L82-L92)).

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

For window $i$, the canonical waveform is

$$
x_i\in\mathbb{R}^{66{,}150}.
$$

`featurelib.svm_vector` divides the waveform into overlapping analysis frames and calculates
several descriptor sequences. If descriptor $j$ has values $f_{i,t,j}$ over $T_j$ frames, the
window keeps only its temporal mean and population standard deviation
([`featurelib.py` L30–61](../src/instrument_robustness/featurelib.py#L30-L61)):

$$
\mu_{i,j}
=
\frac{1}{T_j}
\sum_{t=1}^{T_j} f_{i,t,j}
$$

and

$$
\sigma_{i,j}
=
\sqrt{
\frac{1}{T_j}
\sum_{t=1}^{T_j}
\left(f_{i,t,j}-\mu_{i,j}\right)^2
}.
$$

For descriptor group $g$ with $D_g$ rows, define

$$
\operatorname{MS}(F_i^{(g)})
=
\left[
\mu_{i,1}^{(g)},\ldots,\mu_{i,D_g}^{(g)},
\sigma_{i,1}^{(g)},\ldots,\sigma_{i,D_g}^{(g)}
\right].
$$

These summaries turn a variable number of frames into one fixed vector, concatenated in the exact
implementation order:

$$
v_i=
\operatorname{concat}\!\left(
\operatorname{MS}(\mathrm{MFCC}_i),
\operatorname{MS}(\mathrm{chroma}_i),
\operatorname{MS}(\mathrm{centroid}_i),
\operatorname{MS}(\mathrm{bandwidth}_i),
\operatorname{MS}(\mathrm{rolloff}_i),
\operatorname{MS}(\mathrm{contrast}_i),
\operatorname{MS}(\mathrm{ZCR}_i),
\operatorname{MS}(\mathrm{RMS}_i)
\right)
\in\mathbb{R}^{88}.
$$

The 88 coordinates are:

| Feature group | Frame dimensions | Saved summaries | Vector dimensions | Mathematical idea |
|---|---:|---|---:|---|
| MFCC | 20 | mean and std per coefficient | 40 | cosine transform of the log-mel spectral envelope |
| Chroma STFT | 12 | mean and std per pitch class | 24 | spectral power folded into 12 pitch classes |
| Spectral centroid | 1 | mean and std | 2 | power-weighted mean frequency |
| Spectral bandwidth | 1 | mean and std | 2 | power-weighted frequency spread |
| Spectral rolloff | 1 | mean and std | 2 | frequency containing the default 85% of spectral energy |
| Spectral contrast | 7 bands | mean and std per band | 14 | peak-versus-valley energy in each frequency band |
| Zero-crossing rate | 1 | mean and std | 2 | fraction of neighboring samples that change sign |
| RMS energy | 1 | mean and std | 2 | square root of mean frame power |
| **Total** |  |  | **88** |  |

Some representative frame-level definitions are:

$$
\operatorname{centroid}(t)=
\frac{\sum_k f_kP(t,k)}{\sum_kP(t,k)},
$$

$$
\operatorname{bandwidth}(t)=
\sqrt{
\frac{\sum_kP(t,k)\left(f_k-\operatorname{centroid}(t)\right)^2}
{\sum_kP(t,k)}
},
$$

$$
f_{\mathrm{rolloff}}(t)=
\min\left\{f_q:
\sum_{k:f_k\le f_q}P(t,k)
\ge 0.85\sum_kP(t,k)
\right\},
$$

$$
\operatorname{ZCR}(t)=
\frac{1}{N-1}\sum_{n=1}^{N-1}
\mathbf{1}\!\left[
\operatorname{sign}(x_t[n])\ne\operatorname{sign}(x_t[n-1])
\right],
$$

$$
\operatorname{RMS}(t)=
\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x_t[n]^2}.
$$

MFCCs first describe the log-mel spectral envelope, then decorrelate it with a cosine transform.
For mel-band log energies $L(t,r)$, the idea is

$$
\operatorname{MFCC}_q(t)
=
\sum_{r=0}^{R-1}
L(t,r)
\cos\left[
\frac{\pi q}{R}\left(r+\frac12\right)
\right].
$$

The spectral functions use `N_FFT=2048` and `HOP=512`; MFCC count is 20
([`config.py` L105–114](../src/instrument_robustness/config.py#L105-L114)). Parameters not passed
explicitly—such as rolloff percentage or MFCC DCT choices—are the installed librosa defaults and
should be version-pinned before final paper reproduction.

Step 6 fits one dataset-level mean and standard deviation for each of the 88 vector coordinates
using the $N_{\mathrm{train}}=5864$ train windows only:

$$
\mu_d^{\mathrm{train}}
=
\frac{1}{N_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}v_{i,d},
$$

$$
\sigma_d^{\mathrm{train}}
=
\sqrt{
\frac{1}{N_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}
\left(v_{i,d}-\mu_d^{\mathrm{train}}\right)^2
}.
$$

Standard deviations below $10^{-8}$ are replaced by 1
([`step6_stats.py` L39–76](../src/instrument_robustness/step6_stats.py#L39-L76)). Step 7 applies:

$$
z_{i,d}
=
\frac{v_{i,d}-\mu_d^{\mathrm{train}}}
{\sigma_d^{\mathrm{train}}},
\qquad z_i\in\mathbb{R}^{88},
$$

to train, validation, and test without refitting
([`step7_featurize.py` L56–74](../src/instrument_robustness/step7_featurize.py#L56-L74)).
`train_svm.py` deliberately loads these already-standardized arrays without a second scaler.

Thus the saved matrices are

$$
X_{\mathrm{train}}\in\mathbb{R}^{5864\times88},
\quad
X_{\mathrm{val}}\in\mathbb{R}^{1259\times88},
\quad
X_{\mathrm{test}}\in\mathbb{R}^{1255\times88}.
$$

Every `X` is float32 and `y` is int64. Keys are `X`, `y`, `source_path`, `feature_names`,
`label_names`, and `config_fingerprint`. The NPZ does not store `window_path`, so ordering is tied
to the filtered order of `windows.csv`; multiple windows from one source repeat `source_path`.

For noisy SVM inference, the adapter recomputes the same raw 88 features from each noisy waveform,
loads the saved Step-6 means/stds, and applies them
([`noise_eval_svm.py` L41–82](../src/instrument_robustness/noise_eval_svm.py#L41-L82)). It never fits
statistics on noisy data. The selected SVM compares standardized vectors with the radial-basis
kernel

$$
K(z_i,z_j)=\exp\left(-\gamma\lVert z_i-z_j\rVert_2^2\right).
$$

Noise therefore affects the SVM only through how it changes the waveform-derived vector $z_i$.

## 14. Log-mel pipeline for CNN and CRNN

For the CNN and CRNN, every canonical waveform

$$
x_i\in\mathbb{R}^{66{,}150}
$$

is converted into a time-frequency matrix. The actual parameters are 22,050 Hz sample rate,
2,048-sample Hann FFT frames, 512-sample hop (23.22 ms), 128 mel bands, 0–11,025 Hz frequency
range, power exponent 2, and centered STFT frames. Centering introduces only the FFT's boundary
padding; it does not change the stored 3-second waveform contract. Let $\widetilde x_i$ be $x_i$
with 1,024 zeros added at each edge for centered analysis.

([`featurelib.py` L20–27](../src/instrument_robustness/featurelib.py#L20-L27)). Conceptually:

$$
X_i(m,k)
=
\sum_{n=0}^{2047}
\widetilde x_i[n+512m]\,w[n]e^{-j2\pi kn/2048},
$$

$$
P_i(m,k)=|X_i(m,k)|^2,
$$

$$
M_i(m,r)
=
\sum_kH_r(k)P_i(m,k),
\qquad r=1,\ldots,128,
$$

where $H_r(k)$ is the $r$th triangular mel filter. The number of centered time frames is

$$
T
=
1+\left\lfloor
\frac{66{,}150+2(1024)-2048}{512}
\right\rfloor
=130.
$$

The repository then converts mel power to decibels with reference power 1.0. Before the 80 dB
floor, this is

$$
Q_i(m,r)
=
10\log_{10}\left(\max(M_i(m,r),10^{-10})\right).
$$

Librosa's default dynamic-range limit then gives

$$
L_i(m,r)
=
\max\left(
Q_i(m,r),
\max_{m',r'}Q_i(m',r')-80
\right).
$$

The reference is still 1.0; the maximum appears only in the 80 dB floor.

Step 6 pools every frame of every train window to fit a separate mean and standard deviation for
each mel bin ([`step6_stats.py` L45–68](../src/instrument_robustness/step6_stats.py#L45-L68)).
For mel bin $r$, the fit contains

$$
N_{\mathrm{train}}T
=
5864(130)
=762{,}320
$$

train-frame values. Its statistics are

$$
\mu_r^{\mathrm{train}}
=
\frac{1}{N_{\mathrm{train}}T}
\sum_{i\in\mathrm{train}}\sum_{m=1}^{T}L_i(m,r),
$$

$$
\sigma_r^{\mathrm{train}}
=
\sqrt{
\frac{1}{N_{\mathrm{train}}T}
\sum_{i\in\mathrm{train}}\sum_{m=1}^{T}
\left(L_i(m,r)-\mu_r^{\mathrm{train}}\right)^2
}.
$$

Every split then uses

$$
\widehat L_i(m,r)
=
\frac{L_i(m,r)-\mu_r^{\mathrm{train}}}
{\sigma_r^{\mathrm{train}}}.
$$

No validation, test, or noisy frame contributes to $\mu_r^{\mathrm{train}}$ or
$\sigma_r^{\mathrm{train}}$.

Current float32 arrays have shapes:

| Split | CNN `X` | `y` |
|---|---|---|
| train | `(5864, 128, 130, 1)` | `(5864,)` |
| validation | `(1259, 128, 130, 1)` | `(1259,)` |
| test | `(1255, 128, 130, 1)` | `(1255,)` |

The CNN view adds a singleton channel dimension:

$$
\widehat L_i\in\mathbb{R}^{128\times130}
\longrightarrow
X_i^{\mathrm{CNN}}\in\mathbb{R}^{128\times130\times1}.
$$

The CRNN does not generate new features. It transposes the same matrix so time becomes the sequence
axis:

$$
X_i^{\mathrm{CRNN}}[m,r]
=
X_i^{\mathrm{CNN}}[r,m,0],
\qquad
X_i^{\mathrm{CRNN}}\in\mathbb{R}^{130\times128}.
$$

`crnn_data.load_crnn` performs the transpose and verifies the fingerprint
([`crnn_data.py` L13–23](../src/instrument_robustness/crnn_data.py#L13-L23)). There is no CNN or
CRNN model/training script on current `main`; only their inputs and CRNN loader exist.

Noise must be added in the linear waveform domain before this transform. Adding an arbitrary matrix
to a clean log-mel tensor would not equal a physical waveform mixture. In general,

$$
\operatorname{logmel}(x+\alpha n)
\ne
\operatorname{logmel}(x)+\alpha\operatorname{logmel}(n),
$$

because squaring, mel filtering, the logarithm, and standardization are not jointly linear.

## 15. Raw-waveform path for pretrained models

All pretrained paths start from the same Step-5 mono 3.0-second, 22.05 kHz window. They do **not**
use Step-6 SVM/log-mel statistics
([`pretrained_extractors.py` L1–16](../src/instrument_robustness/pretrained_extractors.py#L1-L16)).

| Model | Current implementation status | What is generated from each window | When/where it is stored |
|---|---|---|---|
| AST | Clean loader/trainer and clean artifacts implemented; no noise adapter | Resample to 16 kHz, then `ASTFeatureExtractor` produces `(1024,128)` | Generated on demand by the dataset; no feature NPZ |
| MERT | Clean extraction/probe/finalizer, artifacts, and noise adapter implemented | Resample to 24 kHz; frozen MERT produces 13 time-pooled 768-D states | Train/validation are cached as `features/mert/{split}.npz`; test extraction is sealed until finalization |
| PANNs CNN14 | Probe/fine-tune code and noise adapter implemented; no local checkpoint/results | Resample to 32 kHz; CNN14 computes 64-bin log-mel internally and yields a 2,048-D embedding | Probe mode caches `features/panns/emb_{split}.npz`; fine-tuning generates inputs on demand |

Let $\mathcal R_{22.05\rightarrow q}$ denote polyphase resampling from 22.05 kHz to model rate
$q$. Because every source window is exactly 3 seconds,

$$
x_i^{(q)}
=
\mathcal R_{22.05\rightarrow q}(x_i)
\in\mathbb{R}^{3q}.
$$

The three waveform lengths are therefore

$$
\begin{aligned}
x_i^{\mathrm{AST}}&\in\mathbb{R}^{48{,}000},\\
x_i^{\mathrm{MERT}}&\in\mathbb{R}^{72{,}000},\\
x_i^{\mathrm{PANNs}}&\in\mathbb{R}^{96{,}000}.
\end{aligned}
$$

### AST

AST's feature extractor maps the resampled waveform to a fixed spectrogram-like tensor:

$$
A_i
=
\Phi_{\mathrm{AST}}\!\left(x_i^{\mathrm{AST}}\right)
\in\mathbb{R}^{1024\times128}.
$$

The 1,024 axis is AST's fixed time/patch input length and the 128 axis is its mel-frequency
dimension. The DataLoader stacks examples as

$$
A_{\mathrm{batch}}\in\mathbb{R}^{B\times1024\times128}.
$$

The repository delegates AST's padding, truncation, and normalization to the pretrained
`ASTFeatureExtractor`, then fine-tunes all classifier parameters. Thus AST is fed a waveform by the
repository loader, but the transformer itself receives the processor-generated matrix.

AST resampling/processing is in
[`pretrained_extractors.py` L54–84](../src/instrument_robustness/pretrained_extractors.py#L54-L84).
`ASTWindowDataset` strictly checks 66,150 source samples before processing
([`ast_data.py` L100–115](../src/instrument_robustness/ast_data.py#L100-L115)). The model processor
owns padding/truncation and normalization. The repository does not hard-code its normalization
values.

### MERT

For each MERT hidden-state index $\ell\in\{0,\ldots,12\}$, the frozen backbone produces

$$
H_i^{(\ell)}\in\mathbb{R}^{T_{\mathrm{MERT}}\times768}.
$$

The repository mean-pools model time:

$$
e_i^{(\ell)}
=
\frac{1}{T_{\mathrm{MERT}}}
\sum_{\tau=1}^{T_{\mathrm{MERT}}}
H_i^{(\ell)}[\tau,:]
\in\mathbb{R}^{768}.
$$

Stacking all 13 pooled states gives the cached representation

$$
E_i
=
\begin{bmatrix}
e_i^{(0)}\\
\vdots\\
e_i^{(12)}
\end{bmatrix}
\in\mathbb{R}^{13\times768}.
$$

The probe learns one scalar parameter $\theta_\ell$ per hidden state and converts them to normalized
layer weights:

$$
a_\ell
=
\frac{e^{\theta_\ell}}
{\sum_{j=0}^{12}e^{\theta_j}},
\qquad
\sum_{\ell=0}^{12}a_\ell=1.
$$

It mixes the states and produces 12 class logits:

$$
\bar e_i
=
\sum_{\ell=0}^{12}a_\ell e_i^{(\ell)}
\in\mathbb{R}^{768},
$$

$$
o_i^{\mathrm{MERT}}
=
W\bar e_i+b
\in\mathbb{R}^{12}.
$$

MERT uses a commit-pinned `m-a-p/MERT-v1-95M` processor and backbone
([`config.py` L116–121](../src/instrument_robustness/config.py#L116-L121)). Equal-length 3 s
waveforms become 72,000 samples at 24 kHz; `padding=True` is still passed for batching
([`pretrained_extractors.py` L103–114](../src/instrument_robustness/pretrained_extractors.py#L103-L114)).
Each hidden state is mean-pooled over model time
([`extract_mert.py` L65–96](../src/instrument_robustness/extract_mert.py#L65-L96)).

### PANNs

PANNs receives the 96,000-sample waveform and computes its own internal 64-bin log-mel
representation. Its CNN14 trunk maps that representation to

$$
p_i
=
\Phi_{\mathrm{CNN14}}\!\left(x_i^{\mathrm{PANNs}}\right)
\in\mathbb{R}^{2048},
$$

followed by the project classifier

$$
o_i^{\mathrm{PANNs}}
=
W_pp_i+b_p
\in\mathbb{R}^{12}.
$$

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

**VERIFIED IMPLEMENTATION.** Let $\mathcal D_{\mathrm{tr}}$, $\mathcal D_{\mathrm{val}}$, and
$\mathcal D_{\mathrm{te}}$ be the clean train, validation, and test sets. Model fitting and selection
obey

$$
\widehat\theta_h
=
\operatorname{fit}\!\left(\mathcal D_{\mathrm{tr}};h\right),
\qquad
h^\star
=
\arg\max_h
F_{1,\mathrm{macro}}\!\left(
\widehat\theta_h,\mathcal D_{\mathrm{val}}
\right).
$$

The selected model is frozen before any noise test. Noise is then applied only to test waveforms.
The condition sets are

$$
\mathcal C=\{\text{white},\text{natural},\text{mechanical}\},
\qquad
\mathcal S=\{20,10,5,0,-5\}\ \mathrm{dB}.
$$

Therefore each test example is evaluated under

$$
1+|\mathcal C||\mathcal S|
=1+3(5)=16
$$

conditions: one canonical clean input and 15 noisy inputs. With $N_{\mathrm{test}}=1{,}310$, the
generator materializes $1{,}310(15)=19{,}650$ noisy WAVs. Train and validation windows are never
noised by the generator. The clean condition is not copied; evaluators read the canonical Step-5
WAVs. The constants are defined in
[`noise_sweep.py` L47–64](../src/instrument_robustness/noise_sweep.py#L47-L64).

Before noisy scoring, the common evaluator must reproduce the official clean test example count and
macro-F1 within $10^{-3}$, or it aborts
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
range. Repository download instructions are in [`NOISE_PLAN.md` L184–195](NOISE_PLAN.md#L184-L195);
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

**VERIFIED IMPLEMENTATION.** For window $i$ and the white-noise category, the deterministic random
generator draws

$$
n_i[t]\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
\qquad
t=0,\ldots,T-1,
\qquad
T=66{,}150.
$$

Thus the population moments are

$$
\mathbb E[n_i[t]]=0,
\qquad
\operatorname{Var}(n_i[t])=1,
\qquad
\mathbb E[n_i[t]^2]=1.
$$

The realized finite vector generally has

$$
\bar n_i=\frac1T\sum_t n_i[t]\ne0,
\qquad
P_{n_i}=\frac1T\sum_t n_i[t]^2\ne1.
$$

The mixer therefore uses the measured $P_{n_i}$, not the theoretical value 1
([`noise_sweep.py` L206–218](../src/instrument_robustness/noise_sweep.py#L206-L218)). Samples are
cast to float32 before scaling.

Downloaded white-noise recordings are unnecessary because the random generator and stable seed
fully define the realization. The source provenance is `generated_gaussian`, source rate 22,050 Hz,
and crop start 0.

## 20. External noise-segment selection

For `natural` or `mechanical`, the seeded RNG first chooses one path uniformly from the appropriate
sorted 800-file ESC-50 pool. Let its decoded waveform be $r$. Stereo is converted to mono by channel
averaging, then the result is resampled to 22,050 Hz:

$$
r^{(22.05)}
=
\mathcal R_{q\rightarrow22.05}(r)
\in\mathbb R^L.
$$

If $L<T$, it is extended periodically rather than zero-padded:

$$
\widetilde r[t]=r^{(22.05)}[t\bmod L]
\quad\text{until}\quad
\operatorname{length}(\widetilde r)\ge T,
\qquad T=66{,}150.
$$

Let $L'$ be the resulting length. A crop offset is drawn uniformly over every valid start,

$$
o\sim\operatorname{Uniform}\{0,1,\ldots,L'-T\},
$$

and the unscaled noise realization is

$$
n_i[t]=\widetilde r[o+t],
\qquad t=0,\ldots,T-1.
$$

A candidate is accepted only if

$$
\operatorname{RMS}(n_i)
=
\sqrt{\frac1T\sum_{t=0}^{T-1}n_i[t]^2}
\ge10^{-6}.
$$

The implementation makes at most 20 attempts, then returns the selected source, original source
rate, and the crop start in **resampled** sample coordinates
([`noise_sweep.py` L189–241](../src/instrument_robustness/noise_sweep.py#L189-L241)).

> **Verified implementation deviation from the proposed generic recipe:** No DC-offset removal is
> performed. The stored crop offset is measured in the **resampled** 22.05 kHz waveform, not in the
> original file's sample coordinates. Crop end is implicit as `start + 66150`, not stored.

The RNG seed determines both source-file selection and crop start. The same draw is reused across
all SNR levels for that clean window and noise type.

## 21. SNR calculation: whole-window mixing plus active-instrument reporting

### General active-region definition

If a window contains padding or long silence, whole-window power can understate the instrument's
power during its sounding region. Let

$$
A=\{t:\text{sample }t\text{ belongs to active instrument audio}\}.
$$

An active-region definition is:

$$
P_x^{(A)}=\frac{1}{|A|}\sum_{t\in A}x_t^2,\qquad
P_n^{(A)}=\frac{1}{|A|}\sum_{t\in A}n_t^2,
$$

$$
\operatorname{SNR}_{\mathrm{dB}}^{(A)}
=10\log_{10}\left(\frac{P_x^{(A)}}{P_n^{(A)}}\right),
\qquad
\alpha=\sqrt{\frac{P_x^{(A)}}{P_n^{(A)}10^{s/10}}},
$$

$$
y_t=x_t+\alpha n_t.
$$

Here $x$ is the clean waveform, $n$ is unscaled noise, $s$ is requested SNR in dB,
$\alpha$ is noise gain, and $y$ is the mixture.

### What this repository actually does

> **Verified implementation:** whole-window SNR sets the noise gain; active-instrument SNR is
> measured and stored as an additional diagnostic.

`mix_at_snr` uses every one of the $T=66{,}150$ samples:

$$
P_x=\frac{1}{T}\sum_{t=1}^{T}x_t^2,\qquad
P_n=\frac{1}{T}\sum_{t=1}^{T}n_t^2,
$$

$$
\rho_s=10^{s/10},
\qquad
\alpha_s=\sqrt{\frac{P_x}{P_n\rho_s}},
\qquad
y_s=x+\alpha_s n.
$$

The scaled noise has exactly the target power before floating-point/storage error:

$$
P_{\alpha_s n}
=
\frac1T\sum_t(\alpha_s n_t)^2
=
\alpha_s^2P_n
=
\frac{P_x}{10^{s/10}}.
$$

Therefore

$$
10\log_{10}\left(\frac{P_x}{P_{\alpha_s n}}\right)
=
10\log_{10}\left(10^{s/10}\right)
=s.
$$

After saving and reloading the noisy WAV, the implementation measures the added component as
$y_s-x$:

$$
\widehat s
=
10\log_{10}
\left(
\frac{\frac1T\sum_t x_t^2}
{\frac1T\sum_t(y_{s,t}-x_t)^2}
\right).
$$

The generator requires

$$
\left|\widehat s-s\right|<0.1\ \mathrm{dB}.
$$

The exact code is [`noise_sweep.py` L244–258](../src/instrument_robustness/noise_sweep.py#L244-L258);
measurement uses the same whole-array definition
([`noise_sweep.py` L261–268](../src/instrument_robustness/noise_sweep.py#L261-L268)).

The same noise realization $n$ is reused along an SNR curve; only its gain changes. For two levels,

$$
\frac{\alpha_{s_1}}{\alpha_{s_2}}
=
10^{(s_2-s_1)/20}.
$$

For example, the 0 dB version uses $10^{10/20}\approx3.162$ times the noise amplitude of the 10 dB
version. This isolates the effect of intensity from the random identity of the noise.

The tiling repair is important here: canonical windows have no synthetic zero-padded tail, so
whole-window power covers repeated trimmed signal rather than a mixture of instrument and added
zeros. However, it still includes naturally quiet portions within each repeated segment.

If a typical normalized clean window has $P_x=0.01$, the target powers are:

| SNR $s$ | $P_x/P_{\alpha n}$ | Target added-noise power $P_{\alpha n}$ | Noise RMS $\sqrt{P_{\alpha n}}$ |
|---:|---:|---:|---:|
| 20 dB | $100$ | $0.000100$ | $0.010000$ |
| 10 dB | $10$ | $0.001000$ | $0.031623$ |
| 5 dB | $3.162$ | $0.003162$ | $0.056234$ |
| 0 dB | $1$ | $0.010000$ | $0.100000$ |
| -5 dB | $0.316$ | $0.031623$ | $0.177828$ |

At 20 dB, signal power is 100 times noise power. At 0 dB they are equal.

For reporting, the clean and added waveforms are split into 2,048-sample frames with a 512-sample
hop. If $r_k$ is the clean RMS in frame $k$, active frames are

$$
\mathcal{K}_{\mathrm{active}}
=\left\{k:r_k\geq \max_j(r_j)\,10^{-30/20}\right\}.
$$

The code averages clean and added-noise frame power over this set and records
`snr_signal_active_db`, plus `signal_active_fraction` and `snr_signal_active_frames`. Thus a paper
must call the requested condition **whole-window power SNR**, while it may separately report the
**energy-derived active-instrument SNR**. This mask is algorithmic, not manually annotated.

## 22. Clipping prevention

Adding noise can make $\max_t|y_t|>1$. Hard clipping,
$\operatorname{clip}(y,-1,1)$, is nonlinear: it changes waveform shape and destroys the requested
signal/noise power relationship.

A possible common peak-protection alternative is

$$
\beta=\min\left(1,\frac{p_{\max}}{\max_t|y_t|}\right),\qquad y'_t=\beta y_t.
$$

Applying $\beta$ to the complete mixture preserves SNR because signal and noise receive the same
gain.

> **Verified implementation:** The noise generator does **not** apply $\beta$, hard clipping, or
> post-mix normalization. It writes `subtype="FLOAT"` WAV, which preserves values beyond
> $[-1,1]$, and records the reloaded peak
> ([`noise_sweep.py` L334–346, L516–553](../src/instrument_robustness/noise_sweep.py#L334-L346)).
> The manifest records `post_mix_normalization: false`.

The unit test writes and reloads a float WAV whose samples are 2.5, verifying that the shared reader
does not clamp it ([`tests/test_noise.py` L188–200](../tests/test_noise.py#L188-L200)).

> **Potential validity concern:** Float headroom preserves mathematical SNR, but values above the
> usual normalized waveform range enter model processors. The clean-parity gate cannot test this
> noisy-range behavior. The final Methods should report float32 storage and lack of post-mix scaling
> explicitly.

## 23. Stable seeds and deterministic regeneration

Let $F$ be the dataset fingerprint, $w$ the window ID, and $c$ the noise category. The current
seed construction is

$$
h=\operatorname{SHA256}\!\left(F\mathbin\Vert\texttt{"|"}\mathbin\Vert
w\mathbin\Vert\texttt{"|"}\mathbin\Vert c\right),
$$

$$
\operatorname{seed}
=
\operatorname{uint32}_{\mathrm{big\ endian}}(h_0,h_1,h_2,h_3),
$$

where $\Vert$ denotes string concatenation and $h_0,\ldots,h_3$ are the first four digest bytes.

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
- SNR—the same realization is intentionally rescaled across the curve. Replicate number is included
  when more than one independent realization is configured.

Identical dataset files, window ID, noise type, NumPy behavior, ESC-50 inventory, and software path
should regenerate the same sample values. The noise manifest records relevant software versions,
corpus hashes, per-output hashes, seed, source, and crop. This supports detection of drift even if a
future library version changes byte-level regeneration.

## 24. Mixture manifest

The corruption is defined centrally by `work/windows_noisy/noise_manifest.json`, which contains one
completion/build/protocol record, and `work/windows_noisy/noise_provenance.csv`, which contains one
row per noisy WAV.

The JSON is written last and is the completion marker
([`noise_sweep.py` L560–605](../src/instrument_robustness/noise_sweep.py#L560-L605)). It records
protocol version, complete state, dataset identity, SNRs/types, test/file counts, waveform format,
seed scheme, ESC-50 corpus provenance, provenance hash, and software versions.

Each CSV row identifies the clean window and hashes it; identifies and hashes the noise source;
records category, requested SNR, seed, original noise rate, resampled crop start, $\alpha$, clean and
unscaled-noise powers, realized SNR, peak, output path, and output hash.

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
conditions ([`NOISE_PLAN.md` L23–34](NOISE_PLAN.md#L23-L34)).

> **Verified implementation:** The official path is offline materialization under
> `$RISE_DATA_ROOT/work/windows_noisy/`. `--validate` dynamically creates only a few checks and
> writes listenable preview WAVs; `--generate` materializes the complete benchmark. Evaluators read
> the completed files and refuse a missing or partial manifest.

## 26. Model-specific noisy-input generation

For clean window $x_i$, category $c$, and SNR $s$, the central generator creates exactly one
materialized waveform

$$
y_{i,c,s}
=
x_i+\alpha_{i,c,s}n_{i,c}
\in\mathbb R^{66{,}150}.
$$

Every model receives this same $y_{i,c,s}$. Only the deterministic model-specific representation
$g_m$ differs:

$$
\begin{aligned}
g_{\mathrm{SVM}}(y)
&=
\operatorname{standardize}_{\mathrm{train}}
\!\left(\operatorname{handcrafted}_{88}(y)\right)
\in\mathbb R^{88},\\
g_{\mathrm{CNN}}(y)
&=
\operatorname{standardize}_{\mathrm{train}}
\!\left(\operatorname{logmel}_{128\times130}(y)\right)
\in\mathbb R^{128\times130\times1},\\
g_{\mathrm{CRNN}}(y)
&=
\operatorname{transpose}\!\left(g_{\mathrm{CNN}}(y)\right)
\in\mathbb R^{130\times128},\\
g_{\mathrm{AST}}(y)
&=
\Phi_{\mathrm{AST}}\!\left(
\mathcal R_{22.05\rightarrow16}(y)
\right)
\in\mathbb R^{1024\times128},\\
g_{\mathrm{MERT}}(y)
&=
\operatorname{meanpool}_{t,\ell}
\!\left(
\Phi_{\mathrm{MERT}}\!\left(
\mathcal R_{22.05\rightarrow24}(y)
\right)
\right)
\in\mathbb R^{13\times768},\\
g_{\mathrm{PANNs}}(y)
&=
\Phi_{\mathrm{CNN14}}\!\left(
\mathcal R_{22.05\rightarrow32}(y)
\right)
\in\mathbb R^{2048}.
\end{aligned}
$$

The prediction for model $m$ is

$$
\widehat y_{i,c,s}^{(m)}
=
\arg\max_{k\in\{0,\ldots,11\}}
f_m\!\left(g_m(y_{i,c,s})\right)_k.
$$

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

$$
F_{1,\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}
\frac{2\,\mathrm{precision}_k\,\mathrm{recall}_k}
{\mathrm{precision}_k+\mathrm{recall}_k},
\qquad K=12,
$$

with zero division set to zero. Accuracy is also saved. Per-class precision/recall/F1 and confusion
matrices are implemented. AST clean evaluation additionally writes family-level performance;
the shared noise evaluator does not aggregate by family.

The implemented degradation quantities are:

$$
\Delta F_1(s,c)=F_{1,\mathrm{clean}}-F_{1,c,s},
\qquad
R_{F_1}(s,c)=\frac{F_{1,c,s}}{F_{1,\mathrm{clean}}},
$$

where $c$ is noise category and $s$ is SNR
([`noise_eval_common.py` L351–356](../src/instrument_robustness/noise_eval_common.py#L351-L356)).
A positive $\Delta F_1$ means performance worsened.

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

> **Scope:** Three noise replicates now measure noise-realization variability, and cluster bootstrap
> quantifies sampling uncertainty across pitch groups. There is still no repeated
> model-training-seed field in the common output contract, so neither measure quantifies
> training-seed uncertainty.

## 28. Data leakage and validity checklist

| Risk | Repository status | Evidence / limitation |
|---|---|---|
| Source recordings crossing clean splits | **PREVENTS IT** | one source row receives one group assignment |
| Pitch groups crossing clean splits | **PREVENTS IT** | Step 3 and Step 4 assert `(label,note)` isolation; the current build has one window per source |
| Normalization fit on validation/test | **PREVENTS IT** | Step 6 selects only `split=="train"` |
| Feature-standardization leakage | **PREVENTS IT** | saved train stats reused; fingerprinted; SVM no second scaler |
| Noise recordings crossing noise splits | **DOES NOT YET ADDRESS IT** | no external-noise split/catalog |
| Different models receiving different random mixtures | **PREVENTS IT** | one centrally materialized, hashed noisy set |
| Test data used for model selection | **UNCLEAR** | SVM/MERT sealed; AST/PANNs use validation but access test inputs earlier |
| Noise category imbalance | **UNCLEAR** | equal 800-file top-level pools and full factorial mixtures; subcategories not audited/balanced |
| Pitch imbalance | **DOES NOT YET ADDRESS IT** | pitch groups are isolated, not balanced; instrument ranges differ |
| Articulation imbalance | **UNCLEAR** | one articulation/class mitigates technique count; normal vs arco-normal still family-linked |
| Dynamic imbalance | **DOES NOT YET ADDRESS IT** | retained and grouped with pitch, not stratified/balanced |
| Silence/content-fraction imbalance | **MEASURES, DOES NOT BALANCE** | tiling removes zero padding; active-instrument fraction/SNR are recorded, but content duration is not balanced by class |
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

> **Scope of the table:** preprocessing still stores no human-annotated active duration. The trim
> and `content_s` values above must not be relabeled as active audio. Noise generation separately
> derives frame activity and records it per mixture; those diagnostics will exist only after the
> official noise sweep is generated.

### Array shapes

| Representation | Train | Validation | Test | Dtype |
|---|---|---|---|---|
| SVM `X` | `(5864,88)` | `(1259,88)` | `(1255,88)` | float32 |
| SVM/CNN `y` | `(5864,)` | `(1259,)` | `(1255,)` | int64 |
| CNN `X` | `(5864,128,130,1)` | `(1259,128,130,1)` | `(1255,128,130,1)` | float32 |
| CRNN view | `(5864,130,128)` | `(1259,130,128)` | `(1255,130,128)` | float32 |
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
10. Broader real-data validation of the energy-derived active-instrument threshold.
11. End-to-end clean-parity tests for each neural noise adapter with its real checkpoint.

## 31. Worked example

This repository-grounded clean example is the first current `windows.csv` row:

| Stage/property | Value |
|---|---|
| Source | `bassoon/A2/bassoon_A2_025_forte_normal.mp3` |
| Duration after trim | 0.3135 s |
| Canonical window | `work/windows/bassoon/A2/bassoon_A2_025_forte_normal_w000.wav` |
| Window size | $22{,}050\ \mathrm{Hz}\times3.0\ \mathrm{s}=66{,}150$ samples |
| Short-window rule | repeat the 0.3135 s segment cyclically until 3.0 s |
| RMS before Step 5 | 0.05489 |
| RMS after Step 5 | 0.10000 |

For an actual 10 dB **white-noise** condition:

1. Load the exact Step-5 PCM16 clean window.
2. Derive the seed from dataset fingerprint, `bassoon_A2_025_forte_normal_w000`, and `white`.
3. Draw 66,150 standard-normal float32 samples.
4. Measure whole-window clean and noise power.
5. Calculate $\alpha$ for 10 dB.
6. Add $y=x+\alpha n$.
7. Do not apply peak scaling or clipping.
8. Save float32 WAV.
9. Reload it and verify achieved whole-window SNR differs from 10 dB by less than 0.1 dB.
10. Save seed, powers, gain, peak, clean/output hashes, and output path.
11. Recompute the model's representation from this saved noisy waveform.

A small numerical illustration close to this non-peak-guarded normalized window is:

$$
P_x=\operatorname{RMS}(x)^2=0.1^2=0.01,
\qquad
P_n=1,
\qquad
s=10\ \mathrm{dB}.
$$

$$
\rho_{10}=10^{10/10}=10,
$$

$$
\alpha_{10}
=
\sqrt{\frac{P_x}{P_n\rho_{10}}}
=
\sqrt{\frac{0.01}{1(10)}}
=0.0316228.
$$

The scaled noise and mixture are

$$
n_{10}[t]=0.0316228\,n[t],
\qquad
y_{10}[t]=x[t]+n_{10}[t],
$$

and their powers are

$$
P_{n_{10}}
=
\alpha_{10}^2P_n
=0.001,
\qquad
\operatorname{RMS}(n_{10})
=\sqrt{0.001}
=0.0316228.
$$

The resulting SNR is therefore

$$
10\log_{10}\left(\frac{P_x}{P_{n_{10}}}\right)
=
10\log_{10}\left(\frac{0.01}{0.001}\right)
=10\ \mathrm{dB}.
$$

This is illustrative because a finite Gaussian draw has power near, not exactly, 1, and the code
uses the measured powers. It also uses the actual clean power rather than assuming 0.01.

After the noisy WAV is saved, each implemented model regenerates its own input from the same
$y_{10}$:

$$
\begin{aligned}
\text{SVM:}&\quad y_{10}\rightarrow v\in\mathbb R^{88}
\rightarrow z\in\mathbb R^{88}\rightarrow\widehat y,\\
\text{MERT:}&\quad y_{10}\rightarrow
\mathcal R_{22.05\rightarrow24}(y_{10})
\rightarrow E\in\mathbb R^{13\times768}\rightarrow\widehat y,\\
\text{PANNs:}&\quad y_{10}\rightarrow
\mathcal R_{22.05\rightarrow32}(y_{10})
\rightarrow p\in\mathbb R^{2048}\rightarrow\widehat y.
\end{aligned}
$$

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
windows, and between “code exists” and “experiment was run.” Do not describe active-instrument SNR
as the mixing target, and do not describe external noise splits as implemented.

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
- The frozen noise grid is white/natural/mechanical at 60, 50, 40, 30, 20, 10, 0, -5, -10, and
  -15 dB plus clean, with three independent noise realizations.
- Only clean test windows were corrupted; clean train/validation data and fitted models remained
  unchanged.
- Noise gain was calculated from mean power over the entire fixed window. Band, segmental,
  active-instrument, and model-effective SNR were recorded alongside it.
- Each configured noise replicate was deterministic per dataset build/window/category/replicate and
  was rescaled across SNRs.
- No post-mix normalization or hard clipping was used; noisy audio was stored as float32 WAV.
- The same materialized noisy WAV was intended for every model.
- SVM, MERT, PANNs, AST, CNN, and CRNN noise adapters exist; real-checkpoint parity and sealed-test
  status still differ by model.

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
5. A replicate axis exists, but the configured count is still one and no across-realization
   uncertainty analysis has been frozen.
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

## 35. Reproduction commands

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
python -m unittest discover -s tests -p 'test_preprocessing.py' -v
```

The generated array shapes expected after a successful build are already listed in Section 29.

### Run focused tests

```bash
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
