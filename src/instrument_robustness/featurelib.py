"""Shared feature extractors used by BOTH Step 6 (stats) and Step 7 (featurize), and later by
the noise experiments. Keeping these in one place is what guarantees clean and noisy features
are produced by identical code.

Because this module is the SHARED clean/noisy path, anything here that depends on the individual
window becomes a confound in the noise sweep at every SNR. Two librosa defaults did exactly that
and are overridden; see config.STFT_PAD_MODE and config.LOGMEL_TOP_DB for the measurements.

  pad_mode  librosa's "constant" pads n_fft//2 zeros at each edge, so the first and last frames of
            every window are computed over synthesised digital silence -- the exact thing step 4
            tiles to avoid, reintroduced at a smaller dose.
  top_db    librosa's 80 clamps to (this window's peak - 80 dB). It covered 16.8% of a clean image
            and 0.0% of a noisy one at every SNR down to -5 dB, so clean and noisy features were
            not produced by the same transform at all.
"""
import warnings
import numpy as np, librosa
from instrument_robustness.config import (SR, WINDOW_S, N_FFT, HOP, N_MELS, N_FRAMES,
                                          N_MFCC, STFT_PAD_MODE, LOGMEL_TOP_DB)
warnings.filterwarnings("ignore")


def load_window(path):
    """One Step-5 window as a waveform of exactly WINDOW_S seconds.

    Raises: ValueError if the file is not exactly that length.

    This used to zero-pad a short window instead of raising. Step 4 guarantees every window is
    exactly WINDOW_S (it tiles short notes rather than padding them), so that branch was dead --
    but had step 4 ever regressed, padding would have injected the one thing this pipeline exists
    to keep out, and the symptom would have been a mysteriously flat noise sweep rather than a
    crash. Fail loudly instead.
    """
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    target = int(round(WINDOW_S * SR))
    if len(y) != target:
        raise ValueError(
            f"{path} is {len(y)} samples, expected exactly {target} ({WINDOW_S}s at {SR}Hz). "
            f"Step 4 should have produced a fixed-length window; do not pad it here.")
    return y


def logmel(y):
    """log-mel spectrogram, shape exactly (N_MELS, N_FRAMES). Not standardized here.

    Preconditions: y is exactly WINDOW_S seconds at SR (use load_window).
    Postcondition: returns a (N_MELS, N_FRAMES) float32 array.
    Raises: ValueError if the frame count is not N_FRAMES.

    Two non-default arguments, both there to keep this transform independent of the individual
    window -- see config for the measurements behind each:

      pad_mode=STFT_PAD_MODE ("reflect")  librosa's "constant" default pads 1024 ZEROS at each
                                          edge, computing the first and last frames of every
                                          window over synthesised silence.
      top_db=LOGMEL_TOP_DB (None)         librosa's default of 80 floors everything below this
                                          window's own peak minus 80 dB. That clamp covers 16.8%
                                          of a clean image and 0.0% of a noisy one at every SNR,
                                          so it made clean and noisy features incomparable.

    This used to zero-pad when a window yielded fewer than N_FRAMES frames. That branch was
    unreachable (66150 samples at hop 512 is always exactly 130 frames) and it zero-padded, which
    is the thing the rest of the pipeline exists to avoid. If it ever becomes reachable, that is a
    step-4 regression and it should crash rather than quietly pad.
    """
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmax=SR // 2, pad_mode=STFT_PAD_MODE)
    M = librosa.power_to_db(S, ref=1.0, top_db=LOGMEL_TOP_DB).astype(np.float32)
    if M.shape[1] != N_FRAMES:
        raise ValueError(
            f"log-mel has {M.shape[1]} frames, expected exactly {N_FRAMES}. A {WINDOW_S}s window "
            f"at hop {HOP} must yield {N_FRAMES}; do not pad or truncate to hide the difference.")
    return M


# ---- SVM handcrafted vector ----
SVM_FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)] +
    [f"chroma{i}_mean" for i in range(12)] + [f"chroma{i}_std" for i in range(12)] +
    ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
     "rolloff_mean", "rolloff_std"] +
    [f"contrast{i}_mean" for i in range(7)] + [f"contrast{i}_std" for i in range(7)] +
    ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]
)  # total = 40 + 24 + 6 + 14 + 4 = 88


def svm_vector(y):
    """One fixed-length (88,) handcrafted vector per window. Not standardized here."""
    def ms(a):  # mean+std across time, per row
        return np.concatenate([a.mean(axis=1), a.std(axis=1)])

    pm = {"pad_mode": STFT_PAD_MODE}     # every framed feature, not just the log-mel
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, **pm)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, **pm)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, **pm)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, **pm)
    roll = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, **pm)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, **pm)
    # zero_crossing_rate takes no pad_mode -- it is time-domain and frames the waveform directly.
    # It is 2 of the 88 features and has no STFT edge to corrupt, so it is left as-is.
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP, **pm)

    vec = np.concatenate([
        ms(mfcc), ms(chroma),
        [cent.mean(), cent.std(), bw.mean(), bw.std(), roll.mean(), roll.std()],
        ms(contrast),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std()],
    ]).astype(np.float32)
    return vec
