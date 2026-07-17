"""Shared feature extractors used by BOTH Step 6 (stats) and Step 7 (featurize), and later by
the noise experiments. Keeping these in one place is what guarantees clean and noisy features
are produced by identical code.
"""
import warnings
import numpy as np, librosa
from instrument_robustness.config import SR, N_FFT, HOP, N_MELS, N_FRAMES, N_MFCC
warnings.filterwarnings("ignore")


def load_window(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    # windows are already exactly 3.0 s; enforce length defensively
    target = int(round(3.0 * SR))
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target]


def logmel(y):
    """log-mel spectrogram, fixed shape (N_MELS, N_FRAMES). Not standardized here."""
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmax=SR // 2)
    M = librosa.power_to_db(S, ref=1.0).astype(np.float32)
    if M.shape[1] < N_FRAMES:
        M = np.pad(M, ((0, 0), (0, N_FRAMES - M.shape[1])))
    return M[:, :N_FRAMES]


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

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    cent = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    roll = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)  # 7 bands
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)

    vec = np.concatenate([
        ms(mfcc), ms(chroma),
        [cent.mean(), cent.std(), bw.mean(), bw.std(), roll.mean(), roll.std()],
        ms(contrast),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std()],
    ]).astype(np.float32)
    return vec
