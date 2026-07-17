"""Step 6 - Compute normalization statistics on the TRAIN split ONLY.

- SVM handcrafted vectors: per-feature mean + std (z-score).
- CNN/CRNN log-mel: PER-BIN (per mel band) mean + std  [documented choice: per-bin, not global].
- AST / MERT are excluded (they carry their own extractors/normalization).
Stats are computed strictly over TRAIN windows and saved to norm_stats.{npz,json}; val, test and
the later noisy set all reuse these exact numbers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per worker -> avoid oversubscription

import json, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from instrument_robustness.config import ROOT, PIPE, STATS_NPZ, STATS_JSON, N_MELS, N_FRAMES
from instrument_robustness.featurelib import load_window, svm_vector, logmel, SVM_FEATURE_NAMES
warnings.filterwarnings("ignore")


def _feats(wrel):
    # accumulate mel sums here so only ~small arrays cross the process boundary
    y = load_window(ROOT / wrel)
    M = logmel(y).astype(np.float64)
    return svm_vector(y), M.sum(axis=1), (M ** 2).sum(axis=1), M.shape[1]


def main():
    win = pd.read_csv(PIPE / "windows.csv")
    train = win[win.split == "train"]["window_path"].tolist()
    print(f"computing TRAIN-ONLY stats over {len(train)} train windows ...")

    svm_rows = []
    # streaming accumulators for per-bin log-mel mean/std
    s1 = np.zeros(N_MELS, np.float64)      # sum
    s2 = np.zeros(N_MELS, np.float64)      # sum of squares
    cnt = 0
    done = 0
    with ProcessPoolExecutor() as ex:
        for vec, m1, m2, nf in ex.map(_feats, train, chunksize=32):
            svm_rows.append(vec)
            s1 += m1
            s2 += m2
            cnt += nf
            done += 1
            if done % 1500 == 0:
                print(f"  {done}/{len(train)}")

    X = np.vstack(svm_rows)
    svm_mean = X.mean(axis=0)
    svm_std = X.std(axis=0)
    svm_std[svm_std < 1e-8] = 1.0          # guard constant features

    mel_mean = (s1 / cnt).astype(np.float32)
    mel_var = (s2 / cnt) - (s1 / cnt) ** 2
    mel_std = np.sqrt(np.maximum(mel_var, 1e-8)).astype(np.float32)

    np.savez(STATS_NPZ,
             svm_mean=svm_mean.astype(np.float32), svm_std=svm_std.astype(np.float32),
             svm_feature_names=np.array(SVM_FEATURE_NAMES),
             logmel_mean=mel_mean, logmel_std=mel_std,
             computed_on="train", n_train_windows=len(train))
    with open(STATS_JSON, "w") as f:
        json.dump({
            "computed_on": "train ONLY",
            "n_train_windows": len(train),
            "svm_dim": int(X.shape[1]),
            "svm_feature_names": SVM_FEATURE_NAMES,
            "svm_mean": svm_mean.round(6).tolist(),
            "svm_std": svm_std.round(6).tolist(),
            "logmel_standardization": "per-bin (per mel band)",
            "logmel_n_mels": N_MELS, "logmel_n_frames": N_FRAMES,
            "logmel_mean_per_bin": mel_mean.round(4).tolist(),
            "logmel_std_per_bin": mel_std.round(4).tolist(),
        }, f, indent=2)

    print(f"\nSVM vector dim: {X.shape[1]}  (mean/std over {X.shape[0]} train windows)")
    print(f"log-mel per-bin stats: mean {N_MELS} bins, std {N_MELS} bins "
          f"(over {cnt} train frames)")
    print(f"wrote {STATS_NPZ.name} and {STATS_JSON.name}  [TRAIN ONLY]")


if __name__ == "__main__":
    main()
