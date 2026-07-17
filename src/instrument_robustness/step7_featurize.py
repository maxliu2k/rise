"""Step 7 (a-c) - Featurize windows for the non-pretrained models: SVM, custom CNN, CRNN.

- Uses the SAME featurelib functions everywhere (so clean/noisy will match later).
- Applies Step-6 TRAIN-ONLY stats to every split (train/val/test identically).
- Saves per split, per model. Labels are integer-encoded by TARGET_LABELS order.
7a SVM  -> features/svm/{split}.npz   X (N,88) standardized, y, source_path, feature_names
7b CNN  -> features/cnn/{split}.npz   X (N,128,130,1) per-bin standardized log-mel, y
7c CRNN -> reuses 7b (see features/crnn/README.txt + crnn_data.py loader)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from instrument_robustness.config import ROOT, PIPE, FEATURES, STATS_NPZ, N_MELS, N_FRAMES, TARGET_LABELS
from instrument_robustness.featurelib import load_window, svm_vector, logmel, SVM_FEATURE_NAMES
warnings.filterwarnings("ignore")

LABEL2IDX = {lab: i for i, lab in enumerate(TARGET_LABELS)}


def _feats(wrel):
    y = load_window(ROOT / wrel)
    return svm_vector(y), logmel(y)


def main():
    st = np.load(STATS_NPZ, allow_pickle=True)
    svm_mean, svm_std = st["svm_mean"], st["svm_std"]
    mel_mean = st["logmel_mean"][:, None]     # (128,1) broadcast over frames
    mel_std = st["logmel_std"][:, None]

    win = pd.read_csv(PIPE / "windows.csv")
    (FEATURES / "svm").mkdir(parents=True, exist_ok=True)
    (FEATURES / "cnn").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        rows = win[win.split == split].reset_index(drop=True)
        paths = rows["window_path"].tolist()
        y = np.array([LABEL2IDX[l] for l in rows["label"]], dtype=np.int64)
        print(f"[{split}] featurizing {len(paths)} windows ...")

        Xsvm = np.empty((len(paths), len(SVM_FEATURE_NAMES)), np.float32)
        Xmel = np.empty((len(paths), N_MELS, N_FRAMES), np.float32)
        with ProcessPoolExecutor() as ex:
            for i, (vec, M) in enumerate(ex.map(_feats, paths, chunksize=32)):
                Xsvm[i] = vec
                Xmel[i] = M

        # 7a SVM: z-score with train stats
        Xsvm = (Xsvm - svm_mean) / svm_std
        np.savez(FEATURES / "svm" / f"{split}.npz",
                 X=Xsvm, y=y, source_path=rows["source_path"].values,
                 feature_names=np.array(SVM_FEATURE_NAMES), label_names=np.array(TARGET_LABELS))

        # 7b CNN: per-bin standardize with train stats, add channel axis
        Xmel = (Xmel - mel_mean) / mel_std
        Xcnn = Xmel[..., None]
        np.savez(FEATURES / "cnn" / f"{split}.npz",
                 X=Xcnn, y=y, source_path=rows["source_path"].values,
                 label_names=np.array(TARGET_LABELS))
        print(f"    svm X {Xsvm.shape} | cnn X {Xcnn.shape}")

    _write_crnn_pointer()
    print("\ndone: features/svm/*, features/cnn/*, features/crnn/ (pointer)")


def _write_crnn_pointer():
    d = FEATURES / "crnn"
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.txt").write_text(
        "CRNN reuses the CNN log-mel arrays (features/cnn/{split}.npz) unchanged.\n"
        "Same per-bin train-standardized log-mel; the ONLY difference is how the model\n"
        "consumes it: the CRNN keeps the time axis as a sequence and must NOT collapse it\n"
        "to statistics.\n\n"
        "Loader lives in the package:  instrument_robustness.crnn_data.load_crnn(split)\n"
        "It returns X shaped (N, n_frames=130, n_mels=128) = (batch, time, features).\n")


if __name__ == "__main__":
    main()
