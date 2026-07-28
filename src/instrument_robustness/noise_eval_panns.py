"""noise_eval_panns.py — run the clean-trained fine-tuned PANNs across all noise conditions.

Inference only, no retraining. Reads the SHARED noisy windows written by noise_sweep.py so the
predictions stay paired with every other model's (McNemar / bootstrap later).

Featurization goes through the SAME path the model was trained on (pretrained_extractors.panns_input:
22050 window -> 32 kHz waveform; CNN14 computes its own log-mel internally). PANNs carries its own
normalization, so no dataset stats are recomputed on noisy audio.

Outputs, under features/panns/noise/:
    panns_ft_test_{noise_type}_{snr}.csv   per-clip: window_id, true, pred, per-class probs
    metrics_{noise_type}_{snr}.json        accuracy, macro-F1, per-class P/R/F1, confusion
    noise_sweep_summary.csv                tidy condition x (accuracy, macro-F1)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
from instrument_robustness.config import ROOT, PIPE, WORK, FEATURES, SR, TARGET_LABELS
from instrument_robustness.pretrained_extractors import panns_input
from instrument_robustness.noise_sweep import SNRS, NOISE_TYPES, NOISY_DIR, window_id_of
warnings.filterwarnings("ignore")

N = len(TARGET_LABELS)
LAB2IDX = {l: i for i, l in enumerate(TARGET_LABELS)}
OUT = FEATURES / "panns" / "noise"
CLIP_LEN = int(round(3.0 * SR))


def get_device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(dev):
    from panns_inference.models import Cnn14
    bb = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
               mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    class PannsClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.head = bb, nn.Linear(2048, N)

        def forward(self, x):
            return self.head(self.backbone(x)["embedding"])

    m = PannsClassifier()
    m.load_state_dict(torch.load(FEATURES / "panns" / "panns_finetune.pt", map_location="cpu"))
    return m.eval().to(dev)


def read_wav(p):
    y, _ = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - y.size))
    return y[:CLIP_LEN]


def condition_paths(df, noise_type, snr):
    """clean -> the original windows; otherwise the shared noisy files."""
    if noise_type == "clean":
        return [ROOT / p for p in df["window_path"]]
    return [NOISY_DIR / noise_type / f"snr{snr}" / f"{window_id_of(p)}.wav" for p in df["window_path"]]


@torch.no_grad()
def run_condition(model, dev, df, noise_type, snr):
    paths = condition_paths(df, noise_type, snr)
    probs = []
    B = 32
    for i in range(0, len(paths), B):
        wavs = [panns_input(read_wav(p)) for p in paths[i:i + B]]
        x = torch.from_numpy(np.stack(wavs)).float().to(dev)
        probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(probs)


def main():
    dev = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PIPE / "windows.csv")
    df = df[df.split == "test"].reset_index(drop=True)
    y_true = np.array([LAB2IDX[l] for l in df["label"]])
    model = build_model(dev)
    print(f"device={dev} | {len(df)} test windows | model=panns_finetune (clean-trained)\n")

    conditions = [("clean", "clean")] + [(nt, snr) for nt in NOISE_TYPES for snr in SNRS]
    summary = []
    for noise_type, snr in conditions:
        probs = run_condition(model, dev, df, noise_type, snr)
        pred = probs.argmax(1)
        acc = accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
        tag = "clean" if noise_type == "clean" else f"{noise_type}_{snr}"

        out = pd.DataFrame({
            "window_id": [window_id_of(p) for p in df["window_path"]],
            "source_path": df["source_path"].values,
            "true_label": [TARGET_LABELS[i] for i in y_true],
            "predicted_label": [TARGET_LABELS[i] for i in pred],
            "correct": y_true == pred,
        })
        for j, lab in enumerate(TARGET_LABELS):
            out[f"prob_{lab}"] = probs[:, j].round(6)
        out.to_csv(OUT / f"panns_ft_test_{tag}.csv", index=False)

        rep = classification_report(y_true, pred, labels=range(N), target_names=TARGET_LABELS,
                                    output_dict=True, zero_division=0)
        with open(OUT / f"metrics_{tag}.json", "w") as f:
            json.dump({"condition": tag, "noise_type": noise_type, "snr_db": snr,
                       "n": int(len(df)), "accuracy": round(float(acc), 4),
                       "macro_f1": round(float(mf1), 4), "classification_report": rep,
                       "confusion_matrix": confusion_matrix(y_true, pred, labels=range(N)).tolist()},
                      f, indent=2)

        summary.append({"noise_type": noise_type, "snr_db": snr, "condition": tag,
                        "accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4)})
        print(f"  {tag:<12} acc {acc:.4f}   macro-F1 {mf1:.4f}")

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "noise_sweep_summary.csv", index=False)
    print("\n" + "=" * 58)
    print("NOISE SWEEP — PANNs fine-tune (clean-trained), TinySOL test")
    print("=" * 58)
    clean_f1 = s[s.condition == "clean"]["macro_f1"].iloc[0]
    print(f"{'condition':<14}{'accuracy':>10}{'macro-F1':>10}{'vs clean':>10}")
    for _, r in s.iterrows():
        d = "" if r.condition == "clean" else f"{r.macro_f1 - clean_f1:+.4f}"
        print(f"{r.condition:<14}{r.accuracy:>10.4f}{r.macro_f1:>10.4f}{d:>10}")
    print(f"\nwrote {len(conditions)} predictions CSVs + metrics JSONs + summary to {OUT}")


if __name__ == "__main__":
    main()
