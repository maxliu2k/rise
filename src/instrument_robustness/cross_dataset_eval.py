"""Cross-dataset evaluation — the definitive shortcut/confound test, BOTH directions.

Run a model trained on one dataset against the other dataset's audio, on the 9 SHARED classes,
with no retraining. If a model learned real timbre it transfers and stays high; if it learned
recording-condition / silence-structure artifacts it collapses on the foreign dataset.

For fairness every prediction is masked to the 9 shared classes (a 9-way decision), and each model
is also evaluated within its OWN test split under the same 9-way masking, so the drop is
apples-to-apples across directions.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from instrument_robustness.config import (PHILHARMONIA_ROOT, TINYSOL_ROOT,
                                          PHILHARMONIA_LABELS, TINYSOL_LABELS)
from instrument_robustness.featurelib import load_window
from instrument_robustness.pretrained_extractors import panns_input
warnings.filterwarnings("ignore")

# data roots + label orders come from config (env-overridable; no hardcoded paths)
PHIL = PHILHARMONIA_ROOT
TINY = TINYSOL_ROOT
PHIL_LABELS = PHILHARMONIA_LABELS       # 9, Philharmonia model output order
TINY_LABELS = TINYSOL_LABELS            # 12, TinySOL model output order
SHARED = PHILHARMONIA_LABELS            # 9 shared classes (the intersection)


def get_device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(model_path, n_classes, dev):
    from panns_inference.models import Cnn14
    bb = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
               mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    class PannsClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.head = nn.Linear(2048, n_classes)

        def forward(self, x):
            return self.head(self.backbone(x)["embedding"])

    m = PannsClassifier()
    m.load_state_dict(torch.load(model_path, map_location="cpu"))
    return m.eval().to(dev)


def evaluate(model, model_labels, data_root, windows_csv, dev, split=None):
    """Predict on windows (optionally one split), masking logits to SHARED, 9-way. Returns y_true,y_pred."""
    shared_cols = [model_labels.index(l) for l in SHARED]     # model-output indices of shared classes
    w = pd.read_csv(windows_csv)
    w = w[w.label.isin(SHARED)]
    if split:
        w = w[w.split == split]
    w = w.reset_index(drop=True)
    y_true = np.array([SHARED.index(l) for l in w["label"]])
    paths = w["window_path"].tolist()
    preds, B = [], 32
    with torch.no_grad():
        for i in range(0, len(paths), B):
            wavs = [panns_input(load_window(data_root / p)) for p in paths[i:i + B]]
            x = torch.from_numpy(np.stack(wavs)).float().to(dev)
            logits = model(x)[:, shared_cols]                 # keep only shared classes
            preds.append(logits.argmax(1).cpu().numpy())      # index into SHARED
    return y_true, np.concatenate(preds)


def score(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def main():
    dev = get_device()
    print(f"device={dev} | 9 shared classes, all predictions masked to a 9-way decision\n")

    phil_model = build_model(PHIL / "features/panns/panns_finetune.pt", 9, dev)
    tiny_model = build_model(TINY / "features/panns/panns_finetune.pt", 12, dev)

    results = {}
    # --- Philharmonia model ---
    print("Philharmonia model: within-test ...")
    results["phil_within"] = score(*evaluate(phil_model, PHIL_LABELS, PHIL,
                                             PHIL / "pipeline/windows.csv", dev, split="test"))
    print("Philharmonia model: -> TinySOL audio ...")
    yt, yp = evaluate(phil_model, PHIL_LABELS, TINY, TINY / "pipeline/windows.csv", dev)
    results["phil_cross"] = score(yt, yp)
    cm_phil = confusion_matrix(yt, yp, labels=range(9))

    # --- TinySOL model ---
    print("TinySOL model: within-test ...")
    results["tiny_within"] = score(*evaluate(tiny_model, TINY_LABELS, TINY,
                                            TINY / "pipeline/windows.csv", dev, split="test"))
    print("TinySOL model: -> Philharmonia audio ...")
    yt2, yp2 = evaluate(tiny_model, TINY_LABELS, PHIL, PHIL / "pipeline/windows.csv", dev)
    results["tiny_cross"] = score(yt2, yp2)
    cm_tiny = confusion_matrix(yt2, yp2, labels=range(9))

    def line(name, acc, f1):
        return f"  {name:<34} acc {acc:.4f}   macro-F1 {f1:.4f}"

    print("\n" + "=" * 66)
    print("RESULTS (9-way masked, macro-F1)")
    print("=" * 66)
    print(line("Philharmonia model, within-test", *results["phil_within"]))
    print(line("Philharmonia model -> TinySOL", *results["phil_cross"]))
    print(f"     drop = {results['phil_cross'][1]-results['phil_within'][1]:+.4f}")
    print()
    print(line("TinySOL model, within-test", *results["tiny_within"]))
    print(line("TinySOL model -> Philharmonia", *results["tiny_cross"]))
    print(f"     drop = {results['tiny_cross'][1]-results['tiny_within'][1]:+.4f}")
    print("=" * 66)
    print(f"CROSS-DATASET macro-F1:  Phil->Tiny {results['phil_cross'][1]:.4f}   "
          f"Tiny->Phil {results['tiny_cross'][1]:.4f}   (chance 0.111)")
    print("=" * 66)
    for tag, cm in (("Phil->TinySOL", cm_phil), ("TinySOL->Phil", cm_tiny)):
        print(f"\nconfusion {tag} (rows=true, cols=pred):")
        print("            " + " ".join(f"{l[:4]:>5}" for l in SHARED))
        for i, row in enumerate(cm):
            print(f"{SHARED[i]:>10}  " + " ".join(f"{v:5d}" for v in row))


if __name__ == "__main__":
    main()
