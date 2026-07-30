"""Cross-dataset evaluation — the definitive shortcut/confound test, BOTH directions.

Run a model trained on one dataset against the other dataset's audio, on the SHARED classes, with
no retraining. If a model learned real timbre it transfers and stays high; if it learned
recording-condition / silence-structure artifacts it collapses on the foreign dataset.

For fairness every prediction is masked to the shared classes, and each model is also evaluated
within its OWN test split under the same masking, so the drop is apples-to-apples across
directions. Under the canonical 12-class set both datasets carry all 12, so the mask is currently
an identity and the decision is 12-way; the masking is kept because it is what makes the
comparison valid whenever the two label sets are NOT identical.

    RISE_TINYSOL_ROOT=/path/to/TinySOL2020 python -m instrument_robustness.cross_dataset_eval \\
        --phil-model /path/to/panns_finetune_philharmonia.pt \\
        --tiny-model /path/to/panns_finetune_tinysol.pt

Note that TINYSOL_ROOT must point at the TinySOL DATA root (the one holding work/windows/ and
pipeline/windows.csv), not at the repository's small committed `tinysol/` artifact mirror.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from instrument_robustness.config import (PHILHARMONIA_ROOT, TINYSOL_ROOT,
                                          PHILHARMONIA_LABELS, TINYSOL_LABELS)
from instrument_robustness.featurelib import load_window
from instrument_robustness.pretrained_extractors import panns_input
warnings.filterwarnings("ignore")

# data roots + label orders come from config (env-overridable; no hardcoded paths). The per-model
# output order is read from each checkpoint in build_model; these are only the fallback for a bare
# state_dict that records none.
PHIL = PHILHARMONIA_ROOT
TINY = TINYSOL_ROOT
SHARED = PHILHARMONIA_LABELS            # the classes scored in both directions


def get_device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def build_model(model_path, dev, fallback_labels):
    """Load a PANNs classifier, sizing the head from the CHECKPOINT rather than from a constant.

    The head width used to be a caller-supplied number (9 for Philharmonia, 12 for TinySOL), which
    silently hard-coded the retired 9-class era. Against a 12-class checkpoint that mismatch
    surfaces as a load_state_dict shape error rather than as a wrong number, so it is read from the
    saved head instead: a 9-class and a 12-class checkpoint both load without the caller having to
    know which it is.

    Returns (model, label_order) so the caller masks against the order the model was ACTUALLY
    trained on, taken from the checkpoint whenever it records one.
    """
    from panns_inference.models import Cnn14
    bb = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
               mel_bins=64, fmin=50, fmax=14000, classes_num=527)

    ckpt = torch.load(model_path, map_location="cpu")
    # train_panns saves {state_dict, label_order, config_fingerprint}; older runs saved a bare
    # state_dict. Accept both.
    is_bundle = isinstance(ckpt, dict) and "state_dict" in ckpt
    state = ckpt["state_dict"] if is_bundle else ckpt
    n_classes = state["head.weight"].shape[0]
    labels = list(ckpt["label_order"]) if is_bundle and "label_order" in ckpt else list(fallback_labels)
    if len(labels) != n_classes:
        raise SystemExit(f"{model_path}: head has {n_classes} classes but label order has "
                         f"{len(labels)} entries -- refusing to guess the mapping")

    class PannsClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.head = nn.Linear(2048, n_classes)

        def forward(self, x):
            return self.head(self.backbone(x)["embedding"])

    m = PannsClassifier()
    m.load_state_dict(state)
    return m.eval().to(dev), labels


def evaluate(model, model_labels, data_root, windows_csv, dev, split=None):
    """Predict on windows (optionally one split), masking logits to SHARED. Returns y_true,y_pred."""
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Explicit paths because the in-place defaults under each data root are whatever the last local
    # run left there, which is not necessarily the checkpoint you mean to report.
    p.add_argument("--phil-model", type=Path, default=PHIL / "features/panns/panns_finetune.pt")
    p.add_argument("--tiny-model", type=Path, default=TINY / "features/panns/panns_finetune.pt")
    return p.parse_args()


def main():
    args = parse_args()
    dev = get_device()
    n_shared = len(SHARED)
    print(f"device={dev} | {n_shared} shared classes, all predictions masked to a "
          f"{n_shared}-way decision\n")
    print(f"phil model: {args.phil_model}")
    print(f"tiny model: {args.tiny_model}\n")

    phil_model, PHIL_LABELS = build_model(args.phil_model, dev, PHILHARMONIA_LABELS)
    tiny_model, TINY_LABELS = build_model(args.tiny_model, dev, TINYSOL_LABELS)

    results = {}
    # --- Philharmonia model ---
    print("Philharmonia model: within-test ...")
    results["phil_within"] = score(*evaluate(phil_model, PHIL_LABELS, PHIL,
                                             PHIL / "pipeline/windows.csv", dev, split="test"))
    print("Philharmonia model: -> TinySOL audio ...")
    yt, yp = evaluate(phil_model, PHIL_LABELS, TINY, TINY / "pipeline/windows.csv", dev)
    results["phil_cross"] = score(yt, yp)
    cm_phil = confusion_matrix(yt, yp, labels=range(n_shared))

    # --- TinySOL model ---
    print("TinySOL model: within-test ...")
    results["tiny_within"] = score(*evaluate(tiny_model, TINY_LABELS, TINY,
                                            TINY / "pipeline/windows.csv", dev, split="test"))
    print("TinySOL model: -> Philharmonia audio ...")
    yt2, yp2 = evaluate(tiny_model, TINY_LABELS, PHIL, PHIL / "pipeline/windows.csv", dev)
    results["tiny_cross"] = score(yt2, yp2)
    cm_tiny = confusion_matrix(yt2, yp2, labels=range(n_shared))

    def line(name, acc, f1):
        return f"  {name:<34} acc {acc:.4f}   macro-F1 {f1:.4f}"

    print("\n" + "=" * 66)
    print(f"RESULTS ({n_shared}-way masked, macro-F1)")
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
          f"Tiny->Phil {results['tiny_cross'][1]:.4f}   (chance {1/n_shared:.3f})")
    print("=" * 66)
    for tag, cm in (("Phil->TinySOL", cm_phil), ("TinySOL->Phil", cm_tiny)):
        print(f"\nconfusion {tag} (rows=true, cols=pred):")
        print("            " + " ".join(f"{l[:4]:>5}" for l in SHARED))
        for i, row in enumerate(cm):
            print(f"{SHARED[i]:>10}  " + " ".join(f"{v:5d}" for v in row))


if __name__ == "__main__":
    main()
