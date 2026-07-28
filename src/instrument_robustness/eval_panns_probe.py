"""eval_panns_probe.py — evaluation-only metrics for the ALREADY-TRAINED PANNs linear probe.

No retraining, no fine-tune, single seed. Reconstructs the exact probe predictions by loading the
saved head (panns_probe.pt) and the cached test/val embeddings (emb_{test,val}.npz), so nothing is
re-fit and the test split is used ONLY for this final evaluation.

Outputs (under features/panns/):
  panns_probe_test_predictions.csv   one row/test window: window_path, true, pred, prob_<class> x9
  panns_probe_confusion_counts.csv   9x9 raw counts (rows=true, cols=pred, TARGET_LABELS order)
  panns_probe_confusion_rownorm.csv  row-normalized version
  panns_probe_confusion.png          heatmap (labeled axes)
  panns_probe_eval.json              classification_report + accuracy/macro-F1 (test & val)
"""
import json
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

from instrument_robustness.config import PIPE, FEATURES, TARGET_LABELS

OUT = FEATURES / "panns"
LABEL2IDX = {l: i for i, l in enumerate(TARGET_LABELS)}
N = len(TARGET_LABELS)


def load_head():
    sd = torch.load(OUT / "panns_probe.pt", map_location="cpu")
    head = nn.Linear(2048, N)
    head.load_state_dict(sd)
    head.eval()
    return head


def split_frame(split):
    """windows.csv rows for this split, in the SAME order the embeddings were extracted
    (get_embeddings used a no-shuffle DataLoader over load_split(), which preserves df order)."""
    df = pd.read_csv(PIPE / "windows.csv")
    return df[df.split == split].reset_index(drop=True)


def predict(head, split):
    d = np.load(OUT / f"emb_{split}.npz")
    E, y = d["E"], d["y"].astype(int)
    df = split_frame(split)
    # verify the cached-embedding order matches windows.csv order (so window_path attaches correctly)
    y_ref = np.array([LABEL2IDX[l] for l in df["label"]], dtype=int)
    assert len(y) == len(y_ref) and np.array_equal(y, y_ref), \
        f"{split}: cached embedding order != windows.csv order"
    with torch.no_grad():
        logits = head(torch.from_numpy(E).float())
        probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(1)
    return df, y, pred, probs


def save_confusion(y, pred):
    cm = confusion_matrix(y, pred, labels=range(N))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(cm, index=TARGET_LABELS, columns=TARGET_LABELS).to_csv(
        OUT / "panns_probe_confusion_counts.csv")
    pd.DataFrame(cm_norm.round(4), index=TARGET_LABELS, columns=TARGET_LABELS).to_csv(
        OUT / "panns_probe_confusion_rownorm.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(TARGET_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(TARGET_LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("PANNs linear probe — TEST confusion (row-normalized)")
    for i in range(N):
        for j in range(N):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "panns_probe_confusion.png", dpi=150)
    plt.close(fig)
    return cm


def main():
    head = load_head()
    df_te, y_te, pred_te, prob_te = predict(head, "test")
    df_va, y_va, pred_va, _ = predict(head, "val")

    # ---- per-clip test predictions CSV (required) ----
    out = pd.DataFrame({
        "window_path": df_te["window_path"].values,
        "source_path": df_te["source_path"].values,
        "true_label": [TARGET_LABELS[i] for i in y_te],
        "predicted_label": [TARGET_LABELS[i] for i in pred_te],
        "correct": (y_te == pred_te),
    })
    for j, lab in enumerate(TARGET_LABELS):
        out[f"prob_{lab}"] = prob_te[:, j].round(6)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "panns_probe_test_predictions.csv", index=False)

    # ---- confusion matrices + heatmap ----
    save_confusion(y_te, pred_te)

    # ---- metrics ----
    rep = classification_report(y_te, pred_te, labels=range(N),
                                target_names=TARGET_LABELS, digits=4, zero_division=0)
    rep_dict = classification_report(y_te, pred_te, labels=range(N),
                                     target_names=TARGET_LABELS, output_dict=True, zero_division=0)
    test_acc = accuracy_score(y_te, pred_te)
    test_f1 = f1_score(y_te, pred_te, average="macro", zero_division=0)
    val_acc = accuracy_score(y_va, pred_va)
    val_f1 = f1_score(y_va, pred_va, average="macro", zero_division=0)

    with open(OUT / "panns_probe_eval.json", "w") as f:
        json.dump({"split": "test", "n_test": int(len(y_te)), "n_val": int(len(y_va)),
                   "test_accuracy": round(float(test_acc), 4),
                   "test_macro_f1": round(float(test_f1), 4),
                   "val_accuracy": round(float(val_acc), 4),
                   "val_macro_f1": round(float(val_f1), 4),
                   "classification_report": rep_dict}, f, indent=2)

    # ---- console summary ----
    print("=" * 64)
    print("PANNs LINEAR PROBE — evaluation (frozen trunk, single seed, TEST split)")
    print("=" * 64)
    print(f"TEST : accuracy {test_acc:.4f}   macro-F1 {test_f1:.4f}   (n={len(y_te)})")
    print(f"VAL  : accuracy {val_acc:.4f}   macro-F1 {val_f1:.4f}   (n={len(y_va)})  "
          f"[overfit check: Δmacro-F1 = {test_f1 - val_f1:+.4f}]")
    print("\nclassification_report (TEST):\n")
    print(rep)
    print("files written under features/panns/:")
    for fn in ("panns_probe_test_predictions.csv", "panns_probe_confusion_counts.csv",
               "panns_probe_confusion_rownorm.csv", "panns_probe_confusion.png",
               "panns_probe_eval.json"):
        print("  -", fn)


if __name__ == "__main__":
    main()
