"""Train the medium CNN over config.SEEDS and report mean +/- std on the held-out test split.

The split is built once by prep_data and held FIXED across seeds, so the reported spread is
model-init and batch-order variance, not split variance.

Also owns the model definition and the split-loading helpers that noise_eval.py imports.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef)

from .config import (
    BATCH_SIZE, CLASSES, DROPOUT, EARLY_STOP_PATIENCE, LEARNING_RATE, MANIFEST_JSON,
    MAX_EPOCHS, MAX_IMBALANCE, OUTPUTS, PLATEAU_FACTOR, PLATEAU_PATIENCE,
    SEED, SEEDS, SPEC_DIR, SPLITS_JSON,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data

def load_manifest():
    if not MANIFEST_JSON.exists() or not SPLITS_JSON.exists():
        sys.exit("ERROR: cache missing — run `python prep_data.py` first.")
    manifest = json.loads(MANIFEST_JSON.read_text())
    splits = json.loads(SPLITS_JSON.read_text())
    return manifest, splits, {r["id"]: r for r in manifest["records"]}


def load_split(split_ids, by_id):
    """Cached spectrograms for a split -> (specs, y, ids).

    Clips are variable length, so specs is a LIST of (1, n_mels, frames) tensors rather than
    one stacked array. The whole set is ~50 MB, so it lives in memory and epochs never touch
    disk.
    """
    ids = sorted(split_ids)
    specs = [torch.from_numpy(np.load(SPEC_DIR / f"{i}.npy")).float().unsqueeze(0)
             for i in ids]
    y = torch.from_numpy(np.array([by_id[i]["label"] for i in ids], dtype=np.int64))
    return specs, y, ids


class LengthBatcher:
    """Yields batches of clips that all have the SAME frame count.

    Variable-length clips cannot be stacked into one tensor, and the usual workaround —
    padding to the batch maximum — would reintroduce exactly the digital silence that
    breaks the noise sweep, and would contaminate BatchNorm statistics. Grouping by exact
    length sidesteps both: every batch is uniform by construction, with zero padding.

    Cost: batches are smaller than BATCH_SIZE when a length has few clips (here ~14 mean vs
    a 32 target). BatchNorm2d tolerates this because it pools over height and width as well
    as batch, so even a single clip yields n_mels x frames values per channel.
    """

    def __init__(self, specs, labels, batch_size, shuffle=False, seed=None):
        self.specs, self.labels, self.batch_size, self.shuffle = specs, labels, batch_size, shuffle
        self.rng = random.Random(seed)
        self.by_len = defaultdict(list)
        for i, s in enumerate(specs):
            self.by_len[s.shape[-1]].append(i)

    def __len__(self):
        return sum(int(np.ceil(len(v) / self.batch_size)) for v in self.by_len.values())

    def __iter__(self):
        batches = []
        for idxs in self.by_len.values():
            idxs = list(idxs)
            if self.shuffle:
                self.rng.shuffle(idxs)
            batches += [idxs[i:i + self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)  # else every epoch walks lengths in the same order
        for b in batches:
            yield torch.stack([self.specs[i] for i in b]), self.labels[list(b)]


def class_weights(y_train, quiet=False):
    """N / (n_classes * n_c) — upweights minority classes. Applied only if the training
    split is more imbalanced than MAX_IMBALANCE."""
    counts = np.bincount(y_train.numpy(), minlength=len(CLASSES))
    ratio = counts.max() / max(counts.min(), 1)
    if ratio <= MAX_IMBALANCE:
        if not quiet:
            print(f"class ratio {ratio:.2f}:1 <= {MAX_IMBALANCE} — no class weights")
        return None, ratio
    w = len(y_train) / (len(CLASSES) * counts)
    if not quiet:
        lo, hi = CLASSES[int(counts.argmin())], CLASSES[int(counts.argmax())]
        print(f"class ratio {ratio:.2f}:1 > {MAX_IMBALANCE} (min {lo} {counts.min()}, "
              f"max {hi} {counts.max()}) — applying class weights "
              f"[{w.min():.3f}..{w.max():.3f}]")
    return torch.tensor(w, dtype=torch.float32), ratio


# --------------------------------------------------------------------------- model

class MediumCNN(nn.Module):
    """3 conv blocks (32/64/128) -> GAP -> Dense 128 -> Dropout -> Dense n_classes."""

    def __init__(self, n_classes=len(CLASSES), dropout=DROPOUT):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.gap(self.features(x)))


# --------------------------------------------------------------------------- loops

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, balanced_accuracy, preds, targets).

    Balanced accuracy — the mean of per-class recall — not raw accuracy. On an imbalanced
    set, a model that collapses onto the largest class is paid its prior and posts a number
    that reads like a result; raw accuracy's floor also drifts with the split, so it is not
    comparable across configurations. Balanced accuracy scores a collapsed model
    1/n_classes, and that floor is fixed whatever the imbalance.
    """
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total_loss += criterion(out, yb).item() * xb.size(0)
        preds.append(out.argmax(1).cpu())
        targets.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(targets), balanced_accuracy_score(targets, preds), preds, targets


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n, preds, targets = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        preds.append(out.argmax(1).detach().cpu())
        targets.append(yb.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total_loss / n, balanced_accuracy_score(targets, preds)


# --------------------------------------------------------------------------- plots

def plot_curves(histories, path):
    """One panel per metric, one line per seed — so seed spread is visible rather than
    averaged away."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    chance = 1.0 / len(CLASSES)
    for seed, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        ax1.plot(ep, h["train_loss"], lw=1.2, alpha=0.8, label=f"train s{seed}")
        ax1.plot(ep, h["val_loss"], lw=1.2, ls="--", alpha=0.8, label=f"val s{seed}")
        ax2.plot(ep, h["train_bacc"], lw=1.2, alpha=0.8, label=f"train s{seed}")
        ax2.plot(ep, h["val_bacc"], lw=1.2, ls="--", alpha=0.8, label=f"val s{seed}")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax2.set(xlabel="epoch", ylabel="balanced accuracy", title="Balanced accuracy")
    ax2.axhline(chance, ls=":", color="#d62728", lw=1.2,
                label=f"chance ({chance:.3f})")
    for ax in (ax1, ax2):
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{len(CLASSES)}-class instrument ID — medium CNN, {len(histories)} seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title=None):
    """Row-normalised: with an imbalanced set, raw counts make the big classes look good and
    hide a small class being systematically swallowed. Rows are true classes, so each row
    sums to 1 and the diagonal is per-class recall."""
    n = len(CLASSES)
    title = title or f"Confusion matrix (test, {n} classes) — row-normalised = recall"
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)
    size = max(5.0, 0.62 * n + 2.2)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(n), yticks=range(n), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="predicted", ylabel="true", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fs = 7 if n > 6 else 9
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{norm[i, j]:.2f}".lstrip("0") if norm[i, j] < 1 else "1.0",
                    ha="center", va="center", fontsize=fs,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(ids, preds, targets, path, limit=8):
    wrong = np.where(preds != targets)[0]
    if wrong.size == 0:
        print("no misclassified test clips — skipping misclassified plot")
        return 0
    sel = wrong[:limit]
    n = len(sel)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, sel):
        spec = np.load(SPEC_DIR / f"{ids[idx]}.npy")
        ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"true {CLASSES[targets[idx]]} / pred {CLASSES[preds[idx]]}\n{ids[idx]}",
                     fontsize=6)
        ax.axis("off")
    fig.suptitle(f"Misclassified test clips ({wrong.size} total, showing {n})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return int(wrong.size)


# --------------------------------------------------------------------------- one seed

def run_seed(seed, data, device):
    """Train and evaluate one seed.

    The split is fixed across seeds (built once by prep_data at config.SEED), so only model
    init and batch order vary — which is exactly the variance we want to measure. Varying
    the split too would conflate model variance with split variance.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    set_seed(seed)

    weights, ratio = class_weights(ytr, quiet=True)
    train_loader = LengthBatcher(Xtr, ytr, BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = LengthBatcher(Xva, yva, BATCH_SIZE)
    test_loader = LengthBatcher(Xte, yte, BATCH_SIZE)

    model = MediumCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)

    history = {k: [] for k in ("train_loss", "train_bacc", "val_loss", "val_bacc", "lr")}
    epoch_times = []
    best_val, best_state, best_epoch, since_improved = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_bacc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_bacc, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_times.append(time.perf_counter() - t0)

        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        for k, v in zip(history, (tr_loss, tr_bacc, va_loss, va_bacc, lr)):
            history[k].append(v)

        flag = ""
        if va_loss < best_val:
            best_val, best_epoch, since_improved = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            since_improved += 1

        print(f"  s{seed} | ep {epoch:>2}/{MAX_EPOCHS} | {epoch_times[-1]:5.1f}s | "
              f"train {tr_loss:.4f}/{tr_bacc:.4f} | val {va_loss:.4f}/{va_bacc:.4f} | "
              f"lr {lr:.1e}{flag}")

        if since_improved >= EARLY_STOP_PATIENCE:
            print(f"  s{seed} | early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    path = OUTPUTS / f"model_s{seed}.pt"
    torch.save({"state_dict": best_state, "seed": seed, "classes": list(CLASSES)}, path)

    te_loss, te_bacc, preds, targets = evaluate(model, test_loader, criterion, device)
    te_mcc = matthews_corrcoef(targets, preds)
    rep = classification_report(targets, preds, labels=list(range(len(CLASSES))),
                                target_names=list(CLASSES), output_dict=True, zero_division=0)
    per_class = {c: {"precision": float(rep[c]["precision"]),
                     "recall": float(rep[c]["recall"]),
                     "support": int(rep[c]["support"])} for c in CLASSES}
    times = np.array(epoch_times)
    gap = history["train_bacc"][best_epoch - 1] - history["val_bacc"][best_epoch - 1]

    print(f"  s{seed} | best ep {best_epoch} | TEST balanced acc {te_bacc:.4f} | "
          f"MCC {te_mcc:.4f} | {times.mean():.1f}s/epoch\n")

    return {
        "seed": seed, "best_epoch": best_epoch, "best_val_loss": float(best_val),
        "test_loss": float(te_loss), "test_balanced_accuracy": float(te_bacc),
        "test_mcc": float(te_mcc), "train_val_bacc_gap": float(gap),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            targets, preds, labels=list(range(len(CLASSES)))).tolist(),
        "epochs_run": len(times), "mean_epoch_s": float(times.mean()),
        "total_s": float(times.sum()), "class_ratio_train": float(ratio),
        "class_weights": None if weights is None else weights.tolist(),
        "model_path": str(path),
    }, history, preds, targets


# --------------------------------------------------------------------------- main

def agg(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                    help="seeds to run (default: config.SEEDS)")
    args = ap.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device} | torch {torch.__version__} | seeds {args.seeds}")
    print(f"{len(CLASSES)} classes: {', '.join(CLASSES)}\n")

    manifest, splits, by_id = load_manifest()
    Xtr, ytr, _ = load_split(splits["train"], by_id)
    Xva, yva, _ = load_split(splits["val"], by_id)
    Xte, yte, test_ids = load_split(splits["test"], by_id)
    frames = [s.shape[-1] for s in Xtr]
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} clips")
    print(f"variable length: {min(frames)}-{max(frames)} frames, {len(set(frames))} distinct")
    class_weights(ytr)
    nb = len(LengthBatcher(Xtr, ytr, BATCH_SIZE))
    print(f"length-bucketed batching: {nb} train batches/epoch, mean size {len(Xtr) / nb:.1f}")
    print(f"model params: {sum(p.numel() for p in MediumCNN().parameters()):,}\n")

    data = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    results, histories, first = [], {}, None
    for seed in args.seeds:
        r, h, preds, targets = run_seed(seed, data, device)
        results.append(r)
        histories[seed] = h
        if first is None:
            first = (preds, targets)

    chance = 1.0 / len(CLASSES)
    bacc = agg([r["test_balanced_accuracy"] for r in results])
    mcc = agg([r["test_mcc"] for r in results])

    print("=" * 76)
    print(f"AGGREGATE OVER {len(results)} SEEDS")
    print("=" * 76)
    print(f"balanced accuracy: chance = 1/{len(CLASSES)} = {chance:.4f} | MCC: 0.0 = no information\n")
    print(f"{'metric':<22}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    rows = (("balanced accuracy", bacc), ("MCC", mcc),
            ("train-val bacc gap", agg([r["train_val_bacc_gap"] for r in results])),
            ("best epoch", agg([r["best_epoch"] for r in results])),
            ("s/epoch", agg([r["mean_epoch_s"] for r in results])))
    for name, a in rows:
        print(f"{name:<22}{a['mean']:>9.4f}{a['std']:>9.4f}{a['min']:>9.4f}{a['max']:>9.4f}")
    print("\nper-seed balanced accuracy: "
          + ", ".join(f"s{r['seed']}={r['test_balanced_accuracy']:.4f}" for r in results))

    print("\nper-class across seeds (recall mean +/- std) — names which classes are hard:")
    print(f"{'class':<14}{'recall':>9}{'std':>8}{'precision':>11}{'std':>8}{'support':>9}")
    per_class_agg = {}
    for c in CLASSES:
        ra = agg([r["per_class"][c]["recall"] for r in results])
        pa = agg([r["per_class"][c]["precision"] for r in results])
        sup = results[0]["per_class"][c]["support"]
        per_class_agg[c] = {"recall": ra, "precision": pa, "support": sup}
        print(f"{c:<14}{ra['mean']:>9.4f}{ra['std']:>8.4f}{pa['mean']:>11.4f}{pa['std']:>8.4f}{sup:>9}")
    order = sorted(CLASSES, key=lambda c: per_class_agg[c]["recall"]["mean"])
    print(f"\nhardest: {order[0]} ({per_class_agg[order[0]]['recall']['mean']:.4f}) | "
          f"easiest: {order[-1]} ({per_class_agg[order[-1]]['recall']['mean']:.4f})")

    # summed over seeds: a single seed's off-diagonal is too sparse to read at 11 classes
    cm_sum = np.sum([np.array(r["confusion_matrix"]) for r in results], axis=0)
    top = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm_sum[i, j] > 0:
                top.append((cm_sum[i, j], CLASSES[i], CLASSES[j]))
    top.sort(reverse=True)
    if top:
        print("\nmost common confusions (summed over seeds):")
        for n, t, p in top[:6]:
            print(f"    {t:>14} -> {p:<14} {n:>4}")

    plot_confusion(cm_sum, OUTPUTS / "confusion_matrix.png",
                   title=f"Confusion matrix — {len(CLASSES)} classes, {len(results)} seeds "
                         f"(row-normalised = recall)")
    plot_curves(histories, OUTPUTS / "curves.png")
    preds, targets = first
    n_wrong = plot_misclassified(test_ids, preds, targets, OUTPUTS / "misclassified.png")

    timing = {"device": str(device), "n_seeds": len(results),
              "mean_epoch_s": agg([r["mean_epoch_s"] for r in results]),
              "total_s_per_seed": agg([r["total_s"] for r in results]),
              "wall_clock_all_seeds_s": float(sum(r["total_s"] for r in results)),
              "n_train": int(len(ytr)), "n_classes": len(CLASSES)}
    print(f"\ntiming: {timing['mean_epoch_s']['mean']:.2f}s/epoch, "
          f"{timing['total_s_per_seed']['mean']:.0f}s/seed, "
          f"{timing['wall_clock_all_seeds_s']:.0f}s total ({device})")

    (OUTPUTS / "timing.json").write_text(json.dumps(timing, indent=2))
    (OUTPUTS / "metrics.json").write_text(json.dumps({
        "classes": list(CLASSES),
        "n_classes": len(CLASSES),
        "chance_balanced_accuracy": chance,
        "seeds": args.seeds,
        "device": str(device),
        "articulation_mode": manifest["articulation_mode"],
        "sample_rate": manifest.get("sample_rate"),
        "bitrate_kbps_by_class": manifest.get("bitrate_kbps_by_class"),
        "n_params": int(sum(p.numel() for p in MediumCNN().parameters())),
        "learning_rate": LEARNING_RATE,
        # raw accuracy and F1 are deliberately not recorded: both pay a collapsed classifier
        # the class prior, and both have floors that drift with the split.
        "test_balanced_accuracy": bacc,
        "test_mcc": mcc,
        "per_class": per_class_agg,
        "confusion_matrix_summed": cm_sum.tolist(),
        "hardest_class": order[0],
        "easiest_class": order[-1],
        "n_misclassified_first_seed": n_wrong,
        "per_seed": results,
        "timing": timing,
    }, indent=2))
    print(f"wrote {OUTPUTS / 'metrics.json'}, {OUTPUTS / 'timing.json'}, and plots")


if __name__ == "__main__":
    main()
