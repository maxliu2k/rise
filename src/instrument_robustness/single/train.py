"""Single-instrument (multi-class) CNN: one instrument per clip, softmax over CLASSES.

Trains over config.SEEDS and reports mean +/- std on the held-out test split. The split is
built once by prep_data and held FIXED across seeds, so the reported spread is model-init and
batch-order variance, not split variance.

Run:  python -m instrument_robustness.single.train [--seeds ...] [--progress]
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

from ..config import (
    BATCH_SIZE, CLASSES, EARLY_STOP_PATIENCE, LEARNING_RATE, MAX_EPOCHS, OUTPUTS,
    PLATEAU_FACTOR, PLATEAU_PATIENCE, SEED_METRICS_DIR, SEEDS, SPEC_DIR, SPECAUGMENT,
    StaleArtifactError, WEIGHT_DECAY, assert_fingerprint, config_fingerprint,
)
from ..cnn_core import (
    agg, class_weights, evaluate, get_device, LengthBatcher, load_manifest, load_split,
    MediumCNN, MediumCRNN, set_seed, spec_augment, train_one_epoch,
)

# Selectable architectures. "cnn" is the default and keeps the canonical output paths; anything
# else writes into outputs/<name>/ so two architectures cannot overwrite each other's results.
ARCHITECTURES = {"cnn": MediumCNN, "crnn": MediumCRNN}


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

def run_seed(seed, data, device, on_epoch=None, ckpt_path=None, model_cls=None):
    """Train and evaluate one seed. on_epoch(epoch) is called after each epoch (progress UI).

    The split is fixed across seeds (built once by prep_data at config.SEED), so only model
    init and batch order vary — which is exactly the variance we want to measure. Varying
    the split too would conflate model variance with split variance.

    ckpt_path overrides where the checkpoint is written. It exists so an experiment that trains
    on a DIFFERENT split (split_policy_probe) cannot overwrite the canonical outputs/model_s*.pt
    — those are the checkpoints noise_eval loads, and silently replacing them with models trained
    on a deliberately-leaked split is exactly the kind of swap nothing downstream would notice.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    set_seed(seed)

    weights, ratio = class_weights(ytr, quiet=True)
    train_loader = LengthBatcher(Xtr, ytr, BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = LengthBatcher(Xva, yva, BATCH_SIZE)
    test_loader = LengthBatcher(Xte, yte, BATCH_SIZE)

    model = (model_cls or MediumCNN)().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    # AdamW with WEIGHT_DECAY=0 (the default) is identical to Adam, so the baseline is
    # unchanged; a non-zero config value turns on decoupled weight decay.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)
    augment = spec_augment if SPECAUGMENT else None

    history = {k: [] for k in ("train_loss", "train_bacc", "val_loss", "val_bacc", "lr")}
    epoch_times = []
    best_val, best_state, best_epoch, since_improved = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_bacc = train_one_epoch(model, train_loader, criterion, optimizer, device, augment)
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
        if on_epoch is not None:
            on_epoch(epoch)

        if since_improved >= EARLY_STOP_PATIENCE:
            print(f"  s{seed} | early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    path = ckpt_path if ckpt_path is not None else OUTPUTS / f"model_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # fingerprint travels with the weights: evaluating these against a cache built under a
    # different config yields a plausible, meaningless number that nothing else would catch.
    torch.save({"state_dict": best_state, "seed": seed, "classes": list(CLASSES),
                "arch": (model_cls or MediumCNN).__name__,
                "fingerprint": config_fingerprint()}, path)

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


# --------------------------------------------------------------------------- seed store


def save_seed_metrics(seed, result, history, preds, targets, store=None):
    """Persist one seed's complete result to outputs/seed_metrics/s{seed}.json.

    Precondition: `result`, `history`, `preds`, `targets` are run_seed()'s outputs for `seed`.
    Postcondition: the file holds everything needed to rebuild every aggregate, table and plot
    for this seed WITHOUT retraining it, plus the config fingerprint it was produced under.
    """
    store = store if store is not None else SEED_METRICS_DIR
    store.mkdir(parents=True, exist_ok=True)
    (store / f"s{seed}.json").write_text(json.dumps({
        "fingerprint": config_fingerprint(),
        "result": result,
        "history": history,
        "preds": [int(p) for p in preds],
        "targets": [int(t) for t in targets],
    }, indent=2))


def load_seed_metrics(seeds, store=None):
    """Load persisted results for every seed in `seeds`, whenever they were trained.

    This is why metrics.json cannot silently cover a subset: aggregation reads the canonical
    seed set from disk rather than whatever one invocation happened to run. Training seed 42,
    then seeds 43 and 44 separately, previously left metrics.json describing 2 of 3 while
    still looking canonical.

    Precondition: each seed has been trained under the CURRENT config.
    Postcondition: returns (results, histories, (preds, targets) of the first seed).
    Raises StaleArtifactError naming the missing seeds and the exact command to produce them,
    or if any seed was trained under a different config.
    """
    store = store if store is not None else SEED_METRICS_DIR
    missing = [s for s in seeds if not (store / f"s{s}.json").exists()]
    if missing:
        have = [s for s in seeds if s not in missing]
        raise StaleArtifactError(
            f"no saved results for seed(s) {missing} — an aggregate over only {have} would be "
            f"written to metrics.json and read as the full result.\n"
            f"  Run: python -m instrument_robustness.single.train --seeds "
            f"{' '.join(str(s) for s in missing)}")
    results, histories, first = [], {}, None
    for s in seeds:
        d = json.loads((store / f"s{s}.json").read_text())
        assert_fingerprint(d.get("fingerprint"), str(store / f"s{s}.json"))
        results.append(d["result"])
        histories[s] = d["history"]
        if first is None:
            first = (np.array(d["preds"]), np.array(d["targets"]))
    return results, histories, first


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                    help="seeds to run (default: config.SEEDS)")
    ap.add_argument("--progress", action="store_true",
                    help="show a pop-up progress bar (auto-skips if no display)")
    ap.add_argument("--model", choices=sorted(ARCHITECTURES), default="cnn",
                    help="architecture to train (default: cnn)")
    args = ap.parse_args()

    # The CNN keeps the canonical top-level paths so nothing existing moves or is rewritten.
    # Any other architecture writes into its own subtree -- outputs/crnn/{metrics,timing}.json,
    # seed_metrics/, checkpoints and plots -- because a second model writing model_s{seed}.pt and
    # metrics.json would silently overwrite the CNN results that noise_eval and ensemble_eval read.
    model_cls = ARCHITECTURES[args.model]
    out = OUTPUTS if args.model == "cnn" else OUTPUTS / args.model
    store = SEED_METRICS_DIR if args.model == "cnn" else out / "seed_metrics"
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device} | torch {torch.__version__} | seeds {args.seeds}")
    print(f"architecture: {model_cls.__name__} -> {out}")
    print(f"{len(CLASSES)} classes: {', '.join(CLASSES)}\n")

    manifest, splits, by_id = load_manifest()
    Xtr, ytr, _ = load_split(splits["train"], by_id)
    Xva, yva, _ = load_split(splits["val"], by_id)
    Xte, yte, test_ids = load_split(splits["test"], by_id)
    frames = [s.shape[-1] for s in Xtr]
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} clips")
    print(f"clip frames: {min(frames)}-{max(frames)}, {len(set(frames))} distinct "
          f"({'fixed' if len(set(frames)) == 1 else 'VARIABLE'})")
    class_weights(ytr)
    nb = len(LengthBatcher(Xtr, ytr, BATCH_SIZE))
    print(f"length-bucketed batching: {nb} train batches/epoch, mean size {len(Xtr) / nb:.1f}")
    print(f"model params: {sum(p.numel() for p in MediumCNN().parameters()):,}\n")

    data = ((Xtr, ytr), (Xva, yva), (Xte, yte))

    # Optional pop-up progress bar. Total is an UPPER bound (seeds x MAX_EPOCHS); early
    # stopping means it usually finishes before filling, so we snap it to 100% at the end.
    popup, total = None, len(args.seeds) * MAX_EPOCHS
    if args.progress:
        from ..progress_popup import make_popup
        popup = make_popup(f"Training {len(CLASSES)}-class CNN", total)

    def on_epoch_for(si, seed):
        def cb(epoch):
            if popup is not None:
                done = si * MAX_EPOCHS + epoch
                popup.update(done, f"seed {seed}  epoch {epoch}/{MAX_EPOCHS}  "
                                   f"({100*done/total:.0f}%)")
        return cb

    for si, seed in enumerate(args.seeds):
        r, h, preds, targets = run_seed(seed, data, device, on_epoch_for(si, seed),
                                        ckpt_path=out / f"model_s{seed}.pt",
                                        model_cls=model_cls)
        save_seed_metrics(seed, r, h, preds, targets, store=store)

    # Aggregate over the CANONICAL seed set, pulling in seeds trained by earlier invocations.
    # Aggregating this run's seeds instead is what let `--seeds 42` followed by
    # `--seeds 43 44` write a 2-seed metrics.json that looked like the full 3-seed result.
    results, histories, first = load_seed_metrics(list(SEEDS), store=store)

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

    plot_confusion(cm_sum, out / "confusion_matrix.png",
                   title=f"Confusion matrix — {len(CLASSES)} classes, {len(results)} seeds "
                         f"(row-normalised = recall)")
    plot_curves(histories, out / "curves.png")
    preds, targets = first
    n_wrong = plot_misclassified(test_ids, preds, targets, out / "misclassified.png")

    timing = {"device": str(device), "n_seeds": len(results),
              "mean_epoch_s": agg([r["mean_epoch_s"] for r in results]),
              "total_s_per_seed": agg([r["total_s"] for r in results]),
              "wall_clock_all_seeds_s": float(sum(r["total_s"] for r in results)),
              "n_train": int(len(ytr)), "n_classes": len(CLASSES)}
    print(f"\ntiming: {timing['mean_epoch_s']['mean']:.2f}s/epoch, "
          f"{timing['total_s_per_seed']['mean']:.0f}s/seed, "
          f"{timing['wall_clock_all_seeds_s']:.0f}s total ({device})")

    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "classes": list(CLASSES),
        "n_classes": len(CLASSES),
        "chance_balanced_accuracy": chance,
        "seeds": list(SEEDS),          # always the canonical set — see load_seed_metrics()
        "fingerprint": config_fingerprint(),
        "device": str(device),
        "articulation_mode": manifest["articulation_mode"],
        "sample_rate": manifest.get("sample_rate"),
        "bitrate_kbps_by_class": manifest.get("bitrate_kbps_by_class"),
        "architecture": model_cls.__name__,
        "n_params": int(sum(p.numel() for p in model_cls().parameters())),
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
    print(f"wrote {out / 'metrics.json'}, {out / 'timing.json'}, and plots")

    if popup is not None:
        popup.update(total, "done")
        popup.close()


if __name__ == "__main__":
    main()
