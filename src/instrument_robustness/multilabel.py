"""Path A toward a multiple-instrument classifier: synthetic mixtures + multi-label CNN.

Sums k isolated notes into one clip and labels it with the SET of instruments present. The
CNN backbone (MediumCNN) is unchanged — its 12 logits are now read as 12 independent
sigmoids ("is instrument j present?") and trained with binary cross-entropy instead of
softmax cross-entropy.

Leakage-safe: mixture sources are drawn PER SPLIT from the pitch-grouped splits.json, so no
source note appears in both train and test. Mixtures are a FIXED set per split (seeded), so
eval is deterministic.

CAVEAT (see config.MIX_*): summed studio notes are not real polyphony. This validates the
multi-label machinery and lets mixing be studied cleanly; it is not a substitute for IRMAS.

    python -m instrument_robustness.multilabel [--seeds 42 43 44]
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
from sklearn.metrics import average_precision_score, f1_score

from .config import (
    BATCH_SIZE, CLASSES, CLASS_TO_IDX, EARLY_STOP_PATIENCE, LEARNING_RATE, MAX_EPOCHS,
    MIX_COUNTS, MIX_POLYPHONY, MIX_SEED, OUTPUTS, PLATEAU_FACTOR, PLATEAU_PATIENCE, SEEDS,
    SPLITS_JSON, WAVE_DIR,
)
from .prep_data import wav_to_logmel
from .train import (agg, get_device, load_manifest, LengthBatcher, MediumCNN, set_seed)


# --------------------------------------------------------------------------- mixtures

def plan_mixtures(split_ids, by_id, n_mix, seed):
    """Choose (source_ids, instrument_set) for each mixture. k distinct instruments drawn
    uniformly from MIX_POLYPHONY, one random source clip each — all from THIS split."""
    rng = random.Random(seed)
    by_inst = defaultdict(list)
    for i in split_ids:
        by_inst[by_id[i]["instrument"]].append(i)
    insts = sorted(by_inst)
    plans = []
    for _ in range(n_mix):
        k = rng.choice(MIX_POLYPHONY)
        chosen = rng.sample(insts, min(k, len(insts)))
        plans.append(([rng.choice(by_inst[c]) for c in chosen], chosen))
    return plans


def render_mixture(source_ids):
    """Equal-RMS sum of the source waveforms, truncated to the shortest (so every sample is
    real audio — no padding), peak-normalised. Returns the mixed waveform."""
    ys = [np.load(WAVE_DIR / f"{i}.npy") for i in source_ids]
    L = min(y.size for y in ys)
    mix = np.zeros(L, dtype=np.float32)
    for y in ys:
        y = y[:L]
        mix += y / (np.sqrt(np.mean(y ** 2)) + 1e-9)   # equal loudness per source
    return (mix / (np.abs(mix).max() + 1e-9)).astype(np.float32)


def build_split(split_ids, by_id, n_mix, seed):
    """-> (specs list, multihot label tensor, polyphony-count array)."""
    plans = plan_mixtures(split_ids, by_id, n_mix, seed)
    specs, labels, ks = [], [], []
    for source_ids, insts in plans:
        specs.append(torch.from_numpy(wav_to_logmel(render_mixture(source_ids))).float().unsqueeze(0))
        y = np.zeros(len(CLASSES), dtype=np.float32)
        for c in insts:
            y[CLASS_TO_IDX[c]] = 1.0
        labels.append(y)
        ks.append(len(insts))
    return specs, torch.from_numpy(np.stack(labels)), np.array(ks)


# --------------------------------------------------------------------------- metrics

def evaluate(model, specs, labels, device):
    """-> (scores array [N,12] of sigmoid probs, mean BCE loss)."""
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    by_len = defaultdict(list)
    for i, s in enumerate(specs):
        by_len[s.shape[-1]].append(i)
    scores = np.zeros((len(specs), len(CLASSES)), dtype=np.float32)
    total, n = 0.0, 0
    with torch.no_grad():
        for idxs in by_len.values():
            for i in range(0, len(idxs), BATCH_SIZE):
                b = idxs[i:i + BATCH_SIZE]
                X = torch.stack([specs[j] for j in b]).to(device)
                yb = labels[b].to(device)
                logits = model(X)
                total += crit(logits, yb).item() * len(b)
                n += len(b)
                scores[b] = torch.sigmoid(logits).cpu().numpy()
    return scores, total / n


def multilabel_metrics(y_true, y_score):
    """mAP is the headline (mean of per-class average precision — threshold-free, the audio
    tagging standard). F1@0.5 and exact-match are reported for intuition."""
    ap = average_precision_score(y_true, y_score, average=None)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "mAP": float(np.nanmean(ap)),
        "micro_AP": float(average_precision_score(y_true, y_score, average="micro")),
        "macro_f1@0.5": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1@0.5": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float((y_pred == y_true).all(axis=1).mean()),
        "per_class_ap": {c: float(ap[i]) for i, c in enumerate(CLASSES)},
    }


# --------------------------------------------------------------------------- one seed

def run_seed(seed, data, device):
    (Xtr, ytr), (Xva, yva), (Xte, yte, kte) = data
    set_seed(seed)
    train_loader = LengthBatcher(Xtr, ytr, BATCH_SIZE, shuffle=True, seed=seed)
    val_loader = LengthBatcher(Xva, yva, BATCH_SIZE)

    model = MediumCNN().to(device)              # identical backbone; 12 logits read as sigmoids
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=PLATEAU_FACTOR,
                                                       patience=PLATEAU_PATIENCE)
    best_val, best_state, since = float("inf"), None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        _, vl = evaluate(model, Xva, yva, device)
        sched.step(vl)
        if vl < best_val:
            best_val, since = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        print(f"  s{seed} | ep {epoch:>2}/{MAX_EPOCHS} | {time.perf_counter()-t0:4.1f}s | "
              f"val BCE {vl:.4f}{' *' if since == 0 else ''}")
        if since >= EARLY_STOP_PATIENCE:
            print(f"  s{seed} | early stop at {epoch}")
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "seed": seed, "classes": list(CLASSES),
                "task": "multilabel"}, OUTPUTS / f"multilabel_s{seed}.pt")
    scores, _ = evaluate(model, Xte, yte, device)
    y_true = yte.numpy()
    m = multilabel_metrics(y_true, scores)
    # mAP by polyphony level — harder (more instruments) should score lower
    by_k = {}
    for k in sorted(set(kte.tolist())):
        mask = kte == k
        by_k[int(k)] = {"n": int(mask.sum()),
                        "mAP": float(np.nanmean(average_precision_score(
                            y_true[mask], scores[mask], average=None)))}
    print(f"  s{seed} | TEST mAP {m['mAP']:.4f} | micro-AP {m['micro_AP']:.4f} | "
          f"macro-F1@.5 {m['macro_f1@0.5']:.4f} | exact {m['exact_match']:.4f}\n")
    return {"seed": seed, **m, "by_polyphony": by_k}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEEDS[0]],
                    help="seeds to run (default: just the first, for a fast look)")
    args = ap.parse_args()

    device = get_device()
    _, splits, by_id = load_manifest()
    print(f"device: {device} | {len(CLASSES)} classes | polyphony {MIX_POLYPHONY} | "
          f"mixtures {MIX_COUNTS}\n")

    # Fixed mixture sets (seeded per split, disjoint sources across splits).
    print("building mixtures...")
    Xtr, ytr, _ = build_split(splits["train"], by_id, MIX_COUNTS["train"], MIX_SEED)
    Xva, yva, _ = build_split(splits["val"], by_id, MIX_COUNTS["val"], MIX_SEED + 1)
    Xte, yte, kte = build_split(splits["test"], by_id, MIX_COUNTS["test"], MIX_SEED + 2)
    prev = yte.numpy().mean(axis=0)
    print(f"  test label prevalence: mean {prev.mean():.3f} per class "
          f"(random-guess mAP ~ {prev.mean():.3f})")
    print(f"  mean instruments/mixture: {yte.numpy().sum(axis=1).mean():.2f}\n")

    data = ((Xtr, ytr), (Xva, yva), (Xte, yte, kte))
    results = [run_seed(s, data, device) for s in args.seeds]

    print("=" * 70)
    print(f"MULTI-LABEL RESULTS — {len(CLASSES)} classes, {len(results)} seed(s)")
    print("=" * 70)
    chance = float(prev.mean())
    for name in ("mAP", "micro_AP", "macro_f1@0.5", "micro_f1@0.5", "exact_match"):
        a = agg([r[name] for r in results])
        print(f"  {name:<14} {a['mean']:.4f}" + (f" +/- {a['std']:.4f}" if len(results) > 1 else ""))
    print(f"  (random-guess mAP ~ {chance:.3f}; perfect = 1.0)")

    print("\nmAP by polyphony (more instruments = harder):")
    for k in sorted(results[0]["by_polyphony"]):
        a = agg([r["by_polyphony"][k]["mAP"] for r in results])
        print(f"  {k} instrument(s) (n={results[0]['by_polyphony'][k]['n']:>3}): mAP {a['mean']:.4f}")

    print("\nper-class AP (mean over seeds), hardest first:")
    order = sorted(CLASSES, key=lambda c: np.mean([r["per_class_ap"][c] for r in results]))
    for c in order:
        v = np.mean([r["per_class_ap"][c] for r in results])
        print(f"  {c:<13} {v:.4f}")

    (OUTPUTS / "multilabel_results.json").write_text(json.dumps({
        "classes": list(CLASSES), "polyphony": list(MIX_POLYPHONY),
        "mix_counts": MIX_COUNTS, "random_map": chance, "seeds": args.seeds,
        "results": results,
    }, indent=2))
    print(f"\nwrote {OUTPUTS / 'multilabel_results.json'}")


if __name__ == "__main__":
    main()
