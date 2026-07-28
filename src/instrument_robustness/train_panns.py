"""train_panns.py — PANNs CNN14 for 9-class instrument classification.

Two modes:
  probe    : freeze the pretrained CNN14 trunk, extract 2048-d embeddings ONCE (cached to
             features/panns/emb_<split>.npz), then train a linear head. Fast (~minutes).
  finetune : backprop through the whole trunk + head. Slower (~4 min/epoch on this Mac's MPS).

Design notes:
  * Input is the raw 3 s window resampled 22050 -> 32000 Hz (pretrained_extractors.panns_input);
    CNN14 computes its own log-mel + normalization internally (it does NOT use the Step-6 stats).
  * Classification head reads CNN14's 2048-d `embedding` with softmax/CrossEntropy — NOT the
    pretrained sigmoid AudioSet head.
  * Split comes straight from windows.csv (source-level split) — no leakage re-introduced.
  * Gentle inverse-frequency class weights (window imbalance is only ~2.3x); model selection and
    reporting use macro-F1.

Usage:
    python -m instrument_robustness.train_panns --mode probe
    python -m instrument_robustness.train_panns --mode finetune --epochs 25
    python -m instrument_robustness.train_panns --smoke        # tiny end-to-end sanity run
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # keep librosa resampling in DataLoader workers single-threaded

import argparse, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from instrument_robustness.config import ROOT, PIPE, FEATURES, TARGET_LABELS
from instrument_robustness.featurelib import load_window
from instrument_robustness.pretrained_extractors import panns_input, PANNS_SR

LABEL2IDX = {l: i for i, l in enumerate(TARGET_LABELS)}
N_CLASSES = len(TARGET_LABELS)
CKPT = ROOT / "checkpoints" / "Cnn14_mAP=0.431.pth"
OUT = FEATURES / "panns"           # cache + results (gitignored)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- data
class WindowWaveformDataset(Dataset):
    """Yields (waveform@32k float32 tensor, label_idx) for each window."""
    def __init__(self, df):
        self.paths = df["window_path"].tolist()
        self.y = [LABEL2IDX[l] for l in df["label"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        y = load_window(ROOT / self.paths[i])   # 22050 Hz, 3 s
        w = panns_input(y)                       # -> float32 @ 32 kHz
        return torch.from_numpy(w), self.y[i]


def load_split(split, limit=None):
    df = pd.read_csv(PIPE / "windows.csv")
    df = df[df.split == split].reset_index(drop=True)
    return df.iloc[:limit].copy() if limit else df


def class_weights(df):
    counts = (df["label"].map(LABEL2IDX).value_counts()
              .reindex(range(N_CLASSES)).fillna(0).values)
    n, k = counts.sum(), N_CLASSES
    w = n / (k * np.maximum(counts, 1))          # inverse frequency, mean ~1
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- model
def build_backbone():
    """Pretrained CNN14 (AudioSet), weights loaded, 527-way head intact (we read `embedding`)."""
    from panns_inference.models import Cnn14
    m = Cnn14(sample_rate=PANNS_SR, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    return m


class PannsClassifier(nn.Module):
    def __init__(self, backbone, freeze=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(2048, N_CLASSES)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.head(self.backbone(x)["embedding"])

    @torch.no_grad()
    def embed(self, x):
        return self.backbone(x)["embedding"]


# --------------------------------------------------------------------------- metrics
def report(y_true, y_pred):
    per = f1_score(y_true, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float((np.asarray(y_true) == np.asarray(y_pred)).mean()), 4),
        "per_class_f1": {TARGET_LABELS[i]: round(float(per[i]), 4) for i in range(N_CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(N_CLASSES)).tolist(),
    }


@torch.no_grad()
def predict_full(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------- embeddings (probe)
def get_embeddings(model, df, split, device, args):
    cache = OUT / f"emb_{split}.npz"
    if cache.exists() and not args.force:
        d = np.load(cache)
        if len(d["y"]) == len(df):
            print(f"  [{split}] embeddings from cache: {d['E'].shape}")
            return d["E"], d["y"]
    loader = DataLoader(WindowWaveformDataset(df), batch_size=args.batch_size,
                        num_workers=args.num_workers)
    Es, ys = [], []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            Es.append(model.embed(x.to(device)).cpu().numpy())
            ys.append(np.asarray(y))
    E, Y = np.concatenate(Es), np.concatenate(ys)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, E=E, y=Y)
    print(f"  [{split}] extracted {E.shape} in {time.time()-t0:.0f}s -> {cache.name}")
    return E, Y


def run_probe(args, device):
    dfs = {s: load_split(s, args.limit) for s in ("train", "val", "test")}
    model = PannsClassifier(build_backbone(), freeze=True).to(device)
    E, Y = {}, {}
    print("extracting embeddings (frozen trunk):")
    for s in ("train", "val", "test"):
        E[s], Y[s] = get_embeddings(model, dfs[s], s, device, args)

    Etr = torch.from_numpy(E["train"]).to(device)
    Ytr = torch.from_numpy(Y["train"]).long().to(device)
    Eva = torch.from_numpy(E["val"]).to(device)

    head = nn.Linear(2048, N_CLASSES).to(device)
    lossfn = nn.CrossEntropyLoss(weight=class_weights(dfs["train"]).to(device))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    best, best_state, bad, bs, n = -1.0, None, 0, 256, len(Ytr)
    print("training linear head on cached embeddings:")
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(head(Etr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pv = head(Eva).argmax(1).cpu().numpy()
        f1 = f1_score(Y["val"], pv, average="macro", zero_division=0)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 20 == 0 or bad == 0:
            print(f"  epoch {ep+1:3d}  val_macroF1={f1:.4f}  (best {best:.4f})")
        if bad >= args.patience:
            print(f"  early stop at epoch {ep+1}")
            break

    head.load_state_dict(best_state)
    with torch.no_grad():
        pv = head(Eva).argmax(1).cpu().numpy()
        pt = head(torch.from_numpy(E["test"]).to(device)).argmax(1).cpu().numpy()
    return {"val": report(Y["val"], pv), "test": report(Y["test"], pt)}, head


def run_finetune(args, device):
    dfs = {s: load_split(s, args.limit) for s in ("train", "val", "test")}
    model = PannsClassifier(build_backbone(), freeze=False).to(device)
    lossfn = nn.CrossEntropyLoss(weight=class_weights(dfs["train"]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr = DataLoader(WindowWaveformDataset(dfs["train"]), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers)
    va = DataLoader(WindowWaveformDataset(dfs["val"]), batch_size=args.batch_size,
                    num_workers=args.num_workers)

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        yv, pv = predict_full(model, va, device)
        f1 = f1_score(yv, pv, average="macro", zero_division=0)
        print(f"epoch {ep+1:2d}/{args.epochs}  loss={tot/len(dfs['train']):.3f}  "
              f"val_macroF1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            print(f"early stop at epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    yv, pv = predict_full(model, va, device)
    yt, pt = predict_full(model, DataLoader(WindowWaveformDataset(dfs["test"]),
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers), device)
    return {"val": report(yv, pv), "test": report(yt, pt)}, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    ap.add_argument("--epochs", type=int, default=None, help="default 200 (probe) / 25 (finetune)")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-3 (probe) / 1e-4 (finetune)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="recompute cached embeddings (probe)")
    ap.add_argument("--limit", type=int, default=None, help="cap windows/split (debug)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity run then exit")
    args = ap.parse_args()

    if args.smoke:
        args.limit, args.epochs, args.num_workers = 24, 2, 0
    if args.epochs is None:
        args.epochs = 200 if args.mode == "probe" else 25
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "probe" else 1e-4

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    if not CKPT.exists():
        raise SystemExit(f"checkpoint not found: {CKPT}\n(see features/panns/EXTRACTION_PLAN.md)")
    print(f"device={device}  mode={args.mode}  epochs={args.epochs}  lr={args.lr}  "
          f"batch={args.batch_size}  workers={args.num_workers}"
          + ("  [SMOKE]" if args.smoke else ""))

    t0 = time.time()
    runner = run_probe if args.mode == "probe" else run_finetune
    metrics, model = runner(args, device)
    metrics["meta"] = {"mode": args.mode, "device": device, "epochs": args.epochs,
                       "lr": args.lr, "seed": args.seed, "minutes": round((time.time()-t0)/60, 1),
                       "n_train": int(len(load_split("train", args.limit)))}

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"results_{args.mode}{'_smoke' if args.smoke else ''}.json"
    ckpt_path = OUT / f"panns_{args.mode}{'_smoke' if args.smoke else ''}.pt"
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), ckpt_path)

    print(f"\n==== PANNs {args.mode} done in {metrics['meta']['minutes']} min ====")
    print(f"VAL  macro-F1 {metrics['val']['macro_f1']}   acc {metrics['val']['accuracy']}")
    print(f"TEST macro-F1 {metrics['test']['macro_f1']}   acc {metrics['test']['accuracy']}")
    print("TEST per-class F1:", metrics["test"]["per_class_f1"])
    print(f"wrote {res_path.name} + {ckpt_path.name} under features/panns/")


if __name__ == "__main__":
    main()
