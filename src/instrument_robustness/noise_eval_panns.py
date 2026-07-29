"""noise_eval_panns.py — run the clean-trained fine-tuned PANNs across every noise condition.

Inference only, no retraining. Reads the SHARED noisy windows written by noise_sweep.py so the
predictions stay paired with every other model's (McNemar / cluster bootstrap later).

Featurization goes through the SAME path the model was trained on (pretrained_extractors.panns_input:
22050 window -> 32 kHz waveform; CNN14 computes its own log-mel internally). PANNs carries its own
normalization, so no dataset stats are recomputed on noisy audio.

CLEAN-PARITY GATE: before any noisy condition is scored, the `clean` condition must reproduce the
model's official clean macro-F1 (from results_finetune.json) to within CLEAN_PARITY_TOL. If it does
not, this evaluator is reading different audio or preprocessing than training did, and every noisy
number below it would be measuring the wrong thing. It aborts rather than reporting.

Outputs, under artifacts/<model>/noise/ (alongside artifacts/svm, artifacts/mert):
    panns_ft_test_{condition}.csv   per-clip: window_id, cluster keys, true, pred, per-class scores
    metrics_{condition}.json        accuracy, macro-F1, per-class P/R/F1, confusion
    noise_sweep_summary.csv         tidy condition x (accuracy, macro-F1)

Per-class columns are named `probability_<class>` when the model emits calibrated probabilities and
`score_<class>` when it emits uncalibrated decision values. Models without probabilities (e.g. an
SVC fitted with probability=False) must NOT be refitted just to satisfy a file schema — paired
accuracy tests need only the predicted label.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json, warnings
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
from instrument_robustness.config import (ROOT, DATA_ROOT, FEATURES, SR, TARGET_LABELS,
                                          WINDOWS_CSV, config_fingerprint)
from instrument_robustness.pretrained_extractors import panns_input
from instrument_robustness.noise_sweep import (SNRS, NOISE_TYPES, NOISY_DIR, window_id_of,
                                               dataset_fingerprint)
warnings.filterwarnings("ignore")

N = len(TARGET_LABELS)
LAB2IDX = {l: i for i, l in enumerate(TARGET_LABELS)}
MODEL_DIR = FEATURES / "panns"                       # where the checkpoint lives (data root)
OUT = DATA_ROOT / "artifacts" / "panns" / "noise"    # where results go (#7)
CLIP_LEN = int(round(3.0 * SR))
CLEAN_PARITY_TOL = 1e-3     # macro-F1 must reproduce the official clean number to this tolerance


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
    ckpt = torch.load(MODEL_DIR / "panns_finetune.pt", map_location="cpu")
    # train_panns saves {state_dict, label_order, config_fingerprint}; older runs saved a bare
    # state_dict. Accept either.
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    order = ckpt.get("label_order") if isinstance(ckpt, dict) else None
    if order is not None and list(order) != list(TARGET_LABELS):
        # Label index order defines what every prediction MEANS. A mismatch would silently
        # relabel every output instead of failing, so refuse rather than guess.
        raise SystemExit(
            "checkpoint label order does not match TARGET_LABELS.\n"
            f"  ckpt: {list(order)}\n  cfg : {list(TARGET_LABELS)}\n"
            "Retrain, or align config.CANONICAL_LABELS."
        )
    m.load_state_dict(state)
    return m.eval().to(dev)


def official_clean_macro_f1():
    """The model's clean test macro-F1 as recorded at training time, for the parity gate."""
    p = MODEL_DIR / "results_finetune.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["test"]["macro_f1"])
    except (KeyError, ValueError, TypeError):
        return None


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
    return [NOISY_DIR / noise_type / f"snr{snr}" / f"{window_id_of(p)}.wav"
            for p in df["window_path"]]


@torch.no_grad()
def run_condition(model, dev, df, noise_type, snr):
    paths = condition_paths(df, noise_type, snr)
    probs, B = [], 32
    for i in range(0, len(paths), B):
        wavs = [panns_input(read_wav(p)) for p in paths[i:i + B]]
        x = torch.from_numpy(np.stack(wavs)).float().to(dev)
        probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(probs)


def cluster_keys(df):
    """Grouping columns for the paired CLUSTER bootstrap (#3).

    Windows from one recording -- and beyond that, every recording of the same (instrument, note)
    pitch group -- are near-duplicates, not independent draws. Bootstrapping over windows would
    report tighter intervals than the data supports. These columns let the analysis resample whole
    clusters. `pitch_group` matches the unit step3 splits on, so it is the conservative choice.
    """
    out = {"source_path": df["source_path"].values}
    if "note" in df.columns:
        out["pitch_group"] = (df["label"].astype(str) + "_" + df["note"].astype(str)).values
    else:
        # windows.csv carries source_path but not note; recover the pitch group from the manifest
        out["pitch_group"] = df["source_path"].astype(str).values
    return out


def main():
    dev = get_device()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(WINDOWS_CSV)
    df = df[df.split == "test"].reset_index(drop=True)
    # attach `note` from the labeled manifest so the pitch-group cluster key is available
    man = DATA_ROOT / "pipeline" / "manifest_labeled.csv"
    if man.exists():
        m = pd.read_csv(man)
        if "note" in m.columns:
            df = df.merge(m[["path", "note"]].rename(columns={"path": "source_path"}),
                          on="source_path", how="left")
    y_true = np.array([LAB2IDX[l] for l in df["label"]])
    model = build_model(dev)
    clusters = cluster_keys(df)
    print(f"device={dev} | {len(df)} test windows | model=panns_finetune (clean-trained)")
    print(f"dataset fingerprint: {dataset_fingerprint()}")
    print(f"clusters: {len(set(clusters['source_path']))} source files, "
          f"{len(set(clusters['pitch_group']))} pitch groups\n")

    conditions = [("clean", "clean")] + [(nt, snr) for nt in NOISE_TYPES for snr in SNRS]
    summary = []
    for noise_type, snr in conditions:
        probs = run_condition(model, dev, df, noise_type, snr)
        pred = probs.argmax(1)
        acc = accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
        tag = "clean" if noise_type == "clean" else f"{noise_type}_{snr}"

        # ---- clean-parity gate (#2): abort before scoring noise if clean does not reproduce ----
        if noise_type == "clean":
            official = official_clean_macro_f1()
            if official is None:
                print("  WARNING: no official clean macro-F1 found; parity gate skipped")
            elif abs(mf1 - official) > CLEAN_PARITY_TOL:
                raise SystemExit(
                    f"CLEAN PARITY FAILED: this evaluator scores clean macro-F1 {mf1:.6f} but the "
                    f"model's official clean result is {official:.6f} "
                    f"(|diff| {abs(mf1-official):.6f} > {CLEAN_PARITY_TOL}).\n"
                    "The noisy conditions would be measuring a different input or preprocessing "
                    "path than training used. Fix that before trusting any noise number."
                )
            else:
                print(f"  clean-parity OK: {mf1:.6f} vs official {official:.6f}")

        out = pd.DataFrame({
            "window_id": [window_id_of(p) for p in df["window_path"]],
            "source_path": clusters["source_path"],      # cluster key for the paired bootstrap
            "pitch_group": clusters["pitch_group"],      # stricter cluster key (step3's split unit)
            "true_label": [TARGET_LABELS[i] for i in y_true],
            "predicted_label": [TARGET_LABELS[i] for i in pred],
            "correct": y_true == pred,
        })
        # PANNs emits a softmax, i.e. genuine probabilities -> `probability_` prefix.
        # A model with only decision values should write `score_<class>` instead, and one with
        # neither may omit these columns entirely (see module docstring).
        for j, lab in enumerate(TARGET_LABELS):
            out[f"probability_{lab}"] = probs[:, j].round(6)
        out.to_csv(OUT / f"panns_ft_test_{tag}.csv", index=False)

        rep = classification_report(y_true, pred, labels=range(N), target_names=TARGET_LABELS,
                                    output_dict=True, zero_division=0)
        with open(OUT / f"metrics_{tag}.json", "w") as f:
            json.dump({"condition": tag, "noise_type": noise_type, "snr_db": snr,
                       "n": int(len(df)), "accuracy": round(float(acc), 4),
                       "macro_f1": round(float(mf1), 4),
                       "dataset_fingerprint": dataset_fingerprint(),
                       "config_fingerprint": config_fingerprint(),
                       "score_type": "probability",
                       "classification_report": rep,
                       "confusion_matrix": confusion_matrix(y_true, pred, labels=range(N)).tolist()},
                      f, indent=2)

        summary.append({"noise_type": noise_type, "snr_db": snr, "condition": tag,
                        "accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4)})
        print(f"  {tag:<16} acc {acc:.4f}   macro-F1 {mf1:.4f}")

    s = pd.DataFrame(summary)
    s.to_csv(OUT / "noise_sweep_summary.csv", index=False)
    print("\n" + "=" * 60)
    print(f"NOISE SWEEP — PANNs fine-tune (clean-trained), {ROOT.name} test")
    print("=" * 60)
    clean_f1 = s[s.condition == "clean"]["macro_f1"].iloc[0]
    print(f"{'condition':<18}{'accuracy':>10}{'macro-F1':>10}{'vs clean':>10}")
    for _, r in s.iterrows():
        d = "" if r.condition == "clean" else f"{r.macro_f1 - clean_f1:+.4f}"
        print(f"{r.condition:<18}{r.accuracy:>10.4f}{r.macro_f1:>10.4f}{d:>10}")
    print(f"\nwrote {len(conditions)} predictions CSVs + metrics JSONs + summary to {OUT}")


if __name__ == "__main__":
    main()
