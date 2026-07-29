"""Bound what an order-sensitive model can extract from temporal structure alone.

    python -m instrument_robustness.single.envelope_probe

WHY THIS EXISTS, AND WHY IT RUNS BEFORE THE CRNN. 97.3% of clips in this cache are TILED — a short
note looped to fill 3.0 s — and source note length correlates with class (a length-only classifier
scores 0.1977 against 0.0833 chance, FINDINGS §8). Tiling therefore writes note length into the
time axis as a periodic repetition. A GAP-CNN discards where features sit in time, so it cannot
read that period; a CRNN keeps the sequence and can.

Rather than train a CRNN and try to infer afterwards whether it cheated, this measures the CEILING
of the shortcut directly. Collapse each cached spectrogram over frequency and you are left with a
130-frame energy envelope carrying essentially no timbre, but retaining the tiling period intact.
Whatever a classifier scores on that is an upper bound on what ANY order-sensitive architecture
can get from temporal structure for free — CRNN, transformer, or an SVM on onset-rate features.

THREE MEASUREMENTS, in increasing specificity:
  1. envelope     -- the whole 130-frame envelope. Broadest: includes attack sharpness and decay
                     rate, which are legitimate timbre-adjacent cues, so this OVERSTATES the
                     tiling-specific risk.
  2. autocorr     -- autocorrelation of the envelope. Periodicity with the envelope's overall
                     shape largely divided out; closer to "tiling period only".
  3. period alone -- a single number, the dominant autocorrelation lag. Narrowest and the most
                     direct proxy for "the model read the loop period".

Also reported: how well the estimated period recovers the true source length. If it does not, the
period is not actually recoverable from the spectrogram and the whole concern is moot.

PRE-REGISTERED INTERPRETATION (written before running):
  * period-alone near chance (<= ~0.10)
        -> tiling period is not usefully recoverable; the CRNN inherits no tiling shortcut and
           needs no special caveat.
  * period-alone clearly above chance but well under the 0.1977 source-length ceiling
        -> a partial shortcut exists. Report the number beside any CRNN score and re-run the
           FINDINGS §6 per-class check on the trained CRNN.
  * period-alone at or above 0.1977
        -> tiling has made note length MORE accessible than it was in the raw durations. Treat any
           CRNN result as suspect until the §6 test clears it, and consider randomising tiling
           phase or repeat count in prep_data.

Scores are reported both TEST-scored (honest generalisation) and TRAIN-scored (optimistic upper
bound, directly comparable to the 0.1977 figure, which was also fit and scored on train).
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ..cnn_core import load_manifest, load_split
from ..config import CLASSES, HOP_LENGTH, OUTPUTS, SR, config_fingerprint

RESULTS_JSON = OUTPUTS / "envelope_probe.json"
CHANCE = 1.0 / len(CLASSES)


def envelope(spec):
    """Per-frame energy of one log-mel spectrogram, z-scored.

    Preconditions: spec is (1, n_mels, frames) as returned by load_split.
    Postcondition: returns (frames,), zero mean and unit variance, so overall clip loudness
    cannot leak — only the SHAPE of the energy over time survives.
    """
    e = np.asarray(spec).squeeze(0).mean(axis=0)
    return (e - e.mean()) / (e.std() + 1e-8)


def autocorr(env):
    """Normalised autocorrelation of an envelope, positive lags only.

    Postcondition: returns (frames,), value at lag 0 equal to 1. A clip tiled with period p shows
    a peak at lag p.
    """
    n = len(env)
    a = np.correlate(env, env, mode="full")[n - 1:]
    return a / (a[0] + 1e-8)


def dominant_lag(ac, min_lag=2):
    """The lag of the highest autocorrelation peak past the main lobe = estimated tiling period.

    Postcondition: returns a lag in frames, or 0 if the autocorrelation never turns back up.

    The naive `argmax(ac[2:])` does NOT work here and was tried first. A sustained note is smooth
    frame to frame, so its autocorrelation decays slowly from lag 0 and the maximum over all
    lags >= 2 is just lag 2 — measuring spectral smoothness, not tiling period. (On real clips
    that estimator correlated with true source length at r = -0.15, i.e. not at all.) The standard
    fix, borrowed from autocorrelation pitch detection: walk down the main lobe to its first local
    minimum, then take the largest peak after it.
    """
    n = len(ac)
    i = min_lag
    while i < n - 1 and ac[i + 1] < ac[i]:      # descend the main lobe
        i += 1
    if i >= n - 1:
        return 0
    return int(np.argmax(ac[i:]) + i)


def fit_score(Xtr, ytr, Xte, yte, model):
    model.fit(Xtr, ytr)
    return {"test": float(balanced_accuracy_score(yte, model.predict(Xte))),
            "train": float(balanced_accuracy_score(ytr, model.predict(Xtr)))}


def main():
    manifest, splits, by_id = load_manifest()
    Xtr_s, ytr, tr_ids = load_split(splits["train"], by_id)
    Xte_s, yte, te_ids = load_split(splits["test"], by_id)
    ytr, yte = ytr.numpy(), yte.numpy()
    print(f"train {len(Xtr_s)} | test {len(Xte_s)} | chance {CHANCE:.4f}")

    Etr = np.stack([envelope(s) for s in Xtr_s])
    Ete = np.stack([envelope(s) for s in Xte_s])
    Atr = np.stack([autocorr(e) for e in Etr])
    Ate = np.stack([autocorr(e) for e in Ete])
    Ptr = np.array([[dominant_lag(a)] for a in Atr], dtype=float)
    Pte = np.array([[dominant_lag(a)] for a in Ate], dtype=float)
    print(f"envelope {Etr.shape[1]} frames | autocorr {Atr.shape[1]} lags\n")

    # Does the estimated period actually recover the true source length? If not, nothing else here
    # matters -- the shortcut would not be reachable from the spectrogram in the first place.
    src = np.array([by_id[i]["source_seconds"] for i in tr_ids], dtype=float)
    tiled = src < (manifest["signal_stats"]["fixed_clip_seconds"] - 1e-6)
    lag_s = Ptr[:, 0] * HOP_LENGTH / SR
    r = float(np.corrcoef(lag_s[tiled], src[tiled])[0, 1]) if tiled.sum() > 2 else float("nan")
    print(f"estimated period vs true source length (tiled clips only, n={int(tiled.sum())}): "
          f"r = {r:+.3f}")

    scaler = StandardScaler()
    results = {
        "envelope": fit_score(scaler.fit_transform(Etr), ytr, scaler.transform(Ete), yte,
                              LogisticRegression(max_iter=2000)),
        "autocorr": fit_score(scaler.fit_transform(Atr), ytr, scaler.transform(Ate), yte,
                              LogisticRegression(max_iter=2000)),
        # depth-4 tree on one feature, matching the existing source-length probe's form
        "period_only": fit_score(Ptr, ytr, Pte, yte, DecisionTreeClassifier(max_depth=4,
                                                                           random_state=0)),
    }

    print(f"\n{'feature set':<14}{'test':>9}{'train':>9}{'lift(test)':>12}")
    for name, sc in results.items():
        print(f"{name:<14}{sc['test']:>9.4f}{sc['train']:>9.4f}{sc['test'] - CHANCE:>+12.4f}")

    period = results["period_only"]["train"]
    RESULTS_JSON.write_text(json.dumps({
        "fingerprint": config_fingerprint(),
        "chance": CHANCE,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "period_vs_source_length_r": r,
        "source_length_ceiling_train": 0.19769815991934794,   # FINDINGS §8, same train-scored form
        "results": results,
    }, indent=2))

    print("\n" + "=" * 70)
    print(f"period-alone (train-scored, comparable to the 0.1977 source-length ceiling): {period:.4f}")
    if results["period_only"]["test"] <= 0.10:
        print("-> at/near chance on test: tiling period is not usefully recoverable.")
        print("   The CRNN inherits no tiling shortcut from this; no special caveat needed.")
    elif period < 0.1977:
        print("-> a PARTIAL shortcut exists, below the raw source-length ceiling.")
        print("   Report this beside any CRNN score, and re-run the FINDINGS S6 per-class")
        print("   check on the trained CRNN before publishing its number.")
    else:
        print("-> tiling has made note length MORE accessible than the raw durations were.")
        print("   Treat any CRNN result as suspect until S6 clears it, and consider")
        print("   randomising tiling phase or repeat count in prep_data.")
    print("=" * 70)
    print(f"\nwrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
