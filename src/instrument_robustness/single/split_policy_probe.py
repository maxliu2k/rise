"""Measure how much a pitch-leaking split inflates the reported score.

    python -m instrument_robustness.single.split_policy_probe

The claim under test: splitting by SOURCE FILE rather than by PITCH GROUP scatters
near-duplicates (the same note at different dynamics) across train and test, so the model is
scored on pitches it has effectively already seen. This quantifies the effect on otherwise
identical data.

This matters beyond this branch. `main`'s pipeline used a file-level split until 2026-07-28;
measured on its own committed splits.csv, 406 of 436 pitch-groups (93.1%) spanned more than one
split and 8036 of 8096 source files (99.3%) sat in a leaking group. Every number that pipeline
produced carries whatever inflation this probe measures.

PRE-REGISTERED INTERPRETATION (written before running):
  * random arm scores HIGHER than grouped by more than the seed spread (0.0138)
        -> the leak inflates the score; magnitude is the correction factor for pre-fix numbers.
  * the two arms differ by less than the seed spread
        -> pitch grouping is not buying accuracy honesty on this dataset, and the earlier
           single-seed estimate of +3.7 points was noise. Say so and retract it.
  * random arm scores LOWER
        -> something is wrong with the probe, not a finding. Investigate before reporting.

WHAT THIS DOES NOT TOUCH: the canonical splits.json, metrics.json, seed_metrics/, or
outputs/model_s*.pt. Checkpoints for the random arm go to outputs/split_probe/ so they can never
be mistaken for, or loaded instead of, the real ones.
"""
import json
import random
from collections import defaultdict

import numpy as np

from ..cnn_core import get_device, load_manifest, load_split
from ..config import (CLASSES, OUTPUTS, SEEDS, SPLIT_FRACTIONS, config_fingerprint)
from .train import load_seed_metrics, run_seed

PROBE_DIR = OUTPUTS / "split_probe"
RESULTS_JSON = OUTPUTS / "split_policy_probe.json"

# Fixed, and deliberately NOT config.SEED: the random split is a property of this probe, not of
# the dataset, and reusing the cache's split seed would invite confusing the two.
SPLIT_RNG_SEED = 20260728


def group_key(rec):
    """The near-duplicate unit: same instrument, same note, any dynamic or duration."""
    return f"{rec['instrument']}_{rec['note']}"


def random_split(records, rng):
    """Assign clips to train/val/test at random, stratified by class, IGNORING pitch groups.

    Preconditions: `records` is the full record list; each has 'id' and 'label'.
    Postcondition: returns {"train": [...], "val": [...], "test": [...]} of clip ids, partitioning
    every record exactly once, with each class split to SPLIT_FRACTIONS as closely as integer
    counts allow.

    This is the WRONG way to split this dataset. It exists so the cost of doing it can be
    measured rather than argued about.
    """
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r["id"])
    out = {"train": [], "val": [], "test": []}
    for label in sorted(by_class):
        ids = sorted(by_class[label])          # sort first so the shuffle is reproducible
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * SPLIT_FRACTIONS["train"]))
        n_va = int(round(n * SPLIT_FRACTIONS["val"]))
        out["train"] += ids[:n_tr]
        out["val"] += ids[n_tr:n_tr + n_va]
        out["test"] += ids[n_tr + n_va:]
    assert sum(len(v) for v in out.values()) == len(records), "split lost or duplicated clips"
    return out


def leak_fraction(splits, by_id):
    """Fraction of pitch-groups that span more than one split.

    Postcondition: returns (fraction_of_groups_leaking, n_groups). 0.0 means every group lives
    in exactly one split, which is what the grouped split guarantees.
    """
    where = defaultdict(set)
    for name, ids in splits.items():
        for i in ids:
            where[group_key(by_id[i])].add(name)
    leaking = sum(1 for v in where.values() if len(v) > 1)
    return leaking / max(len(where), 1), len(where)


def build_data(splits, by_id):
    Xtr, ytr, _ = load_split(splits["train"], by_id)
    Xva, yva, _ = load_split(splits["val"], by_id)
    Xte, yte, _ = load_split(splits["test"], by_id)
    return ((Xtr, ytr), (Xva, yva), (Xte, yte))


def summarise(baccs):
    a = np.array(baccs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "per_seed": [float(x) for x in a], "n": int(a.size)}


def main():
    device = get_device()
    manifest, grouped, by_id = load_manifest()
    records = manifest["records"]
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SPLIT_RNG_SEED)
    randomly = random_split(records, rng)

    g_leak, n_groups = leak_fraction(grouped, by_id)
    r_leak, _ = leak_fraction(randomly, by_id)
    print(f"device: {device} | {len(records)} clips | {n_groups} pitch-groups")
    print(f"grouped split : leak {g_leak:.4f}  "
          f"(train {len(grouped['train'])}/val {len(grouped['val'])}/test {len(grouped['test'])})")
    print(f"random  split : leak {r_leak:.4f}  "
          f"(train {len(randomly['train'])}/val {len(randomly['val'])}/test {len(randomly['test'])})")
    assert g_leak == 0.0, "the canonical split is supposed to be leak-free — check prep_data"
    if r_leak < 0.5:
        print(f"\nWARNING: the random split only leaks {r_leak:.1%} of groups. With so little "
              f"contamination this probe cannot measure much; treat the result as a lower bound.")
    print()

    # The grouped arm is already trained: outputs/seed_metrics/ is the canonical store, produced
    # by the same run_seed on the same data. Re-running it would burn ~90 min to reproduce
    # numbers already verified reproducible twice. One seed IS re-run below as a harness check.
    grouped_results, _, _ = load_seed_metrics(list(SEEDS))
    grouped_baccs = [r["test_balanced_accuracy"] for r in grouped_results]
    print(f"grouped arm (canonical, from outputs/seed_metrics/): "
          f"{', '.join(f'{b:.4f}' for b in grouped_baccs)}\n")

    # Harness check: seed 42 through THIS script's code path must reproduce the canonical number.
    # If it does not, the two arms are not comparable and the probe is measuring its own harness.
    print("harness check — grouped split, seed 42, via this script:")
    g_data = build_data(grouped, by_id)
    check, _, _, _ = run_seed(SEEDS[0], g_data, device,
                              ckpt_path=PROBE_DIR / f"grouped_s{SEEDS[0]}.pt")
    expected = grouped_baccs[0]
    got = check["test_balanced_accuracy"]
    print(f"  reproduced {got:.4f} | canonical {expected:.4f} | delta {abs(got - expected):.2e}")
    if abs(got - expected) > 1e-9:
        raise SystemExit(
            f"ERROR: the harness does not reproduce the canonical grouped result "
            f"({got:.6f} vs {expected:.6f}). The two arms would not be comparable, so the "
            f"comparison is meaningless. Fix this before trusting any random-arm number.")
    print("  OK — identical, so the arms are comparable\n")

    print("random arm:")
    r_data = build_data(randomly, by_id)
    random_baccs = []
    for seed in SEEDS:
        res, _, _, _ = run_seed(seed, r_data, device, ckpt_path=PROBE_DIR / f"random_s{seed}.pt")
        random_baccs.append(res["test_balanced_accuracy"])
        print(f"  s{seed} random-split test balanced acc {res['test_balanced_accuracy']:.4f}\n")

    g, r = summarise(grouped_baccs), summarise(random_baccs)
    inflation = r["mean"] - g["mean"]
    spread = max(g["std"], r["std"])

    RESULTS_JSON.write_text(json.dumps({
        "fingerprint": config_fingerprint(),
        "seeds": list(SEEDS),
        "split_rng_seed": SPLIT_RNG_SEED,
        "n_clips": len(records),
        "n_pitch_groups": n_groups,
        "classes": list(CLASSES),
        "grouped": {"leak_fraction": g_leak, **g},
        "random": {"leak_fraction": r_leak, **r},
        "inflation_balanced_accuracy": inflation,
        "seed_spread": spread,
        "exceeds_seed_spread": bool(abs(inflation) > spread),
    }, indent=2))

    print("=" * 68)
    print(f"grouped (leak {g_leak:.3f}): {g['mean']:.4f} +/- {g['std']:.4f}")
    print(f"random  (leak {r_leak:.3f}): {r['mean']:.4f} +/- {r['std']:.4f}")
    print(f"inflation from leaking the split: {inflation:+.4f} balanced accuracy")
    print(f"seed spread (max std): {spread:.4f}")
    if abs(inflation) <= spread:
        print("-> WITHIN the seed spread. Not a resolvable effect at 3 seeds; do not quote it.")
    elif inflation > 0:
        print(f"-> the leak inflates the score by {inflation:.4f} ({inflation / spread:.1f}x the spread).")
    else:
        print("-> random scored LOWER, which the pre-registered reading calls a probe bug. "
              "Investigate; do not report as a finding.")
    print("=" * 68)
    print(f"\nwrote {RESULTS_JSON}")
    print(f"probe checkpoints in {PROBE_DIR} (NOT the canonical outputs/model_s*.pt)")


if __name__ == "__main__":
    main()
