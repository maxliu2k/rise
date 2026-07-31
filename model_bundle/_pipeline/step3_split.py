"""Step 3 - Split BY PITCH GROUP, stratified by label (70/15/15).

Two ordering constraints, both load-bearing:

1. The split happens BEFORE any windowing. Every window later inherits its source's tag, so a
   source's windows can never straddle splits.
2. The unit of the split is the PITCH GROUP -- all files sharing (label, note) -- not the
   individual source file.

Why (2) matters. `violin_A4_15_forte_arco-normal.mp3` and `violin_A4_15_piano_arco-normal.mp3`
are different source files holding the same note at different dynamics. They are near-duplicates.
Splitting by file scatters them across train and test, so the model is scored on notes it has
effectively already seen and the number comes out inflated.

This is not hypothetical. The previous file-level split, measured on its own committed splits.csv:
406 of 436 pitch-groups (93.1%) spanned more than one split, 361 spanned all three, and 8036 of
8096 source files (99.3%) sat in a leaking group.

The cost has since been measured properly (cnn-ensemble, single/split_policy_probe.py: 3 seeds,
identical data and training, only the split policy varying):

    grouped (leak 0.000):  0.9600 +/- 0.0138
    random  (leak 0.967):  0.9957 +/- 0.0006     -> +0.0357, i.e. 2.6x the seed spread

Note the seed spread as much as the mean. The leaked arm varies by 0.0006 against 0.0138 -- a
23-fold collapse -- because a leaked split makes the task easy enough that initialisation stops
mattering. Near-perfect accuracy with almost no seed variance is the signature of memorising
near-duplicates, and it is a free diagnostic: be suspicious of an implausibly tight spread.

The old `assert df.groupby("path")["split"].nunique().max() == 1` never guarded anything: each
source is exactly one row, so it held by construction and could not detect the leak above. A check
that cannot fail is worse than no check, because it reads as reassurance. `verify_no_group_leak`
is the one that can actually fire.

Output: splits.csv (source_path, trimmed_path, label, split, is_phrase). Schema unchanged, so
Step 4 needs no modification.
"""
import random

import pandas as pd

from instrument_robustness.config import (MANIFEST_TRIMMED, SPLITS_CSV, SPLIT_FRACS, SEED,
                                          TARGET_LABELS, assert_artifact_fingerprint,
                                          write_artifact_fingerprint)

SPLIT_NAMES = ("train", "val", "test")


def group_key(label, note):
    """The unit of the split. Same instrument + same note = near-duplicate recordings."""
    return f"{label}_{note}"


def assign_groups(sizes, fracs, rng):
    """Assign whole pitch-groups to splits, targeting a share of FILES.

    Preconditions: `sizes` maps group key -> number of source files in it, all of one label;
    `fracs` maps split name -> target fraction, summing to 1.
    Postcondition: returns {group_key: split_name}, every key assigned exactly once.

    Groups have unequal sizes, so this targets file counts rather than group counts -- handing out
    equal numbers of groups would not produce a 70/15/15 split of clips. Largest groups are placed
    first so a large group cannot overshoot a nearly-full bin at the end; the shuffle breaks ties
    reproducibly given SEED.
    """
    total = sum(sizes.values())
    targets = {name: fracs[name] * total for name in fracs}
    filled = {name: 0 for name in fracs}
    keys = list(sizes)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -sizes[k])
    out = {}
    for k in keys:
        name = max(fracs, key=lambda s: targets[s] - filled[s])
        out[k] = name
        filled[name] += sizes[k]
    return out


def verify_no_group_leak(df):
    """Crash unless every pitch-group lives in exactly one split.

    Postcondition: returns the number of pitch-groups.
    Raises: AssertionError naming the offending groups.
    """
    spans = df.groupby("grp")["split"].nunique()
    bad = spans[spans > 1]
    assert bad.empty, (
        f"{len(bad)} pitch-group(s) span more than one split -- near-duplicate leak:\n"
        + "\n".join(f"    {k}" for k in list(bad.index)[:10])
        + ("\n    ..." if len(bad) > 10 else ""))
    return len(spans)


def main():
    assert_artifact_fingerprint(MANIFEST_TRIMMED, "step2_trim")
    df = pd.read_csv(MANIFEST_TRIMMED)
    missing = [c for c in ("path", "label", "note") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"ERROR: {MANIFEST_TRIMMED} lacks column(s) {missing}. The pitch-grouped split needs "
            f"`note` to build its group key. Rebuild the index with "
            f"`python -m instrument_robustness.prep_data`.")
    if df["note"].isna().any():
        raise SystemExit(
            f"ERROR: {int(df['note'].isna().sum())} row(s) in {MANIFEST_TRIMMED} have no `note`. "
            f"A row with no pitch cannot be grouped, and dropping it silently would bias the "
            f"split. Fix the index rather than skipping them.")

    fracs = dict(zip(SPLIT_NAMES, SPLIT_FRACS))
    df["grp"] = [group_key(lab, note) for lab, note in zip(df["label"], df["note"])]

    # Assign per class, so each class hits 70/15/15 independently (this is the stratification).
    rng = random.Random(SEED)
    tag = {}
    for label in sorted(df["label"].unique()):
        sizes = df[df["label"] == label].groupby("grp").size().to_dict()
        tag.update(assign_groups(sizes, fracs, rng))
    df["split"] = df["grp"].map(tag)

    # `note` is carried through so step 4 can re-run the leak check at WINDOW level. Without it,
    # the only downstream check possible was "each source's windows share a split", which is true
    # by construction and therefore guards nothing -- the same dead check this docstring warns
    # about, one stage later.
    out = df[["path", "trimmed_path", "label", "note", "split", "is_phrase"]].rename(
        columns={"path": "source_path"})
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPLITS_CSV, index=False)

    # --- verifications ---
    assert df["split"].notna().all(), "some source got no split tag"
    n_groups = verify_no_group_leak(df)
    print(f"leak check passed: {n_groups} pitch-groups, none spanning splits")

    counts = pd.crosstab(df["label"], df["split"])
    for name in SPLIT_NAMES:                    # a split with zero rows would drop its column
        if name not in counts.columns:
            counts[name] = 0
    counts = counts[list(SPLIT_NAMES)]
    print("\nper-class source-file counts per split:")
    print(counts.to_string())
    print("\ntotals:", df["split"].value_counts().to_dict())

    got, want = set(counts.index), set(TARGET_LABELS)
    assert got == want, f"label mismatch: missing {want - got}, unexpected {got - want}"
    assert (counts > 0).all().all(), "a class is absent from a split:\n" + counts.to_string()
    print(f"all {len(TARGET_LABELS)} classes present in train/val/test: True")

    achieved = {n: counts[n].sum() / len(df) for n in SPLIT_NAMES}
    print("achieved fractions: " + ", ".join(f"{n} {achieved[n]:.3f}" for n in SPLIT_NAMES)
          + f"  (target {', '.join(f'{f:.2f}' for f in SPLIT_FRACS)})")
    write_artifact_fingerprint(SPLITS_CSV, "step3_split")
    print(f"\nwrote {SPLITS_CSV}")


if __name__ == "__main__":
    main()
