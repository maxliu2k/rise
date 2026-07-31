"""Step 0 - Filter the manifest to the 12 target instruments, one articulation each, and drop
defective files.

- Keep only rows whose label is one of the 12 targets.
- Keep only the ONE plain articulation per class (config.STRICT_ARTICULATIONS): `normal` for
  winds and brass, `arco-normal` for bowed strings.
- Drop zero-byte / missing files (verified on disk, not just via inventory).
- duration_s is carried through and is the authoritative duration field (never `length`).
Output: manifest_labeled.csv

Why the articulation filter exists. Playing technique is class-correlated: only bowed strings have
`arco-*` and `pizz-*`, only brass have the tonguing variants, only strings have `col-legno`. Left
in, the technique partially identifies the family for free and the model can score well without
learning timbre.

It also skews class balance badly, because the extra articulations are not spread evenly. Measured
on the 12-class manifest: 1818 of 10196 rows are non-plain, but 650 of those are violin and only 52
are trumpet -- violin nearly doubles (852 -> 1502) while trumpet barely moves. Imbalance goes from
1.97:1 to 3.10:1 purely from which instruments happen to have more recorded techniques.

Filtering costs little: `normal`/`arco-normal` is 82% of the archive, not a slice of it.
"""
import pandas as pd

from instrument_robustness.config import (
    CONFLICTING_LABEL_PATHS,
    MANIFEST_FINGERPRINT,
    MANIFEST_IN,
    MANIFEST_LABELED,
    MANIFEST_PRODUCER_STAGES,
    ROOT,
    STRICT_ARTICULATIONS,
    TARGET_LABELS,
    assert_artifact_fingerprint,
    write_artifact_fingerprint,
)


def exclude_conflicting_labels(df, expected_paths=CONFLICTING_LABEL_PATHS):
    conflicts = df[df["path"].isin(expected_paths)]
    found = set(conflicts["path"])
    missing = set(expected_paths) - found
    if missing:
        raise RuntimeError(
            "Expected conflicting-label source(s) are absent from manifest.csv: "
            + ", ".join(sorted(missing))
        )
    return df[~df["path"].isin(expected_paths)].copy(), sorted(found)

def main():
    assert_artifact_fingerprint(
        MANIFEST_IN,
        MANIFEST_PRODUCER_STAGES,
        fingerprint_path=MANIFEST_FINGERPRINT,
    )
    df = pd.read_csv(MANIFEST_IN)
    n0 = len(df)

    df = df[df["label"].isin(TARGET_LABELS)].copy()
    n_label = len(df)

    keep = [t in STRICT_ARTICULATIONS.get(lab, set())
            for lab, t in zip(df["label"], df["technique"])]
    dropped = df[[not k for k in keep]]
    if len(dropped):
        by_class = dropped.groupby("label").size()
        print(f"dropping {len(dropped)} non-plain rows (technique not in STRICT_ARTICULATIONS):")
        for lab, n in by_class.sort_values(ascending=False).items():
            print(f"  {lab:<14}{n:>5}")
    df = df[keep].copy()
    n_plain = len(df)

    df, excluded_conflicts = exclude_conflicting_labels(df)
    print(f"excluding {len(excluded_conflicts)} byte-identical, conflicting-label source files:")
    for path in excluded_conflicts:
        print(f"  - {path}")

    # Verify each file exists and is non-empty on disk.
    sizes = df["path"].map(lambda p: (ROOT / p).stat().st_size if (ROOT / p).exists() else -1)
    bad = df[sizes <= 0]
    if len(bad):
        print(f"Dropping {len(bad)} zero-byte/missing file(s):")
        for p in bad["path"]:
            print(f"  - {p}")
    df = df[sizes > 0].copy()

    df = df.sort_values("path").reset_index(drop=True)
    MANIFEST_LABELED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_LABELED, index=False)

    print(f"\nrows in manifest.csv          : {n0}")
    print(f"rows after label filter       : {n_label}")
    print(f"rows after articulation filter: {n_plain}")
    print(f"rows after conflict exclusion   : {len(df)}")
    print(f"rows after dropping defects   : {len(df)}")
    print(f"\nper-class counts (is_phrase split):")
    tab = df.groupby(["label", "is_phrase"]).size().unstack(fill_value=0)
    tab.columns = [("note" if c == 0 else "phrase") for c in tab.columns]
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())

    counts = tab["total"]
    ratio = counts.max() / max(counts.min(), 1)
    print(f"\nclass imbalance: {ratio:.2f}:1 "
          f"(min {counts.idxmin()} {counts.min()}, max {counts.idxmax()} {counts.max()})")

    # One articulation per class is the whole point of the filter above; if a class somehow kept
    # more than one, the technique shortcut is back and every downstream number is suspect.
    per_class_techs = df.groupby("label")["technique"].nunique()
    bad = per_class_techs[per_class_techs > 1]
    assert bad.empty, f"class(es) kept >1 articulation: {bad.to_dict()}"
    missing = set(TARGET_LABELS) - set(df["label"])
    assert not missing, f"no rows survived for {sorted(missing)} -- check STRICT_ARTICULATIONS"

    write_artifact_fingerprint(MANIFEST_LABELED, "step0_filter")
    print(f"\nwrote {MANIFEST_LABELED}")

if __name__ == "__main__":
    main()
