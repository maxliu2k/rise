"""Step 0 - Filter the manifest to the 9 target instruments and drop defective files.

- Keep only rows whose label is one of the 9 targets.
- Drop zero-byte / missing files (verified on disk, not just via inventory).
- duration_s is carried through and is the authoritative duration field (never `length`).
Output: manifest_9.csv
"""
import pandas as pd
from instrument_robustness.config import ROOT, MANIFEST_IN, MANIFEST_9, TARGET_LABELS

def main():
    df = pd.read_csv(MANIFEST_IN)
    n0 = len(df)

    df = df[df["label"].isin(TARGET_LABELS)].copy()
    n_label = len(df)

    # Verify each file exists and is non-empty on disk.
    sizes = df["path"].map(lambda p: (ROOT / p).stat().st_size if (ROOT / p).exists() else -1)
    bad = df[sizes <= 0]
    if len(bad):
        print(f"Dropping {len(bad)} zero-byte/missing file(s):")
        for p in bad["path"]:
            print(f"  - {p}")
    df = df[sizes > 0].copy()

    df = df.sort_values("path").reset_index(drop=True)
    df.to_csv(MANIFEST_9, index=False)

    print(f"\nrows in manifest.csv        : {n0}")
    print(f"rows after label filter     : {n_label}")
    print(f"rows after dropping defects : {len(df)}")
    print(f"\nper-class counts (is_phrase split):")
    tab = df.groupby(["label", "is_phrase"]).size().unstack(fill_value=0)
    tab.columns = [("note" if c == 0 else "phrase") for c in tab.columns]
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())
    print(f"\nwrote {MANIFEST_9}")

if __name__ == "__main__":
    main()
