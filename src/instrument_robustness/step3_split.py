"""Step 3 - Split BY SOURCE FILE, stratified by label (70/15/15).

This is the single most important ordering constraint: the split happens at the level of the
original source recording, BEFORE any windowing. Every window later inherits its source's tag.
Output: splits.csv (source_path -> split). Verify: no source in two splits; all 9 classes present
in all three splits.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from instrument_robustness.config import PIPE, SPLIT_FRACS, SEED, TARGET_LABELS

def main():
    df = pd.read_csv(PIPE / "manifest_9_trimmed.csv")
    # one row == one source recording; splitting rows == splitting sources
    tr_frac, va_frac, te_frac = SPLIT_FRACS

    train, temp = train_test_split(
        df, train_size=tr_frac, stratify=df["label"], random_state=SEED)
    # split temp into val/test proportionally
    va_rel = va_frac / (va_frac + te_frac)
    val, test = train_test_split(
        temp, train_size=va_rel, stratify=temp["label"], random_state=SEED)

    tag = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        for p in part["path"]:
            tag[p] = name
    df["split"] = df["path"].map(tag)

    out = df[["path", "trimmed_path", "label", "split", "is_phrase"]].rename(
        columns={"path": "source_path"})
    out.to_csv(PIPE / "splits.csv", index=False)

    # --- verifications ---
    assert df["split"].notna().all(), "some source got no split tag"
    counts = pd.crosstab(df["label"], df["split"])[["train", "val", "test"]]
    print("per-class source-file counts per split:")
    print(counts.to_string())
    print("\ntotals:", df["split"].value_counts().to_dict())
    all_present = (counts > 0).all().all() and set(counts.index) == set(TARGET_LABELS)
    print("all 9 classes present in train/val/test:", all_present)
    # no leakage possible: each source appears in exactly one row -> one split
    assert df.groupby("path")["split"].nunique().max() == 1, "a source appears in >1 split"
    print("no source file appears in more than one split: True")
    print(f"\nwrote {PIPE / 'splits.csv'}")

if __name__ == "__main__":
    main()
