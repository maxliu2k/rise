#!/usr/bin/env python3
"""Step 2: collapse note folders into per-instrument pools.

This is a *labeling* step, not a filesystem step. The note subfolders are a
storage detail; every fact needed for modeling already lives in the filename
and in inventory.csv. So "pooling all violin recordings regardless of pitch"
is just: select rows where instrument == 'violin'. No files move.

Emits manifest.csv -- the training index: one row per usable sample,
with the class label and the columns a grouped train/test split needs.
"""

import csv
import os
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
INVENTORY = os.path.join(ROOT, "inventory.csv")
OUT = os.path.join(ROOT, "manifest.csv")

# 'normal' (winds/brass) and 'arco-normal' (strings) name the same idea:
# the instrument's default playing technique. Unify so technique can be
# reasoned about across families.
PLAIN = {"normal", "arco-normal"}

rows = list(csv.DictReader(open(INVENTORY)))

kept, dropped = [], []
for r in rows:
    # Drop unreadable/empty audio -- these crash loaders mid-epoch.
    if not r["duration_s"] or float(r["duration_s"]) <= 0.0:
        dropped.append((r["path"], "zero-length or unreadable"))
        continue
    kept.append({
        "path": r["path"],
        "label": r["instrument"],      # <- the class. Pitch is NOT the class.
        "family": r["family"],
        "duration_s": r["duration_s"],
        "sample_rate": r["sample_rate"],
        # Carried for splitting/analysis, not as model inputs:
        "note": r["note"],
        "midi": r["midi"],
        "dynamic": r["dynamic"],
        "technique": r["technique"],
        "is_plain": int(r["technique"] in PLAIN),
        "is_phrase": int(r["length"] == "phrase"),
    })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
    w.writeheader()
    w.writerows(kept)

labels = Counter(r["label"] for r in kept)
print(f"pooled {len(kept)} samples into {len(labels)} classes -> manifest.csv")
print(f"dropped {len(dropped)}:")
for p, why in dropped:
    print(f"  - {p} ({why})")
print()
print(f"{'label':<12}{'n':>6}{'plain':>8}{'phrase':>8}{'notes':>7}")
for lab, n in labels.most_common():
    sub = [r for r in kept if r["label"] == lab]
    print(f"{lab:<12}{n:>6}{sum(r['is_plain'] for r in sub):>8}"
          f"{sum(r['is_phrase'] for r in sub):>8}{len({r['note'] for r in sub}):>7}")
