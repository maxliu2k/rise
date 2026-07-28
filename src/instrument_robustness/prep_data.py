"""Acquire the canonical dataset and write manifest.csv -- the index every later step reads.

THIS IS THE ONLY SUPPORTED WAY TO OBTAIN THE DATASET.

    python -m instrument_robustness.prep_data

Do not copy a data tree from a teammate, and do not unpack a pre-derived feature or window
archive. Both were previously distributed via download_data.py (Google Drive), and both are
unsafe for the same reason: a derived artifact cannot prove which config produced it. A feature
array built under a different label set or window length still loads, still trains, and still
produces plausible numbers. Rebuilding takes minutes; discovering three weeks later that a result
came off the wrong data does not.

What this does:
  1. Downloads the 12 instrument archives from the Internet Archive mirror of the Philharmonia
     sample library (CC-BY-SA 4.0). The official philharmonia.co.uk URLs predate their site
     redesign and no longer resolve.
  2. Extracts into DATA_ROOT/<instrument>/<note>/<file>.mp3 -- the layout manifest.csv already
     uses, so inventory.py's folder/note cross-check still holds.
  3. Writes manifest.csv with every file and every articulation. Narrowing to the target labels
     and the plain articulation set is step0_filter's job, not this script's.
  4. Writes manifest_fingerprint.json so downstream artifacts can be checked against the config
     that produced this index.

Undecodable and unparseable files are COUNTED AND REPORTED, never silently dropped. A count that
quietly falls is indistinguishable from a dataset that quietly shrank.
"""
import csv
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter

from mutagen.mp3 import MP3

from instrument_robustness.config import (ARCHIVE_BASE, DATA_RAW, DATA_ROOT, FEATURES,
                                          MANIFEST_FINGERPRINT, MANIFEST_IN, PIPE,
                                          STRICT_ARTICULATIONS, TARGET_LABELS, WORK, ZIP_NAME,
                                          write_artifact_fingerprint)

# Philharmonia uses 's' for sharps (As4 = A#4). No flats appear in this set.
SEMITONES = {"C": 0, "Cs": 1, "D": 2, "Ds": 3, "E": 4, "F": 5,
             "Fs": 6, "G": 7, "Gs": 8, "A": 9, "As": 10, "B": 11}
NOTE_RE = re.compile(r"^([A-G]s?)([0-8])$")

FAMILY = {
    "violin": "strings", "viola": "strings", "cello": "strings", "double-bass": "strings",
    "flute": "woodwind", "clarinet": "woodwind", "oboe": "woodwind", "bassoon": "woodwind",
    "trumpet": "brass", "trombone": "brass", "tuba": "brass", "french-horn": "brass",
}

MANIFEST_COLUMNS = ["path", "label", "family", "duration_s", "sample_rate", "note", "midi",
                    "dynamic", "technique", "is_plain", "is_phrase"]


def zip_stem(inst):
    """Instrument key -> the archive's zip/dir name.

    They differ: zips use spaces where filenames use hyphens, and `cor anglais.zip` holds
    `english-horn_*.mp3`. Getting this wrong yields a 404, not a wrong result.
    """
    return ZIP_NAME.get(inst, inst)


def midi_number(note):
    """'A4' -> 69. Returns None if the note does not parse."""
    m = NOTE_RE.match(note)
    if not m:
        return None
    pitch, octave = m.group(1), int(m.group(2))
    return 12 * (octave + 1) + SEMITONES[pitch]


def download_and_extract(force=False):
    """Fetch and unpack every target instrument into DATA_ROOT/<instrument>/<note>/.

    Postcondition: every label in TARGET_LABELS has a directory under DATA_ROOT containing its
    mp3s, sorted into per-note folders.
    Raises: SystemExit if an archive yields no mp3s (a silent empty class would poison every
    downstream count).
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for inst in TARGET_LABELS:
        stem = zip_stem(inst)
        zip_path = DATA_RAW / f"{stem}.zip"
        dest = DATA_ROOT / inst
        if dest.exists() and not force:
            print(f"  {inst:14s} already extracted")
            continue
        if not zip_path.exists() or force:
            url = f"{ARCHIVE_BASE}/{stem.replace(' ', '%20')}.zip"
            print(f"  downloading {url}")
            urllib.request.urlretrieve(url, zip_path)

        staging = DATA_RAW / f"_stage_{inst}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging)

        found = [p for p in staging.rglob("*.mp3") if "__MACOSX" not in p.parts]
        if not found:
            sys.exit(f"ERROR: {zip_path.name} contained no mp3s -- extraction failed?")
        for src in found:
            fields = src.stem.split("_")
            note = fields[1] if len(fields) == 5 else "_unsorted"
            out_dir = dest / note
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(out_dir / src.name))
        shutil.rmtree(staging)
        print(f"  {inst:14s} extracted {len(found)} files")


def build_rows():
    """Walk the extracted tree and build one manifest row per mp3.

    Postcondition: returns (rows, problems). `problems` counts every file NOT represented in
    `rows`, by reason. rows + sum(problems.values()) == total mp3s seen.
    """
    rows, problems = [], Counter()
    for inst in TARGET_LABELS:
        root = DATA_ROOT / inst
        if not root.exists():
            sys.exit(f"ERROR: {root} missing -- run download_and_extract first.")
        files = sorted(p for p in root.rglob("*.mp3") if "__MACOSX" not in p.parts)
        if not files:
            sys.exit(f"ERROR: no mp3s under {root}.")
        for path in files:
            fields = path.stem.split("_")
            if len(fields) != 5:
                problems["bad_field_count"] += 1
                continue
            instrument, note, length, dynamic, technique = fields
            if instrument != inst:
                # a file in cello's archive not named cello_* -- surface it, do not trust it
                problems["instrument_mismatch"] += 1
                continue
            midi = midi_number(note)
            if midi is None:
                problems["unparseable_note"] += 1
                continue
            try:
                audio = MP3(str(path))
                duration_s = round(audio.info.length, 4)
                sample_rate = audio.info.sample_rate
            except Exception as e:                      # corrupt/0-byte file: count it, keep going
                problems["unreadable_audio"] += 1
                print(f"  ! unreadable: {path.name}: {e}", file=sys.stderr)
                continue
            rows.append({
                "path": path.relative_to(DATA_ROOT).as_posix(),
                "label": instrument,
                "family": FAMILY[instrument],
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "note": note,
                "midi": midi,
                "dynamic": dynamic,
                "technique": technique,
                "is_plain": int(technique in STRICT_ARTICULATIONS.get(instrument, set())),
                "is_phrase": int(length == "phrase"),
            })
    return rows, problems


def ensure_skeleton():
    """Create the directory layout every later step writes into.

    Steps 0-6 write into PIPE and WORK without creating them. That went unnoticed while the only
    data root in use was the committed all-samples/ tree, which already had pipeline/ present --
    against a FRESH root (which this script now produces) step0 died on a missing directory. The
    entry point that creates the root is the right place to establish its shape.
    """
    for d in (PIPE, WORK, FEATURES, DATA_RAW):
        d.mkdir(parents=True, exist_ok=True)


def main():
    print(f"data root: {DATA_ROOT}")
    print(f"{len(TARGET_LABELS)} classes: {', '.join(TARGET_LABELS)}\n")

    ensure_skeleton()
    print("acquiring archives ...")
    download_and_extract()

    print("\nbuilding manifest ...")
    rows, problems = build_rows()

    MANIFEST_IN.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_IN, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    write_artifact_fingerprint(
        MANIFEST_IN,
        "prep_data",
        fingerprint_path=MANIFEST_FINGERPRINT,
        metadata={"n_rows": len(rows), "excluded": dict(problems)},
    )

    per_class = Counter(r["label"] for r in rows)
    plain = Counter(r["label"] for r in rows if r["is_plain"])
    print(f"\n{'class':<14}{'total':>8}{'plain':>8}")
    for lab in TARGET_LABELS:
        print(f"{lab:<14}{per_class[lab]:>8}{plain[lab]:>8}")
    missing = [lab for lab in TARGET_LABELS if per_class[lab] == 0]
    assert not missing, f"no files for {missing} -- the manifest would silently omit them"

    total_seen = len(rows) + sum(problems.values())
    print(f"\nrows written: {len(rows)} of {total_seen} files seen")
    if problems:
        print("excluded (counted, not silently dropped):")
        for reason, n in problems.most_common():
            print(f"  {reason:24s} {n}")
    else:
        print("excluded: none")

    # A handful of bad files is a known property of the archive: viola_D6_05_piano_arco-normal.mp3
    # is corrupt upstream, and all-samples/manifest.py independently drops the same one. Do NOT
    # delete it -- it would just be re-downloaded, and the count is the record that it exists.
    # But a LARGE exclusion rate is a broken download or a missing decoder wearing the same
    # costume, and that must not pass as a normal run.
    excluded_frac = sum(problems.values()) / max(total_seen, 1)
    if excluded_frac > 0.01:
        raise SystemExit(
            f"ERROR: {sum(problems.values())} of {total_seen} files excluded "
            f"({excluded_frac:.1%}) -- above the 1% tolerance for known-bad archive files.\n"
            f"  {dict(problems)}\n"
            f"  This looks like a truncated download or a decoder problem, not archive rot. "
            f"Delete {DATA_RAW} and re-run rather than accepting a quietly smaller dataset.")

    ratio = max(per_class.values()) / max(min(per_class.values()), 1)
    print(f"\nclass imbalance (all articulations): {ratio:.2f}:1")
    print(f"\nwrote {MANIFEST_IN}")
    print(f"wrote {fp_path}")
    print("\nnext: python -m instrument_robustness.step0_filter")


if __name__ == "__main__":
    main()
