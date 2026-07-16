#!/usr/bin/env python3
"""Build an inventory table of every audio sample in the all-samples tree.

Emits inventory.csv with one row per .mp3, combining metadata parsed from the
filename with real values read from the MP3 headers.

Philharmonia filenames are uniformly 5 underscore-separated fields:
    <instrument>_<note>_<length>_<dynamic>_<technique>.mp3
e.g. violin_A4_025_forte_arco-normal.mp3

Requires: mutagen
"""

import csv
import os
import re
import sys
from collections import Counter

from mutagen.mp3 import MP3

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "inventory.csv")

# Philharmonia uses 's' for sharps (As4 = A#4). No flats appear in this set.
SEMITONES = {"C": 0, "Cs": 1, "D": 2, "Ds": 3, "E": 4, "F": 5,
             "Fs": 6, "G": 7, "Gs": 8, "A": 9, "As": 10, "B": 11}
NOTE_RE = re.compile(r"^([A-G]s?)([0-8])$")

# Everything under Strings/ is a string instrument; the rest are top-level.
FAMILY = {"violin": "strings", "viola": "strings", "cello": "strings",
          "flute": "woodwind", "clarinet": "woodwind", "bassoon": "woodwind",
          "trumpet": "brass", "trombone": "brass", "tuba": "brass"}


def midi_number(note):
    """'A4' -> 69. Returns None if the note doesn't parse."""
    m = NOTE_RE.match(note)
    if not m:
        return None
    pitch, octave = m.group(1), int(m.group(2))
    return 12 * (octave + 1) + SEMITONES[pitch]


def main():
    rows = []
    problems = Counter()

    for dirpath, _, filenames in os.walk(ROOT):
        for fn in filenames:
            if not fn.lower().endswith(".mp3"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)

            fields = fn[:-4].split("_")
            if len(fields) != 5:
                problems["bad_field_count"] += 1
                continue
            instrument, note, length, dynamic, technique = fields

            # The note folder each file now lives in, to cross-check the sort.
            folder_note = os.path.basename(dirpath)
            if folder_note != note:
                problems["folder_note_mismatch"] += 1

            midi = midi_number(note)
            if midi is None:
                problems["unparseable_note"] += 1

            try:
                audio = MP3(path)
                duration = round(audio.info.length, 4)
                sample_rate = audio.info.sample_rate
                channels = audio.info.channels
                bitrate = audio.info.bitrate
            except Exception as e:  # unreadable/corrupt file
                problems["unreadable_audio"] += 1
                duration = sample_rate = channels = bitrate = None
                print(f"  ! unreadable: {rel}: {e}", file=sys.stderr)

            rows.append({
                "instrument": instrument,
                "family": FAMILY.get(instrument, "unknown"),
                "note": note,
                "midi": midi,
                "octave": int(note[-1]) if midi is not None else None,
                "length": length,          # 025 / 05 / 1 / 15 / long / very-long / phrase
                "dynamic": dynamic,
                "technique": technique,
                "duration_s": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "bitrate": bitrate,
                "bytes": os.path.getsize(path),
                "folder_note": folder_note,
                "path": rel,
                "filename": fn,
            })

    rows.sort(key=lambda r: (r["instrument"], r["midi"] if r["midi"] is not None else -1,
                             r["filename"]))

    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {os.path.relpath(OUT, ROOT)}")
    if problems:
        print("problems:", dict(problems))
    else:
        print("problems: none")


if __name__ == "__main__":
    main()
