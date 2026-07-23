"""Download, inventory, filter, and cache the Philharmonia samples for config.CLASSES.

Run `python prep_data.py --inventory-only` to see the articulation histogram, the codec
check, and the strict-vs-sustained decision without processing any audio.

Owns the canonical audio -> spectrogram path (`load_chunks`, `wav_to_logmel`). noise_eval.py
imports these rather than reimplementing them: if the noise sweep ran through a different
spectrogram path than training did, it would be testing the wrong thing.
"""

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict

import librosa
import numpy as np
from tqdm import tqdm

from .config import (
    ARCHIVE_BASE, CLASSES, CLASS_TO_IDX, CLIP_SAMPLES, CLIP_SECONDS, DATA_RAW, FMAX, FMIN,
    ZIP_NAME,
    HOP_LENGTH, MANIFEST_JSON, MAX_CHUNKS_PER_FILE, MAX_IMBALANCE,
    MIN_FRAMES, MIN_STRICT_N, N_FFT, N_MELS, SEED, SPEC_DIR, SPLIT_FRACTIONS, SPLITS_JSON,
    SR, STRICT_ARTICULATIONS, SUSTAINED_ARTICULATIONS, TRIM_TOP_DB, WAVE_DIR,
)


# --------------------------------------------------------------------------- download

def zip_stem(inst):
    """Instrument key -> the archive's zip/dir name. They differ: zips use spaces where the
    filenames use hyphens, and `cor anglais.zip` holds `english-horn_*.mp3`."""
    return ZIP_NAME.get(inst, inst)


def download_and_extract(force=False):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for inst in CLASSES:
        stem = zip_stem(inst)
        zip_path = DATA_RAW / f"{stem}.zip"
        out_dir = DATA_RAW / stem
        if not zip_path.exists() or force:
            url = f"{ARCHIVE_BASE}/{stem.replace(' ', '%20')}.zip"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, zip_path)
        if not out_dir.exists() or force:
            print(f"extracting {zip_path.name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
    print()


def find_mp3s(inst):
    """All mp3s for an instrument, ignoring __MACOSX and other zip cruft."""
    root = DATA_RAW / zip_stem(inst)
    return sorted(p for p in root.rglob("*.mp3") if "__MACOSX" not in p.parts)


# --------------------------------------------------------------------------- parsing

def parse_filename(path):
    """`trumpet_A3_15_forte_normal.mp3` -> dict, or None if it doesn't fit the scheme.

    Articulations use hyphens internally (arco-col-legno-battuto), and durations may be
    `long`/`very-long`, so a clean 5-field underscore split is the whole contract.
    """
    parts = path.stem.split("_")
    if len(parts) != 5:
        return None
    instrument, note, duration, dynamics, articulation = parts
    if not instrument or not note or not articulation:
        return None
    return {
        "id": path.stem,
        "path": str(path),
        "instrument": instrument,
        "note": note,
        "duration": duration,
        "dynamics": dynamics,
        "articulation": articulation,
    }


def build_records():
    records, unparseable = [], []
    for inst in CLASSES:
        files = find_mp3s(inst)
        if not files:
            sys.exit(f"ERROR: no mp3s found under {DATA_RAW / inst} — extraction failed?")
        for p in files:
            rec = parse_filename(p)
            if rec is None:
                unparseable.append(p.name)
            elif rec["instrument"] != inst:
                # a file in cello.zip not named cello_* — surface it rather than trusting it
                unparseable.append(p.name)
            else:
                records.append(rec)
    return records, unparseable


# --------------------------------------------------------------------------- inventory

# Below this, the MP3 encoder's class-correlated spectral edge is out of band. Measured
# across all three bitrate groups: every codec brick wall sits above 19kHz, and the
# class-correlated spectral difference above ~14kHz. 14000 is the conservative bound.
CODEC_EDGE_HZ = 14000


def check_bitrates(records):
    """Are the classes encoded identically? If not, the encoder is partly a class label.

    Philharmonia encodes at three bitrates that cut across instrument families:
        64kbps: bassoon, clarinet, double-bass, percussion, trumpet
        80kbps: bass-clarinet, cor anglais, saxophone, trombone, tuba
        96kbps: banjo, cello, contrabassoon, flute, french-horn, guitar, mandolin, oboe,
                viola, violin
    MP3 lowpasses as a function of bitrate, so the encoder leaves a spectral edge that
    partitions the classes into 3 groups for free — nothing to do with the instruments.

    Measured: every codec brick wall is above 19kHz, and the class-correlated difference
    above ~14kHz (cello-vs-trumpet gap: +23.5dB at 15kHz, +30.2dB at 18kHz). At SR=22050 the
    Nyquist is 11025Hz and the resampler discards all of it. Verified no aliasing, and
    in-band the classes differ in the physically correct direction (trumpet brighter than
    cello, as brass should be against bowed strings).

    This check exists because that safety is a property of SR, not of the data. Raise SR
    toward 44100 and the encoder becomes visible and perfectly predictive of the group.
    """
    try:
        from mutagen.mp3 import MP3
    except ImportError:
        print("note: mutagen not installed — skipping the bitrate/codec check\n")
        return None

    by_class = defaultdict(Counter)
    for r in records:
        try:
            by_class[r["instrument"]][MP3(r["path"]).info.bitrate // 1000] += 1
        except Exception:
            continue
    if not by_class:
        return None

    print("=" * 72)
    print("CODEC CHECK")
    print("=" * 72)
    modal = {inst: c.most_common(1)[0][0] for inst, c in by_class.items()}
    groups = defaultdict(list)
    for inst, br in sorted(modal.items()):
        groups[br].append(inst)
    for br in sorted(groups):
        print(f"  {br:>3} kbps: {', '.join(groups[br])}")

    if len(groups) > 1:
        print(f"\n  WARNING: {len(groups)} bitrate groups across {len(modal)} classes.")
        print("  MP3 lowpasses by bitrate, so the encoder leaves a spectral edge that splits")
        print(f"  the classes into {len(groups)} groups for free, above ~{CODEC_EDGE_HZ // 1000}kHz.")
        if SR / 2 < CODEC_EDGE_HZ:
            print(f"  MITIGATED: SR={SR} -> Nyquist {SR // 2}Hz discards it before the model")
            print("  sees anything. Verified: no aliasing; in-band, classes differ in the")
            print("  physically expected direction.")
        else:
            print(f"  *** NOT MITIGATED at SR={SR} (Nyquist {SR // 2}Hz). The codec edge is")
            print(f"  *** INSIDE the analysis band and is a free {len(groups)}-way shortcut.")
            print(f"  *** Lower SR below {2 * CODEC_EDGE_HZ}, or lowpass all classes to a")
            print("  *** common cutoff, or pick classes from a single bitrate group.")
    else:
        print(f"\n  all classes encoded at {list(groups)[0]}kbps — no codec confound")
    print()
    return {k: int(v) for k, v in modal.items()}


def print_inventory(records, unparseable):
    print("=" * 72)
    print("INVENTORY (before filtering)")
    print("=" * 72)
    for inst in CLASSES:
        arts = Counter(r["articulation"] for r in records if r["instrument"] == inst)
        total = sum(arts.values())
        print(f"\n{inst}: {total} parseable clips, {len(arts)} articulations")
        for art, n in arts.most_common():
            print(f"    {art:<28} {n:>5}")
    if unparseable:
        print(f"\nskipped {len(unparseable)} unparseable filenames, e.g.:")
        for name in unparseable[:5]:
            print(f"    {name}")
    else:
        print("\nno unparseable filenames")
    print()


def choose_articulation_set(records):
    """Strict single-articulation if it leaves enough data, else widen to sustained family.

    Only small N widens the set. Class imbalance deliberately does NOT: widening is not a
    remedy for it (empirically it makes the ratio slightly worse here, since arco-normal
    already dominates cello more than normal dominates trumpet). Class weights in train.py
    are the remedy. Keeping strict also preserves the single-articulation comparison that
    is the whole point of filtering.
    """
    def counts_for(mapping):
        return {
            inst: sum(
                1 for r in records
                if r["instrument"] == inst and r["articulation"] in mapping[inst]
            )
            for inst in CLASSES
        }

    def ratio_of(c):
        return max(c.values()) / max(min(c.values()), 1)

    def show(label, c):
        r = ratio_of(c)
        lo_i = min(c, key=c.get)
        hi_i = max(c, key=c.get)
        print(f"  {label:<10} n={sum(c.values()):>5}  ratio {r:.2f}:1  "
              f"(min {lo_i} {c[lo_i]}, max {hi_i} {c[hi_i]})")
        for inst in CLASSES:
            print(f"      {inst:<14}{c[inst]:>5}  {'#' * int(40 * c[inst] / max(c.values()))}")
        return r

    strict = counts_for(STRICT_ARTICULATIONS)
    lo = min(strict.values())

    print("=" * 72)
    print("ARTICULATION GATE")
    print("=" * 72)
    ratio = show("strict", strict)

    if lo >= MIN_STRICT_N:
        print(f"\n-> STRICT (min {lo} >= {MIN_STRICT_N})")
        if ratio > MAX_IMBALANCE:
            print(f"   ratio {ratio:.2f}:1 > {MAX_IMBALANCE} -> train.py will apply class weights")
        print()
        return "strict", STRICT_ARTICULATIONS, strict

    sustained = counts_for(SUSTAINED_ARTICULATIONS)
    s_ratio = show("sustained", sustained)
    print(f"\n-> SUSTAINED FAMILY (strict min {lo} < {MIN_STRICT_N})")
    if s_ratio > MAX_IMBALANCE:
        print(f"   ratio {s_ratio:.2f}:1 > {MAX_IMBALANCE} -> train.py will apply class weights")
    print()
    return "sustained", SUSTAINED_ARTICULATIONS, sustained


# --------------------------------------------------------------------------- audio

class DecodeError(Exception):
    """A file that cannot be decoded at all — distinct from one that decodes to silence."""


def load_trimmed(path):
    """mp3 -> mono 22.05k with leading/trailing silence stripped. None if it's all silence.

    Only the edges are trimmed. Internal silence is left alone — measured, the phrase files
    contain none (gaps of 0.00-0.05s); they are continuous crescendos, not separated notes,
    so librosa.effects.split would gain nothing over trim here.

    Raises DecodeError on unreadable files. The archive ships at least two 0-byte MP3s
    (viola_D6_05_piano_arco-normal, saxophone_Fs3_15_fortissimo_normal); soundfile rejects
    them, librosa falls back to audioread, and audioread dies with EOFError. One corrupt
    file must not kill a 7,500-file run — but it must be counted, not silently dropped.
    """
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        raise DecodeError(f"{type(e).__name__}: {str(e)[:60]}") from e
    if y.size == 0:
        return None
    y_trim, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return None if y_trim.size == 0 else y_trim


def load_chunks(path):
    """mp3 -> list of variable-length waveforms. Nothing is padded or tiled.

    A note shorter than CLIP_SAMPLES is kept at its true length. A file longer than that is
    cut into chunks of exactly CLIP_SAMPLES, capped at MAX_CHUNKS_PER_FILE. Every returned
    sample is real recorded audio. Also returns (source_samples, samples_discarded).
    """
    y_trim = load_trimmed(path)   # may raise DecodeError; build_cache counts those
    if y_trim is None:
        return [], 0, 0
    src = int(y_trim.size)

    # too short to survive 3 MaxPool stages — would produce an empty time axis
    if 1 + src // HOP_LENGTH < MIN_FRAMES:
        return [], src, src

    if src < CLIP_SAMPLES:
        return [y_trim.astype(np.float32)], src, 0  # leave it be

    n = min(src // CLIP_SAMPLES, MAX_CHUNKS_PER_FILE)
    chunks = [y_trim[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES].astype(np.float32)
              for i in range(n)]
    return chunks, src, src - n * CLIP_SAMPLES


def wav_to_logmel(y):
    """Variable-length waveform -> per-spectrogram-standardized log-mel, shape (N_MELS, frames).

    pad_mode="reflect": center=True must pad n_fft//2 samples at each edge before framing.
    The librosa default is "constant" (zeros), which reinserts digital silence at the edges
    of every spectrogram — undercutting the pipeline's no-silence invariant and hitting short
    clips hardest (~1/3 of frames for a 0.26s clip). Reflecting the clip's own audio instead
    keeps the edge frames looking like the instrument, not a gap (edge deficit ~1dB -> ~0.3dB).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, pad_mode="reflect",
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return ((logmel - logmel.mean()) / (logmel.std() + 1e-8)).astype(np.float32)


def build_cache(records):
    WAVE_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    kept, dropped, too_short, undecodable = [], [], [], []
    total_src, total_discarded, capped = 0, 0, []

    for rec in tqdm(records, desc="caching", unit="file"):
        try:
            chunks, src, discarded = load_chunks(rec["path"])
        except DecodeError as e:
            undecodable.append((rec["id"], str(e)))
            continue
        if not chunks:
            (too_short if src else dropped).append(rec["id"])
            continue
        total_src += src
        total_discarded += discarded
        if src // CLIP_SAMPLES > MAX_CHUNKS_PER_FILE:
            capped.append((rec["id"], src // CLIP_SAMPLES))

        for i, y in enumerate(chunks):
            cid = f"{rec['id']}__c{i}" if len(chunks) > 1 else rec["id"]
            np.save(WAVE_DIR / f"{cid}.npy", y)
            np.save(SPEC_DIR / f"{cid}.npy", wav_to_logmel(y))
            kept.append(dict(rec, id=cid, parent_id=rec["id"], chunk_index=i,
                             n_chunks=len(chunks),
                             clip_seconds=round(y.size / SR, 4),
                             n_frames=1 + y.size // HOP_LENGTH,
                             source_seconds=round(src / SR, 4),
                             label=CLASS_TO_IDX[rec["instrument"]]))

    if undecodable:
        print(f"\n  {len(undecodable)} file(s) could not be decoded and were EXCLUDED "
              f"(corrupt in the archive):")
        for cid, err in undecodable[:5]:
            print(f"      {cid}  [{err}]")
    if dropped:
        print(f"dropped {len(dropped)} files that were silent after trimming")
    if too_short:
        print(f"dropped {len(too_short)} files under {MIN_FRAMES} frames "
              f"(~{MIN_FRAMES * HOP_LENGTH / SR:.2f}s): too short to survive 3 pooling stages")

    # Discarded audio is otherwise invisible: the code just runs and the data is gone.
    print(f"\nclips: variable length, nothing padded or tiled (cap {MAX_CHUNKS_PER_FILE} chunks/file)")
    print(f"    {len(records)} files -> {len(kept)} clips ({len(kept) / max(len(records), 1):.2f} per file)")
    multi = [r for r in kept if r["n_chunks"] > 1]
    if multi:
        parents = {r["parent_id"] for r in multi}
        print(f"    {len(parents)} files yielded >1 chunk, contributing {len(multi)} clips "
              f"({len(multi) / len(kept):.1%} of the set)")
    print(f"    audio discarded: {total_discarded / SR:.0f}s of {total_src / SR:.0f}s "
          f"({total_discarded / max(total_src, 1):.1%})")
    if capped:
        print(f"    {len(capped)} file(s) hit the {MAX_CHUNKS_PER_FILE}-chunk cap "
              f"(would have yielded up to {max(n for _, n in capped)}):")
        for cid, n in sorted(capped, key=lambda x: -x[1])[:3]:
            print(f"        {cid} -> {n} chunks available, kept {MAX_CHUNKS_PER_FILE}")
    return kept


def report_signal_stats(records):
    L = np.array([r["clip_seconds"] for r in records])
    F = np.array([r["n_frames"] for r in records])
    print(f"\nclip length (every sample is real audio — nothing padded or tiled):")
    print(f"    {L.min():.2f}s to {L.max():.2f}s | median {np.median(L):.2f}s")
    print(f"    {F.min()} to {F.max()} frames | {len(np.unique(F))} distinct lengths")
    print(f"    clips at the {CLIP_SECONDS}s cap: {(L >= CLIP_SECONDS - 1e-6).sum()} "
          f"({(L >= CLIP_SECONDS - 1e-6).mean():.1%})")

    # Clip length is now a visible property of each example — it is literally the width of
    # the spectrogram — rather than something padding normalised away. If it predicts the
    # class, the CNN could read width instead of timbre.
    per_class = {}
    print("\n    clip length by class:")
    for cls in CLASSES:
        c = np.array([r["clip_seconds"] for r in records if r["instrument"] == cls])
        per_class[cls] = {"median_s": float(np.median(c)), "mean_s": float(c.mean()),
                          "iqr_s": [float(np.percentile(c, 25)), float(np.percentile(c, 75))]}
        print(f"        {cls:<14} median {np.median(c):.3f}s | IQR "
              f"{np.percentile(c, 25):.2f}-{np.percentile(c, 75):.2f}s | n {c.size}")
    lift = length_confound_lift(records)
    print(f"\n        confound check: length-only classifier scores "
          f"{lift['balanced_accuracy']:.4f} balanced acc vs {lift['chance']:.4f} chance "
          f"(lift {lift['lift']:+.4f}, optimistic — scored on train)")
    print("        -> WARNING: clip length alone is predictive; the model may read width, "
          "not timbre" if lift["lift"] > 0.15
          else "        -> length is not a strong shortcut; distributions overlap")

    return {"clip_seconds": {"min": float(L.min()), "max": float(L.max()),
                             "median": float(np.median(L))},
            "n_frames": {"min": int(F.min()), "max": int(F.max()),
                         "distinct": int(len(np.unique(F)))},
            "clip_seconds_by_class": per_class,
            "length_confound": lift}


def length_confound_lift(records):
    """How well does clip length ALONE predict the class?

    A median-gap test is the wrong instrument: distributions can differ in median while
    overlapping so heavily that length carries almost no information. What matters is
    predictive lift — and measured in BALANCED accuracy, so chance is 1/n_classes whatever
    the imbalance, consistent with how everything else here is scored.

    Fits a shallow decision tree on the single `clip_seconds` feature. Deliberately weak: if
    even a strong learner can't beat chance on length alone, length is not a shortcut.
    """
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    L = np.array([r["clip_seconds"] for r in records]).reshape(-1, 1)
    y = np.array([r["label"] for r in records])
    tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                  random_state=SEED).fit(L, y)
    # scored on TRAIN — deliberately optimistic. An upper bound on what length could leak.
    bacc = float(balanced_accuracy_score(y, tree.predict(L)))
    chance = 1.0 / len(CLASSES)
    return {"balanced_accuracy": bacc, "chance": chance, "lift": bacc - chance}


# --------------------------------------------------------------------------- split

def grouped_split(records, rng):
    """70/15/15, stratified by class, with whole pitch-groups kept in one split.

    The same note at different dynamics/durations produces near-identical clips. A plain
    random split scatters them across train and test and inflates accuracy. Grouping by
    `{instrument}_{note}` makes the test set genuinely unseen pitches.
    """
    assignment = {}
    for cls in CLASSES:
        cls_recs = [r for r in records if r["instrument"] == cls]
        groups = defaultdict(list)
        for r in cls_recs:
            groups[f"{r['instrument']}_{r['note']}"].append(r["id"])

        keys = list(groups)
        rng.shuffle(keys)
        # largest-first greedy tracks the target fractions far better than arbitrary
        # order when groups differ in size; the shuffle breaks ties reproducibly.
        keys.sort(key=lambda k: len(groups[k]), reverse=True)

        total = len(cls_recs)
        targets = {s: total * f for s, f in SPLIT_FRACTIONS.items()}
        counts = {s: 0 for s in SPLIT_FRACTIONS}
        for k in keys:
            # assign to whichever split is furthest below its target, proportionally
            split = max(targets, key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1e-9))
            for cid in groups[k]:
                assignment[cid] = split
            counts[split] += len(groups[k])

    splits = defaultdict(list)
    for cid, split in assignment.items():
        splits[split].append(cid)
    return {s: sorted(splits[s]) for s in SPLIT_FRACTIONS}


def verify_no_group_leak(records, splits):
    """Two checks. Pitch-groups must not span splits, and — since chunking makes near-copies
    of one recording — neither may a file's chunks. The second is implied by the first
    (chunks inherit their parent's note), but it is asserted separately: it is the check
    that would catch a future change to the group key, and chunk leakage is the failure
    mode that would silently hand us an inflated test score."""
    by_id = {r["id"]: r for r in records}
    group_to_splits, parent_to_splits = defaultdict(set), defaultdict(set)
    for split, ids in splits.items():
        for cid in ids:
            r = by_id[cid]
            group_to_splits[f"{r['instrument']}_{r['note']}"].add(split)
            parent_to_splits[r["parent_id"]].add(split)

    leaked = {g: s for g, s in group_to_splits.items() if len(s) > 1}
    if leaked:
        sys.exit(f"ERROR: {len(leaked)} pitch-groups span multiple splits, e.g. {list(leaked)[:3]}")
    chunk_leaked = {p: s for p, s in parent_to_splits.items() if len(s) > 1}
    if chunk_leaked:
        sys.exit(f"ERROR: {len(chunk_leaked)} files have chunks in multiple splits, "
                 f"e.g. {list(chunk_leaked)[:3]}")
    print(f"\nleak check passed: {len(group_to_splits)} pitch-groups and "
          f"{len(parent_to_splits)} source files, none spanning splits")


def report_splits(records, splits):
    by_id = {r["id"]: r for r in records}
    print("\nsplit composition (clips per class):")
    w = max(9, max(len(c) for c in CLASSES) + 1)
    print(f"    {'class':<{w}}" + "".join(f"{s:>8}" for s in SPLIT_FRACTIONS) + f"{'total':>8}")
    for c in CLASSES:
        per = [sum(1 for i in splits[s] if by_id[i]["instrument"] == c) for s in SPLIT_FRACTIONS]
        print(f"    {c:<{w}}" + "".join(f"{n:>8}" for n in per) + f"{sum(per):>8}")
    tot = [len(splits[s]) for s in SPLIT_FRACTIONS]
    print(f"    {'ALL':<{w}}" + "".join(f"{n:>8}" for n in tot) + f"{sum(tot):>8}")
    frac = [n / sum(tot) for n in tot]
    print(f"    {'(frac)':<{w}}" + "".join(f"{f:>8.3f}" for f in frac))
    train_per = [sum(1 for i in splits["train"] if by_id[i]["instrument"] == c) for c in CLASSES]
    print(f"\n    train imbalance {max(train_per) / max(min(train_per), 1):.2f}:1 "
          f"(min {min(train_per)}, max {max(train_per)})")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-only", action="store_true",
                    help="print the articulation histogram and gate decision, then stop")
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = ap.parse_args()

    download_and_extract(force=args.force)
    records, unparseable = build_records()
    print_inventory(records, unparseable)
    bitrates = check_bitrates(records)
    mode, mapping, counts = choose_articulation_set(records)

    if args.inventory_only:
        return

    kept = [r for r in records if r["articulation"] in mapping[r["instrument"]]]
    print(f"filtering to {mode}: {len(kept)} clips of {len(records)}\n")

    cached = build_cache(kept)
    signal_stats = report_signal_stats(cached)

    rng = random.Random(SEED)
    splits = grouped_split(cached, rng)
    verify_no_group_leak(cached, splits)
    report_splits(cached, splits)

    SPLITS_JSON.write_text(json.dumps(splits, indent=2))
    MANIFEST_JSON.write_text(json.dumps({
        "articulation_mode": mode,
        "articulations": {k: sorted(v) for k, v in mapping.items()},
        "class_counts": counts,
        "seed": SEED,
        "sample_rate": SR,
        "bitrate_kbps_by_class": bitrates,
        "variable_length": True,
        "max_clip_seconds": CLIP_SECONDS,
        "max_chunks_per_file": MAX_CHUNKS_PER_FILE,
        "n_source_files": len({r["parent_id"] for r in cached}),
        "signal_stats": signal_stats,
        "n_unparseable": len(unparseable),
        "records": cached,
    }, indent=2))
    print(f"\nwrote {SPLITS_JSON}")
    print(f"wrote {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
