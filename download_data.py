#!/usr/bin/env python3
"""
download_data.py — fetch the large data artifacts from Google Drive and unpack
them into the layout the `instrument_robustness` package expects.

The repo only contains CODE + small artifacts (splits.csv, windows.csv,
norm_stats.json). The large data (feature arrays + windowed audio) lives in
Google Drive and is pulled by this script.

USAGE
-----
    pip install gdown            # one-time, if not already installed
    python download_data.py      # run from the repo root

    # optional: skip the big audio if you're only training SVM/CNN/CRNN
    python download_data.py --features-only

WHERE THINGS LAND
-----------------
Everything is placed under DATA_ROOT (default: <repo>/all-samples), matching
config.py. Override with the RISE_DATA_ROOT env var if your data lives elsewhere.

    all-samples/
      features/            <- from features.zip   (SVM + CNN arrays, per split)
      work/windows/        <- from audio zip       (3s normalized windows; needed
                                                     only for AST/MERT/PANNs)

BEFORE YOU RUN
--------------
Paste your Drive share links (or bare file IDs) into the two variables below.
In Drive: right-click the zip -> Share -> "Anyone with the link" -> Copy link.
A link looks like:  https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
You can paste the whole link OR just the <FILE_ID> — both work.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

# ============================================================================
# 1) PASTE YOUR GOOGLE DRIVE LINKS / IDs HERE
# ============================================================================
FEATURES_ZIP_LINK = "https://drive.google.com/file/d/1Hrf3fp1D94lq5JexfauU_JiEOrVRHriW/view?usp=sharing"   # -> all-samples/features/
AUDIO_ZIP_LINK    = "https://drive.google.com/file/d/1jPML8itVgSCOkX2W34vBWb2b8DWDEd5j/view?usp=sharing"   # -> all-samples/work/windows/
# ============================================================================


# ---- Resolve DATA_ROOT the same way config.py does (no import needed) -------
def data_root() -> Path:
    repo = Path(__file__).resolve().parent
    return Path(os.environ.get("RISE_DATA_ROOT", repo / "all-samples")).resolve()


DATA_ROOT = data_root()
FEATURES_DIR = DATA_ROOT / "features"
WORK_DIR = DATA_ROOT / "work"
WINDOWS_DIR = WORK_DIR / "windows"


def extract_file_id(link_or_id: str) -> str | None:
    """Accept a full Drive URL or a bare file ID; return the file ID."""
    if not link_or_id or link_or_id.startswith("PASTE_"):
        return None
    # Already a bare ID (no slashes, no spaces)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", link_or_id):
        return link_or_id
    # Pattern: /d/<id>/  or  ?id=<id>
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", link_or_id)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", link_or_id)
    if m:
        return m.group(1)
    return None


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        sys.exit(
            "\n[!] gdown is not installed. Install it first:\n"
            "    pip install gdown\n"
        )


def download_zip(link_or_id: str, dest_zip: Path, label: str) -> bool:
    """Download a Drive file to dest_zip. Returns True on success, False if skipped."""
    file_id = extract_file_id(link_or_id)
    if file_id is None:
        print(f"[skip] {label}: no valid Drive link/ID set — skipping.")
        return False

    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[get ] {label}: downloading -> {dest_zip.name}")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    out = gdown.download(url, str(dest_zip), quiet=False)
    if out is None or not dest_zip.exists():
        print(
            f"[err ] {label}: download failed. Common causes:\n"
            "        - link not set to 'Anyone with the link'\n"
            "        - Drive 'virus scan' interstitial on very large files\n"
            f"        Try opening the link in a browser to confirm access, then re-run."
        )
        return False
    return True


def unzip_into(zip_path: Path, target_dir: Path, label: str):
    """Extract a zip into target_dir, creating it if needed."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {label}: extracting -> {target_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    print(f"[ ok  ] {label}: extracted.")


def _find_dir(root: Path, name: str, max_depth: int = 4) -> Path | None:
    """Find the first directory named `name` under root, up to max_depth deep."""
    root = root.resolve()
    for p in root.rglob(name):
        if p.is_dir():
            try:
                depth = len(p.relative_to(root).parts)
            except ValueError:
                continue
            if depth <= max_depth:
                return p
    return None


def normalize_layout(kind: str):
    """
    Make the extracted data match config.py regardless of how the zip was structured.

    kind == "features": ensure  all-samples/features/{svm,cnn}/...
    kind == "audio":    ensure  all-samples/work/windows/*.wav
    Moves misplaced folders into the correct location if needed.
    """
    import shutil

    if kind == "features":
        want = FEATURES_DIR
        # Already correct?
        if (want / "svm").exists() or (want / "cnn").exists():
            return
        # Zip may have opened straight to svm/ + cnn/ under DATA_ROOT, or nested one level.
        found_svm = _find_dir(DATA_ROOT, "svm")
        found_cnn = _find_dir(DATA_ROOT, "cnn")
        src_parent = None
        if found_svm is not None:
            src_parent = found_svm.parent
        elif found_cnn is not None:
            src_parent = found_cnn.parent
        if src_parent is not None and src_parent != want:
            want.mkdir(parents=True, exist_ok=True)
            for sub in ("svm", "cnn", "crnn", "ast", "mert", "panns"):
                s = src_parent / sub
                if s.exists() and not (want / sub).exists():
                    print(f"[fix ] features: moving {s} -> {want / sub}")
                    shutil.move(str(s), str(want / sub))

    elif kind == "audio":
        want = WINDOWS_DIR
        if want.exists() and any(want.rglob("*.wav")):
            return
        # Find a directory literally named "windows" anywhere reasonable.
        found = _find_dir(DATA_ROOT, "windows")
        if found is not None and found != want:
            want.parent.mkdir(parents=True, exist_ok=True)
            print(f"[fix ] audio: moving {found} -> {want}")
            import shutil as _sh
            _sh.move(str(found), str(want))
            return
        # Otherwise: the zip may have extracted loose .wav files under DATA_ROOT or a temp dir.
        loose = list(DATA_ROOT.glob("*.wav"))
        if loose:
            want.mkdir(parents=True, exist_ok=True)
            print(f"[fix ] audio: moving {len(loose)} loose .wav files -> {want}")
            for w in loose:
                w.rename(want / w.name)


def verify_features() -> bool:
    """Sanity-check that the expected feature arrays landed."""
    expected = [
        FEATURES_DIR / "svm" / "train.npz",
        FEATURES_DIR / "svm" / "val.npz",
        FEATURES_DIR / "svm" / "test.npz",
        FEATURES_DIR / "cnn" / "train.npz",
        FEATURES_DIR / "cnn" / "val.npz",
        FEATURES_DIR / "cnn" / "test.npz",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print("[warn] features: some expected files are missing:")
        for p in missing:
            print(f"         - {p.relative_to(DATA_ROOT)}")
        print(
            "       If your zip has a different internal folder structure, the files\n"
            "       may have landed one level too deep/shallow. Check all-samples/features/\n"
            "       and move them so they match config.FEATURES (all-samples/features/<model>/)."
        )
        return False
    print("[ ok  ] features: all SVM + CNN arrays present.")
    return True


def verify_audio() -> bool:
    if not WINDOWS_DIR.exists():
        print("[warn] audio: all-samples/work/windows/ not found after unzip.")
        return False
    n = sum(1 for _ in WINDOWS_DIR.rglob("*.wav"))
    if n == 0:
        print(
            "[warn] audio: no .wav windows found under work/windows/.\n"
            "       Your audio zip may unpack to a different subfolder — check work/\n"
            "       and make sure the windows end up at all-samples/work/windows/."
        )
        return False
    print(f"[ ok  ] audio: {n} window .wav files present.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Download project data from Google Drive.")
    ap.add_argument(
        "--features-only",
        action="store_true",
        help="Download only the feature arrays (skip the windowed audio). "
        "Enough for SVM / CNN / CRNN; NOT enough for AST / MERT / PANNs.",
    )
    ap.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep the downloaded .zip files instead of deleting them after extraction.",
    )
    args = ap.parse_args()

    ensure_gdown()

    print(f"[info] DATA_ROOT = {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    tmp = DATA_ROOT / "_downloads"
    tmp.mkdir(exist_ok=True)

    # --- Features ---
    feat_zip = tmp / "features.zip"
    if download_zip(FEATURES_ZIP_LINK, feat_zip, "features"):
        # Features belong under all-samples/ ; the zip is expected to contain a top-level
        # "features/" folder. Extract into DATA_ROOT so it lands at all-samples/features/.
        unzip_into(feat_zip, DATA_ROOT, "features")
        normalize_layout("features")
        if not args.keep_zips:
            feat_zip.unlink(missing_ok=True)
        verify_features()

    # --- Audio (windows) ---
    if not args.features_only:
        audio_zip = tmp / "audio.zip"
        if download_zip(AUDIO_ZIP_LINK, audio_zip, "audio"):
            # Audio windows belong at all-samples/work/windows/. If your zip already
            # contains a "work/windows/" tree, extract into DATA_ROOT; if it contains
            # just the window files, they'll be extracted under work/windows/ below.
            # Default assumption: zip has a top-level "work/" folder -> extract to DATA_ROOT.
            unzip_into(audio_zip, DATA_ROOT, "audio")
            normalize_layout("audio")
            if not args.keep_zips:
                audio_zip.unlink(missing_ok=True)
            verify_audio()
    else:
        print("[skip] audio: --features-only set, skipping windowed audio.")

    # cleanup temp dir if empty
    try:
        tmp.rmdir()
    except OSError:
        pass

    print(
        "\nDone. Next steps:\n"
        "  pip install -e .            # install the package (if you haven't)\n"
        "  # everyone trains on the SAME split defined in all-samples/pipeline/splits.csv\n"
        "  python -m instrument_robustness.step7_featurize   # only if you need to re-featurize\n"
    )


if __name__ == "__main__":
    main()
