"""Build a TinySOL manifest.csv so the existing pipeline can run on TinySOL.

TinySOL is WAV, isolated sustained notes (no phrases, no bitrate confound). This emits the SAME
manifest schema + canonical label strings as the Philharmonia build, so the two datasets are
directly comparable. Run with RISE_DATA_ROOT pointed at the TinySOL folder and RISE_TARGET_LABELS
set to the desired class set.

    RISE_DATA_ROOT=/Users/gavin/Downloads/TinySOL2020 \
    RISE_TARGET_LABELS="violin,viola,cello,double bass,flute,clarinet,oboe,bassoon,trumpet,french horn,trombone,tuba" \
    python -m instrument_robustness.build_tinysol_manifest
"""
import warnings
import pandas as pd, soundfile as sf
from instrument_robustness.config import ROOT, MANIFEST_IN, TARGET_LABELS
warnings.filterwarnings("ignore")

# TinySOL instrument-folder name -> canonical label (matching the Philharmonia build)
FOLDER2LABEL = {
    "Violin": "violin", "Viola": "viola", "Violoncello": "cello", "Contrabass": "double bass",
    "Flute": "flute", "Clarinet_Bb": "clarinet", "Oboe": "oboe", "Bassoon": "bassoon",
    "Trumpet_C": "trumpet", "Horn": "french horn", "Trombone": "trombone", "Bass_Tuba": "tuba",
}
FAMILY = {
    "violin": "strings", "viola": "strings", "cello": "strings", "double bass": "strings",
    "flute": "woodwind", "clarinet": "woodwind", "oboe": "woodwind", "bassoon": "woodwind",
    "trumpet": "brass", "french horn": "brass", "trombone": "brass", "tuba": "brass",
}


def main():
    rows = []
    for wav in sorted(ROOT.rglob("*.wav")):
        if "_removed" in wav.parts:
            continue
        inst = next((p for p in wav.parts if p in FOLDER2LABEL), None)
        if inst is None:
            continue
        label = FOLDER2LABEL[inst]
        if label not in TARGET_LABELS:          # respect the requested class set
            continue
        info = sf.info(str(wav))
        rows.append({
            "path": str(wav.relative_to(ROOT)),
            "label": label,
            "family": FAMILY[label],
            "duration_s": round(info.frames / info.samplerate, 4),
            "sample_rate": info.samplerate,
            "is_plain": 1,
            "is_phrase": 0,                      # TinySOL = isolated notes only
        })
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df.to_csv(MANIFEST_IN, index=False)
    print(f"wrote {MANIFEST_IN}  ({len(df)} files, {df['label'].nunique()} classes)")
    print("\nper-class counts:")
    print(df.groupby("label").size().reindex(TARGET_LABELS).to_string())
    print(f"\nduration: median={df.duration_s.median():.2f}s  "
          f">=3s={ (df.duration_s>=3).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
