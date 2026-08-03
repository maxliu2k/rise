"""Consolidate every model's clean and noise results into one comparison table.

    python -m instrument_robustness.summarize_results            # print
    python -m instrument_robustness.summarize_results --write     # also write docs/RESULTS.md

WHY THIS EXISTS. Each model writes results in its own schema and directory. Reading any single
file by hand is how a superseded result gets quoted instead of the canonical one. This reads only
sources it can verify against the CURRENT config_fingerprint() and marks
anything that does not match as STALE rather than printing its number as if it were live.

It is deliberately a thin reader over a small explicit registry, not a directory crawl: the point
is that every number in the table has a named, checked provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from instrument_robustness.config import ARTIFACTS, REPO_ROOT, config_fingerprint

# Each entry names WHERE the canonical clean test metric lives and HOW to reach the numbers inside
# it. `fp` is the dotted path to that file's recorded config_fingerprint; `test` is the dotted path
# to the block holding macro_f1/accuracy. A model with no trained checkpoint yet is listed with
# source=None so the table shows "pending" rather than omitting the row.
# `split` records which split the quoted macro_f1 is on. Every canonical row comes from the
# one-time finalizer's test_summary.json; validation-only runs remain pending here.
CLEAN_SOURCES = {
    name: dict(source=f"{name.lower()}/test_summary.json", fp="config_fingerprint", test="test_metrics", split="test")
    for name in ("AST", "SVM", "PANNs", "MERT", "CNN", "CRNN")
}


def _md_table(df):
    """Render a DataFrame as a GitHub markdown table without a tabulate dependency."""
    cols = list(df.columns)
    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(cell(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([head, sep, *body])


def _dig(obj, dotted):
    for key in dotted.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _n_examples(doc, test_block):
    for candidate in (doc.get("test_examples"), _dig(test_block, "n_examples")):
        if candidate:
            return int(candidate)
    cm = _dig(test_block, "confusion_matrix")
    return int(np.asarray(cm).sum()) if cm is not None else None


def clean_row(name, spec, current_fp):
    if spec.get("source") is None:
        return dict(model=name, split="—", macro_f1=None, accuracy=None, n=None, status="pending (not trained)")
    path = ARTIFACTS / spec["source"]
    if not path.is_file():
        return dict(model=name, split="—", macro_f1=None, accuracy=None, n=None, status=f"MISSING {spec['source']}")
    doc = json.loads(path.read_text())
    fp = _dig(doc, spec["fp"])
    block = _dig(doc, spec["test"]) or {}
    # Test blocks record macro_f1 explicitly. The CNN/CRNN 5-seed validation summary records its
    # headline under `mean` inside single_seed_val -- which, since the metric standardisation, IS
    # macro-F1 (train_cnn selects on it). Before that change `mean` was balanced accuracy, so a
    # pre-standardisation summary would put a balanced-accuracy number in a macro-F1 column with
    # no visible sign. Refuse rather than guess: the field says which metric it is.
    # THE SELECTION METRIC IS CHECKED FOR EVERY MODEL, not only the ones missing macro_f1.
    # It used to be checked only inside the `macro is None` branch below, which meant any result
    # carrying an explicit macro_f1 skipped the gate entirely -- a checkpoint selected on
    # balanced accuracy still printed `canonical` purely because its summary happened to record
    # a macro-F1 number too. That is exactly the silently-wrong-number shape this table exists to
    # stop, and AST sat in it for weeks.
    #
    # finalize_svm and finalize_mert only began propagating the field from the validation summary
    # after their one permitted test evaluation was already spent, so their existing
    # test_summary.json cannot carry it and cannot be regenerated without re-spending the test.
    # Fall back to the sibling validation_summary.json, which does record it. Nothing here guesses:
    # if neither file states the metric, the row is refused.
    metric = doc.get("selection_metric")
    if metric is None:
        sibling = path.parent / "validation_summary.json"
        if sibling.is_file():
            metric = json.loads(sibling.read_text()).get("selection_metric")
    if metric != "validation_macro_f1":
        why = (f"selected on {metric}" if metric is not None
               else "no selection_metric in test or validation summary")
        return dict(model=name, split=spec["split"], macro_f1=None, accuracy=None, n=None,
                    status=f"STALE ({why}); retrain")

    macro = _dig(block, "macro_f1")
    if macro is None:
        # No explicit macro_f1, so we are reading `mean` out of a CNN/CRNN 5-seed block. Since the
        # metric standardisation that IS macro-F1; the gate above has already established it.
        macro = _dig(block, "mean")
    stale = fp != current_fp
    status = "STALE (config mismatch)" if stale else "canonical"
    if not stale and spec["split"].startswith("val"):
        status = "canonical (val only, test not run)"
    return dict(
        model=name,
        split=spec["split"],
        macro_f1=macro,
        accuracy=_dig(block, "accuracy"),
        n=_n_examples(doc, block),
        status=status,
    )


def noise_summary(name):
    """Return a one-line noise summary for a model, or None if it has no sweep yet."""
    path = ARTIFACTS / name.lower() / "noise" / "noise_sweep_summary.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    clean = df[df.noise_type == "clean"]["macro_f1"]
    clean_f1 = float(clean.iloc[0]) if len(clean) else float("nan")
    noisy = df[df.noise_type != "clean"]
    # retention averaged over replicates, at the two anchor SNRs the abstract cares about
    out = {"model": name, "clean_f1": round(clean_f1, 4)}
    for snr in (20, 0):
        at = noisy[noisy.snr_db == snr]
        for nt in ("white", "natural", "mechanical"):
            v = at[at.noise_type == nt]["macro_f1_retention"]
            out[f"{nt[:4]}@{snr}"] = round(float(v.mean()), 3) if len(v) else None
    return out


def build():
    current_fp = config_fingerprint()
    clean = pd.DataFrame(clean_row(n, s, current_fp) for n, s in CLEAN_SOURCES.items())
    noise = [r for r in (noise_summary(n) for n in CLEAN_SOURCES) if r is not None]
    return clean, pd.DataFrame(noise)


def render(clean, noise):
    lines = ["# Model comparison — 12-class Philharmonia", ""]
    lines.append("Regenerate with `python -m instrument_robustness.summarize_results --write`. "
                 "Every clean number is verified against the current `config_fingerprint()`; a row "
                 "marked STALE was trained under a different config and must not be quoted.")
    lines += ["", "## Clean test baselines", "", _md_table(clean)]
    if len(noise):
        lines += ["", "## Noise robustness — macro-F1 retention vs clean (replicates averaged)",
                  "", _md_table(noise),
                  "", "Columns are retention at the named noise type and SNR (dB). "
                  "1.0 = no degradation; 0.083 macro-F1 is 12-class chance."]
    else:
        lines += ["", "_No noise sweeps found yet._"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write docs/RESULTS.md in addition to printing")
    args = ap.parse_args()
    clean, noise = build()
    text = render(clean, noise)
    print(text)
    if args.write:
        out = REPO_ROOT / "docs" / "RESULTS.md"
        # encoding is explicit: the table uses an em-dash for missing values, and write_text()
        # defaults to the platform encoding -- on Windows that silently rewrites the whole file as
        # cp1252, so regenerating on a different machine than last time changes every non-ASCII
        # byte and shows up as a spurious diff.
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
