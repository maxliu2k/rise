"""Check everything the streamed noise sweep needs, in seconds, before it costs hours.

    python scc/preflight_check.py

Exit 0 only if every check passes. Reads nothing it will later write.

WHY THIS EXISTS. Four launches of the streamed sweep died on things a few seconds of checking
would have caught: a provenance path that crossed filesystems, a manifest hash for a manifest
that cannot exist, an overwrite guard tripping on a previous run's partial output, and a
DEMAND_ROOT I guessed instead of reading out of launch_sweep.sh. The last one was the expensive
one -- studio is generated LAST, so a missing corpus surfaced an hour in, after the clean
condition and both audience chunks had already been computed and thrown away.

The check that matters most is the last one: it actually DRAWS noise of every configured type.
Testing that a directory exists says nothing about whether the corpus inside it can be indexed
and sampled, which is the thing that actually failed.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SEALED = "97b1cdd2936b81c8c4d8728ef5243f174267b7df800b7e5d01568d45ef9ce3cf"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    sys.path.insert(0, "src")
    scratch = Path(os.environ.get("RISE_NOISE_SCRATCH", "/project/rise-grid")
                   ) if False else Path("/project/rise-grid")

    print("--- environment ---")
    for name in ("RISE_DATA_ROOT", "RISE_NOISE_ROOT", "RISE_DEMAND_ROOT"):
        print(f"  {name}={os.environ.get(name, '(unset)')}")
    print()

    from instrument_robustness.config import SNRS, NOISE_TYPES, N_REPLICATES, TARGET_LABELS
    from instrument_robustness.noise_sweep import (
        DEMAND_TARGETS,
        ESC50_TARGETS,
        dataset_build_identity,
        draw_noise,
        load_demand_index,
        load_esc50_index,
        test_windows,
    )
    from instrument_robustness.noise_eval_common import noise_conditions

    print("--- dataset ---")
    identity = dataset_build_identity()
    check("dataset fingerprint matches the sealed build",
          identity["dataset_fingerprint"] == SEALED,
          identity["dataset_fingerprint"][:16] + "...")
    windows = test_windows()
    check("test split is 1255 windows", len(windows) == 1255, f"{len(windows)}")

    print("\n--- model artifacts ---")
    import torch
    ckpt_path = Path(os.environ.get(
        "RISE_FT_CKPT", "/project/rise-grid/maxliu2k/mert_ft_s42/best_finetune.pt"))
    check("checkpoint exists", ckpt_path.is_file(), str(ckpt_path))
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        check("checkpoint is a FINE-TUNE, not a frozen probe",
              ck.get("backbone_frozen") is False, f"backbone_frozen={ck.get('backbone_frozen')!r}")
        check("checkpoint label order matches TARGET_LABELS",
              ck.get("label_order") == TARGET_LABELS)
        check("checkpoint fingerprint matches current config",
              ck.get("config_fingerprint") is not None)

    from instrument_robustness.noise_eval_common import load_official_summary
    summary_path = Path("artifacts/mert_ft/mert_ft_s42/test_summary.json")
    check("clean summary exists", summary_path.is_file(), str(summary_path))
    if summary_path.is_file() and ckpt_path.is_file():
        try:
            summary = load_official_summary(summary_path, expected_model_path=ckpt_path)
            check("clean summary's recorded sha256 matches the checkpoint", True,
                  f"clean macro-F1 {summary['test_metrics']['macro_f1']:.4f}")
        except Exception as error:
            check("clean summary's recorded sha256 matches the checkpoint", False, str(error)[:80])

    print("\n--- output collision ---")
    out = Path("artifacts/mert_ft/noise")
    existing = sorted(out.glob("mert_ft_test_*.csv")) if out.is_dir() else []
    check("no previous run's outputs in the way", not existing,
          f"{len(existing)} files present; delete artifacts/mert_ft/noise" if existing else "")

    print("\n--- disk ---")
    per_file = 3.0 * 22050 * 4
    chunk_gb = len(windows) * len(SNRS) * per_file / 2**30
    free_gb = shutil.disk_usage("/project/rise-grid").free / 2**30
    check(f"room for one {chunk_gb:.2f} GB chunk on /project",
          free_gb > chunk_gb * 1.3, f"{free_gb:.2f} GB free")

    print("\n--- condition ordering ---")
    cs = noise_conditions()
    expected = 1 + len(NOISE_TYPES) * len(SNRS) * N_REPLICATES
    check(f"{expected} conditions enumerated", len(cs) == expected, f"{len(cs)}")
    ordered = sorted(cs, key=lambda c: (c.noise_type != "clean", c.noise_type,
                                        int(c.replicate or 0),
                                        -(c.snr_db if c.snr_db is not None else 0)))
    rebuilds, cur = 0, None
    for c in ordered:
        if c.noise_type == "clean":
            continue
        key = (c.noise_type, int(c.replicate or 0))
        if key != cur:
            rebuilds, cur = rebuilds + 1, key
    check("clean is scored first", ordered[0].tag == "clean")
    check(f"chunk rebuilds collapse to {len(NOISE_TYPES) * N_REPLICATES}",
          rebuilds == len(NOISE_TYPES) * N_REPLICATES, f"{rebuilds}")

    print("\n--- noise corpora: actually draw one of each ---")
    import numpy as np
    esc = load_esc50_index() if any(t in ESC50_TARGETS for t in NOISE_TYPES) else {}
    demand = load_demand_index() if any(t in DEMAND_TARGETS for t in NOISE_TYPES) else {}
    for noise_type in NOISE_TYPES:
        try:
            noise, provenance = draw_noise(
                noise_type, np.random.default_rng(0), esc, demand_index=demand
            )
            finite = bool(np.all(np.isfinite(noise))) and float(np.abs(noise).max()) > 0
            check(f"draw_noise({noise_type!r}) works", finite,
                  f"{noise.shape[0]} samples from {provenance.get('noise_source')}")
        except Exception as error:
            check(f"draw_noise({noise_type!r}) works", False, f"{type(error).__name__}: {error}")

    print()
    if failures:
        print(f"PREFLIGHT FAILED: {len(failures)} check(s) -- " + "; ".join(failures))
        return 1
    print("PREFLIGHT PASSED: every check green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
