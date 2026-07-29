"""Run the whole preprocessing pipeline as one command.

    python -m instrument_robustness.run_pipeline

Nine stages in order, stopping at the first failure. Takes about 8 minutes on a warm cache, or
~13 including the first download.

WHY THIS EXISTS. The stages are individually runnable and always were, but "run these nine
commands in this exact order, and export three thread variables first" is an instruction people
skim. The fingerprint chain already makes a WRONG order fail loudly — each stage asserts its
predecessor's sidecar — but nothing stopped someone stopping halfway, or forgetting the
environment, and a half-built data root looks much like a finished one from the outside.

The individual stages remain the debugging path. This is the documented one.

THREAD LIMITS ARE SET HERE, not left to the caller. Steps 6 and 7 fan out with
ProcessPoolExecutor, and every worker that imports numpy will otherwise start its own BLAS thread
pool, oversubscribing the machine badly. Stages run as subprocesses precisely so these take effect
— setting os.environ inside a process that has already imported numpy is too late.

RESUMING. On failure the runner prints the exact --from invocation to continue with, so a config
change late in the pipeline does not mean re-downloading the archives:

    python -m instrument_robustness.run_pipeline --from step4_window

Note that resuming is only safe when the earlier stages' outputs are still valid. Change anything
in config_fingerprint() and every prior artifact is stale by definition; the next stage will
refuse it rather than let you resume onto a mismatched base. That refusal is the point.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# Order is load-bearing, not alphabetical. Splitting must precede windowing so windows inherit
# their source's split; stats must precede featurization and are computed on TRAIN only.
STAGES = (
    "prep_data",
    "step0_filter",
    "step1_resample",
    "step2_trim",
    "step3_split",
    "step4_window",
    "step5_normalize",
    "step6_stats",
    "step7_featurize",
)

# One thread per worker process. The stages parallelise across files themselves.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}


def select(stages, start: str | None, end: str | None) -> tuple[str, ...]:
    """The contiguous slice of `stages` from `start` to `end` inclusive.

    Preconditions: start and end, if given, are stage names in `stages`.
    Postcondition: returns a non-empty tuple preserving pipeline order.
    Raises: SystemExit naming the valid stages if a name is unknown or the range is empty.
    """
    for name, value in (("--from", start), ("--to", end)):
        if value is not None and value not in stages:
            raise SystemExit(f"unknown stage {value!r} for {name}. Valid stages:\n  "
                             + "\n  ".join(stages))
    lo = stages.index(start) if start else 0
    hi = stages.index(end) if end else len(stages) - 1
    if lo > hi:
        raise SystemExit(f"--from {start} comes after --to {end}; nothing to run")
    return tuple(stages[lo:hi + 1])


def run_stage(stage: str, env: dict) -> tuple[int, float]:
    """Run one stage as a subprocess, streaming its output. Returns (exit code, seconds).

    A subprocess rather than an import so the thread limits actually apply, and so a stage that
    dies cannot take the runner's state with it.
    """
    t0 = time.perf_counter()
    rc = subprocess.call([sys.executable, "-u", "-m", f"instrument_robustness.{stage}"], env=env)
    return rc, time.perf_counter() - t0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the preprocessing pipeline end to end.",
        epilog="Stages: " + ", ".join(STAGES))
    p.add_argument("--from", dest="start", metavar="STAGE",
                   help="resume from this stage instead of the beginning")
    p.add_argument("--to", dest="end", metavar="STAGE", help="stop after this stage")
    p.add_argument("--list", action="store_true", help="list the stages in order and exit")
    p.add_argument("--dry-run", action="store_true", help="show what would run, run nothing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for i, s in enumerate(STAGES, 1):
            print(f"  {i}. {s}")
        return

    stages = select(STAGES, args.start, args.end)
    root = os.environ.get("RISE_DATA_ROOT", "<repo>/all-samples (default)")
    print(f"data root: {root}")
    print(f"stages   : {len(stages)} of {len(STAGES)}  ({stages[0]} -> {stages[-1]})\n")
    if args.dry_run:
        for s in stages:
            print(f"  would run: python -m instrument_robustness.{s}")
        return

    env = {**os.environ, **THREAD_ENV}
    timings: list[tuple[str, float]] = []
    total0 = time.perf_counter()

    for i, stage in enumerate(stages, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(stages)}] {stage}\n{'=' * 70}")
        rc, secs = run_stage(stage, env)
        timings.append((stage, secs))
        if rc != 0:
            print(f"\n{'!' * 70}")
            print(f"FAILED at {stage} (exit {rc}) after {secs:.0f}s. Later stages were NOT run.")
            print(f"Fix the cause, then resume with:")
            print(f"    python -m instrument_robustness.run_pipeline --from {stage}")
            print(f"{'!' * 70}")
            raise SystemExit(rc)

    total = time.perf_counter() - total0
    print(f"\n{'=' * 70}\nPIPELINE COMPLETE\n{'=' * 70}")
    for stage, secs in timings:
        print(f"  {stage:<18}{secs:>7.0f}s")
    print(f"  {'TOTAL':<18}{total:>7.0f}s")


if __name__ == "__main__":
    main()
