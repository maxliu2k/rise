"""Choose the SNR grid from evidence, on the VALIDATION split only.

    python -m instrument_robustness.snr_pilot --model svm
    python -m instrument_robustness.snr_pilot --model svm --noise white natural --limit 400
    python -m instrument_robustness.snr_pilot --model mert --limit 240 --device cuda

MEASURE MORE THAN ONE MODEL BEFORE FREEZING THE GRID. The SVM and the pretrained models fail in
different places: the SVM's 88 features are frame statistics standardized on clean data and collapse
at noise floors a pretrained encoder ignores. A grid tuned to either one alone risks the other
sitting at ceiling or at floor across the whole range, and a condition where every model scores the
same measures nothing.

WHY THIS EXISTS. The official grid (config.SNRS) was inherited rather than measured. If every model
is already near its floor at most of those levels, the sweep costs ~5.2 GB and a lot of compute to
report a row of near-chance numbers, and the interesting part of the degradation curve -- where
models separate -- sits above the highest level tested. This measures where that region actually is
before anything is materialized.

    for each candidate SNR s and noise type c:
        mix validation windows at s, score them, report macro-F1 and retention vs clean

THREE PROPERTIES THAT MAKE THE ANSWER TRANSFERABLE.

  1. VALIDATION ONLY. Test is never read. Choosing a grid is a design decision, and making it
     against test would convert test into a second validation set -- the grid is a knob like any
     hyperparameter.
  2. THE SAME MIXER. It calls noise_sweep.draw_noise and noise_sweep.mix_at_snr, not a local
     reimplementation, so a level that looks usable here behaves identically in the real sweep.
  3. NOTHING IS MATERIALIZED. Mixtures are made in memory and discarded. This writes no audio and
     cannot touch work/windows_noisy/, so it cannot contaminate or pre-empt a canonical sweep.

White noise needs no corpus, so `--noise white` runs without ESC-50; the ESC-50 categories are
skipped with a warning if the corpus is absent rather than failing the whole run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from instrument_robustness.config import (
    ARTIFACTS,
    NOISE_TYPES,
    ROOT,
    SNRS,
    STATS_NPZ,
    TARGET_LABELS,
    WINDOWS_CSV,
    assert_artifact_fingerprint,
    assert_fingerprint,
    config_fingerprint,
)

# Cover the frozen grid plus 70 dB, the rejected upper candidate where MERT was indistinguishable
# from clean. Keeping the measured extension makes future validation audits reproducible.
CANDIDATE_SNRS = [70, 60, 50, 40, 30, 20, 10, 0, -5, -10, -15]

# Retention = noisy macro-F1 / clean macro-F1. Outside this band a level carries little
# information: above it the model is barely affected, below it the model has collapsed and
# differences between models are floor artifacts rather than robustness.
USABLE_RETENTION_HIGH = 0.98
USABLE_RETENTION_LOW = 0.15


def validation_windows(limit: int | None, seed: int) -> pd.DataFrame:
    """The validation windows to pilot on, optionally subsampled evenly across classes.

    Postcondition: returns a frame with window_path and label, containing every label. Subsampling
    is per class so a smaller pilot does not silently drop the rarest instruments.
    Raises: ValueError if the validation split is empty or missing a label.
    """
    assert_artifact_fingerprint(WINDOWS_CSV, "step5_normalize")
    frame = pd.read_csv(WINDOWS_CSV)
    frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{WINDOWS_CSV} contains no validation windows")
    missing = sorted(set(TARGET_LABELS) - set(frame["label"]))
    if missing:
        raise ValueError(f"Validation split is missing labels: {missing}")
    if limit is not None and limit < len(frame):
        per_class = max(1, limit // len(TARGET_LABELS))
        frame = pd.concat(
            [
                group.sample(
                    n=min(per_class, len(group)),
                    random_state=seed,
                )
                for _, group in frame.groupby("label", sort=False)
            ],
            ignore_index=False,
        )
        frame = frame.sort_index().reset_index(drop=True)
    return frame


def warn_if_model_saw_validation(model_path: Path) -> None:
    """Warn when the supplied checkpoint was refit on train+val.

    `finalize_svm` and `finalize_mert` deliberately refit the selected configuration on train+val
    before the single test evaluation, and they name that artifact `final_model.joblib` /
    `final_probe.pt`. Scoring THAT model on validation windows scores it on its own training data:
    the clean baseline comes out near 1.0, every retention figure is measured against a memorized
    reference, and the usable band looks narrower than it is. The validation-selected checkpoint
    (`best_model.joblib` / `best_probe.pt`) is the one fit on train only.
    """
    stem = Path(model_path).name
    if stem.startswith("final_"):
        print(
            f"! {stem} is the train+val refit. Scoring it on validation windows scores it on its\n"
            f"  own training data -- the clean baseline will be optimistic and the retention curve\n"
            f"  measured against it. Prefer the validation-selected checkpoint "
            f"(best_{stem.split('_', 1)[1]})."
        )


def svm_scorer(model_path: Path, stats_path: Path):
    """Return a predict(waveforms) -> (N, n_classes) callable for the clean-trained SVM.

    Uses the same handcrafted features and the same saved train-only statistics as
    noise_eval_svm, so pilot scores are on the same scale as the real sweep's.
    """
    from instrument_robustness.featurelib import svm_vector
    from instrument_robustness.noise_eval_svm import load_training_statistics
    from instrument_robustness.svm_model import load_svm

    warn_if_model_saw_validation(model_path)
    model = load_svm(model_path)
    mean, std = load_training_statistics(stats_path)

    def predict(waveforms: list[np.ndarray]) -> np.ndarray:
        raw = np.vstack([svm_vector(w) for w in waveforms]).astype(np.float32, copy=False)
        return np.asarray(model.decision_function((raw - mean) / std), dtype=np.float64)

    return predict


def resolve_mert_embedding_schema(
    model_path: Path,
    checkpoint: dict[str, object],
) -> dict[str, str]:
    """Resolve the frozen backbone identity for a validation-selected MERT probe.

    New checkpoints store the schema directly. Older checkpoints store it only in the adjacent
    validation summary, so accept that record after verifying that it names this exact checkpoint.
    """
    schema = checkpoint.get("embedding_schema")
    if isinstance(schema, dict) and schema.get("model_id") and schema.get("model_revision"):
        return {
            "model_id": str(schema["model_id"]),
            "model_revision": str(schema["model_revision"]),
        }

    model_path = Path(model_path)
    summary_path = model_path.with_name("validation_summary.json")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"{model_path} lacks an immutable embedding schema and no readable "
            f"{summary_path.name} is available."
        ) from error
    if not isinstance(summary, dict):
        raise SystemExit(f"{summary_path} must contain a JSON object.")

    assert_fingerprint(summary.get("config_fingerprint"), str(summary_path))
    if summary.get("label_order") != TARGET_LABELS:
        raise SystemExit(f"{summary_path} uses an unexpected label order.")
    if summary.get("test_evaluated") is not False:
        raise SystemExit(f"{summary_path} is not a validation-only MERT selection.")

    digest = hashlib.sha256()
    with model_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    recorded_hash = (
        summary.get("output_files", {}).get("model", {}).get("sha256")
    )
    if recorded_hash != digest.hexdigest():
        raise SystemExit(
            f"{model_path} does not match the validation-selected model recorded by "
            f"{summary_path}."
        )

    schema = summary.get("embedding_schema")
    if not isinstance(schema, dict) or not schema.get("model_id") or not schema.get(
        "model_revision"
    ):
        raise SystemExit(
            f"{summary_path} lacks an immutable embedding schema "
            "(model_id + model_revision)."
        )
    return {
        "model_id": str(schema["model_id"]),
        "model_revision": str(schema["model_revision"]),
    }


def mert_scorer(model_path: Path, device: str = "auto"):
    """Return a predict(waveforms) -> (N, n_classes) callable for the frozen MERT probe.

    Reuses `extract_mert.extract_mert_batch` and the probe loader rather than reimplementing the
    24 kHz resample, the frozen backbone, or the per-layer time pooling -- the pilot must measure
    the same representation the real sweep will.

    Costs far more per window than the SVM: every level re-runs the 95M-parameter backbone, since
    the whole point is that the noisy waveform must travel the full path. Use --limit generously,
    and a GPU if you have one.
    """
    try:
        import torch

        from instrument_robustness.extract_mert import choose_device, extract_mert_batch
        from instrument_robustness.mert_probe import load_mert_probe
        from instrument_robustness.pretrained_extractors import (
            build_mert_model,
            build_mert_processor,
        )
    except ImportError as error:
        raise SystemExit(
            "The MERT pilot needs the optional extras: pip install -e '.[mert]'"
        ) from error

    warn_if_model_saw_validation(model_path)
    target_device = choose_device(device, torch)
    probe, checkpoint = load_mert_probe(model_path, device=target_device)

    # Pin the backbone to the revision the probe was trained against. A probe is a linear readout
    # of specific frozen features; run it on a different checkpoint's features and it still
    # produces confident, meaningless logits.
    schema = resolve_mert_embedding_schema(model_path, checkpoint)
    model_id = schema["model_id"]
    revision = schema["model_revision"]
    processor = build_mert_processor(model_id, revision)
    backbone = build_mert_model(model_id, revision)
    resolved = getattr(backbone.config, "_commit_hash", None) or revision
    if resolved != revision:
        raise SystemExit(
            f"MERT resolved revision {resolved!r}; the probe expects {revision!r}"
        )
    backbone.requires_grad_(False)
    backbone.eval().to(target_device)
    print(f"  MERT backbone {model_id} @ {revision[:12]} on {target_device}")

    def predict(waveforms: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(waveforms), batch_size):
            chunk = waveforms[start : start + batch_size]
            embeddings = extract_mert_batch(
                chunk,
                processor=processor,
                model=backbone,
                target_device=target_device,
                torch=torch,
            )
            with torch.inference_mode():
                logits = probe(
                    torch.from_numpy(embeddings).float().to(target_device)
                )
            batches.append(logits.float().cpu().numpy())
        return np.concatenate(batches, axis=0)

    return predict


def build_scorer(
    model: str,
    model_path: Path | None,
    stats_path: Path,
    device: str = "auto",
):
    """Resolve the requested model to a predict(waveforms) -> (N, n_classes) callable.

    Each scorer reuses that model's own adapter helpers, so the pilot and the real sweep cannot
    diverge on how a waveform becomes a prediction.

    Defaults point at the VALIDATION-SELECTED checkpoint (`best_*`), not the train+val refit
    (`final_*`): the pilot scores validation windows, and the refit was trained on them.

    Adding a model is a few lines -- CNN and CRNN would wire through
    `logmel_input.cnn_batch_from_waveforms` plus `ensemble_scores.combiner_scores`, and AST through
    `pretrained_extractors.ast_input`. Neither is here yet because neither has a current clean
    checkpoint to pilot.
    """
    if model == "svm":
        default = ARTIFACTS / "svm" / "best_model.joblib"
        return svm_scorer(model_path or default, stats_path)
    if model == "mert":
        default = ARTIFACTS / "mert" / "best_probe.pt"
        return mert_scorer(model_path or default, device)
    raise SystemExit(
        f"--model {model!r} is not wired into the pilot. Implemented: svm, mert. "
        "See build_scorer's docstring to add CNN, CRNN or AST."
    )


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = scores.argmax(axis=1)
    labels = range(len(TARGET_LABELS))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
    }


def recommend(
    rows: list[dict[str, object]],
    candidate_snrs: list[int] | None = None,
) -> dict[str, object]:
    """Turn the measured curve into a suggested grid.

    Preconditions: `candidate_snrs` is the grid actually measured. It must be passed whenever the
    caller overrode the default, because the shoulder widening steps through this list -- reading
    the module default instead would raise on any custom level.
    Postcondition: returns the usable SNR band per noise type and a suggested shared grid, or an
    explanation when no level in the candidate range is informative.

    The band is the levels whose retention sits inside [USABLE_RETENTION_LOW,
    USABLE_RETENTION_HIGH], widened by one candidate step on each side so the curve's shoulders are
    visible in the final results rather than inferred.
    """
    frame = pd.DataFrame(rows)
    candidates = sorted(
        set(candidate_snrs if candidate_snrs is not None else CANDIDATE_SNRS),
        reverse=True,
    )
    per_type: dict[str, object] = {}
    usable_all: set[int] = set()
    for noise_type, group in frame.groupby("noise_type"):
        group = group.sort_values("snr_db", ascending=False)
        inside = group.loc[
            group["retention"].between(USABLE_RETENTION_LOW, USABLE_RETENTION_HIGH),
            "snr_db",
        ].tolist()
        if inside:
            top, bottom = max(inside), min(inside)
            widened = [
                s
                for s in candidates
                if s <= _step_out(candidates, top, up=True)
                and s >= _step_out(candidates, bottom, up=False)
            ]
            per_type[str(noise_type)] = {
                "usable_snrs": sorted(inside, reverse=True),
                "band_with_shoulders": widened,
                "floor_at_or_below": (
                    min(group.loc[group["retention"] < USABLE_RETENTION_LOW, "snr_db"], default=None)
                ),
            }
            usable_all.update(widened)
        else:
            per_type[str(noise_type)] = {
                "usable_snrs": [],
                "note": (
                    "no candidate level landed between "
                    f"{USABLE_RETENTION_LOW} and {USABLE_RETENTION_HIGH} retention; "
                    "widen --snrs upward if everything collapsed, or downward if nothing moved"
                ),
            }
    return {
        "per_noise_type": per_type,
        "suggested_shared_grid": sorted(usable_all, reverse=True),
        "current_grid": list(SNRS),
    }


def _step_out(candidates: list[int], value: int, *, up: bool) -> int:
    """One candidate step above (or below) `value`, clamped to the candidate range."""
    ordered = sorted(candidates, reverse=True)
    index = ordered.index(value)
    target = index - 1 if up else index + 1
    target = min(max(target, 0), len(ordered) - 1)
    return ordered[target]


def run_pilot(
    *,
    model: str,
    model_path: Path | None,
    stats_path: Path,
    noise_types: list[str],
    snrs: list[int],
    limit: int | None,
    seed: int,
    output: Path | None,
    device: str = "auto",
) -> dict[str, object]:
    """Measure clean and noisy validation performance across the candidate grid."""
    from instrument_robustness.noise_sweep import (
        DEMAND_TARGETS,
        ESC50_TARGETS,
        dataset_build_identity,
        dataset_fingerprint,
        draw_noise,
        load_clean,
        load_demand_index,
        load_esc50_index,
        mix_at_snr,
        window_id_of,
        window_seed,
    )

    # Gate on what can be DRAWN, not on what is currently CONFIGURED.
    #
    # This used to check against config.NOISE_TYPES, which made the tool unusable for the one job
    # it exists to do: you cannot pilot a noise type until it is in NOISE_TYPES, and you must not
    # put it in NOISE_TYPES until it has been piloted. Every new type was locked out of its own
    # pilot. The right gate is whether draw_noise can produce it.
    drawable = {"white", *ESC50_TARGETS, *DEMAND_TARGETS}
    unknown = sorted(set(noise_types) - drawable)
    if unknown:
        raise SystemExit(
            f"Unknown noise type(s): {unknown}; draw_noise can produce {sorted(drawable)}"
        )

    frame = validation_windows(limit, seed)
    y_true = np.asarray(
        [TARGET_LABELS.index(label) for label in frame["label"]], dtype=np.int64
    )
    identity = dataset_build_identity()
    fingerprint = dataset_fingerprint(identity)
    predict = build_scorer(model, model_path, stats_path, device)

    esc_index: dict[str, list] = {}
    if any(t in ESC50_TARGETS for t in noise_types):
        try:
            esc_index = load_esc50_index()
        except (FileNotFoundError, ValueError) as error:
            skipped = [t for t in noise_types if t in ESC50_TARGETS]
            print(f"! ESC-50 unavailable ({error}); skipping {skipped}")
            noise_types = [t for t in noise_types if t not in ESC50_TARGETS]

    demand_index: dict[str, list] = {}
    if any(t in DEMAND_TARGETS for t in noise_types):
        try:
            demand_index = load_demand_index()
        except (FileNotFoundError, ValueError) as error:
            skipped = [t for t in noise_types if t in DEMAND_TARGETS]
            print(f"! DEMAND unavailable ({error}); skipping {skipped}")
            noise_types = [t for t in noise_types if t not in DEMAND_TARGETS]

    if not noise_types:
        raise SystemExit(
            "No noise types left to pilot. Set RISE_NOISE_ROOT / RISE_DEMAND_ROOT, "
            "or pass --noise white."
        )

    clean = [load_clean(str(p)) for p in frame["window_path"]]
    window_ids = [window_id_of(str(p)) for p in frame["window_path"]]
    clean_metrics = metrics(y_true, np.asarray(predict(clean)))
    clean_macro_f1 = clean_metrics["macro_f1"]
    print(
        f"\nmodel={model}  validation windows={len(frame)}  "
        f"clean macro-F1={clean_macro_f1:.4f}  accuracy={clean_metrics['accuracy']:.4f}"
    )
    if clean_macro_f1 <= 0:
        raise SystemExit("Clean macro-F1 is zero; fix the clean model before piloting noise.")
    if clean_macro_f1 >= 0.999:
        print(
            "! clean macro-F1 is essentially perfect, which usually means this checkpoint was\n"
            "  trained on these windows. Retention below is measured against a memorized baseline."
        )

    rows: list[dict[str, object]] = []
    for noise_type in noise_types:
        # One realization per window, reused across every level -- the same property the real
        # sweep relies on, so a difference between levels is intensity and not a fresh draw.
        realizations = [
            draw_noise(
                noise_type,
                np.random.default_rng(window_seed(wid, noise_type, fingerprint)),
                esc_index,
                demand_index=demand_index,
            )[0]
            for wid in window_ids
        ]
        print(f"\n{noise_type}")
        print(f"  {'SNR':>5}  {'macro-F1':>9}  {'accuracy':>9}  {'retention':>9}  {'drop':>8}")
        for snr in sorted(snrs, reverse=True):
            mixed = [
                mix_at_snr(c, n, snr)[0] for c, n in zip(clean, realizations)
            ]
            noisy_metrics = metrics(y_true, np.asarray(predict(mixed)))
            retention = noisy_metrics["macro_f1"] / clean_macro_f1
            row = {
                "noise_type": noise_type,
                "snr_db": int(snr),
                "macro_f1": noisy_metrics["macro_f1"],
                "accuracy": noisy_metrics["accuracy"],
                "retention": retention,
                "macro_f1_drop": clean_macro_f1 - noisy_metrics["macro_f1"],
                "in_current_grid": int(snr) in SNRS,
            }
            rows.append(row)
            marker = "  <- current grid" if row["in_current_grid"] else ""
            print(
                f"  {snr:>5}  {noisy_metrics['macro_f1']:>9.4f}  "
                f"{noisy_metrics['accuracy']:>9.4f}  {retention:>9.3f}  "
                f"{row['macro_f1_drop']:>8.4f}{marker}"
            )

    suggestion = recommend(rows, sorted(snrs, reverse=True))
    print("\nrecommendation")
    for noise_type, detail in suggestion["per_noise_type"].items():
        if detail.get("usable_snrs"):
            print(
                f"  {noise_type:<11} usable {detail['usable_snrs']} "
                f"-> with shoulders {detail['band_with_shoulders']}"
            )
        else:
            print(f"  {noise_type:<11} {detail['note']}")
    print(f"  suggested shared grid: {suggestion['suggested_shared_grid']}")
    print(f"  current config.SNRS:   {suggestion['current_grid']}")
    print(
        "\nThis pilot changes nothing. Edit config.SNRS yourself once you are satisfied, and "
        "remember a grid change invalidates any completed sweep."
    )

    report = {
        "model": model,
        "split": "val",
        "n_windows": int(len(frame)),
        "limit": limit,
        "seed": seed,
        "clean": clean_metrics,
        "candidate_snrs": sorted(snrs, reverse=True),
        "noise_types": noise_types,
        "retention_band": [USABLE_RETENTION_LOW, USABLE_RETENTION_HIGH],
        "rows": rows,
        "recommendation": suggestion,
        "dataset": identity,
        "config_fingerprint": config_fingerprint(),
    }
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {output}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick the SNR grid from validation evidence. Writes no audio."
    )
    parser.add_argument(
        "--model",
        default="svm",
        choices=("svm", "mert"),
        help="which clean-trained model to pilot",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="defaults to the VALIDATION-SELECTED checkpoint: artifacts/<model>/best_*",
    )
    parser.add_argument("--stats", type=Path, default=STATS_NPZ, help="SVM only")
    parser.add_argument("--noise", nargs="+", default=list(NOISE_TYPES), dest="noise_types")
    parser.add_argument("--snrs", type=int, nargs="+", default=list(CANDIDATE_SNRS))
    parser.add_argument(
        "--limit",
        type=int,
        default=360,
        help="windows to sample, spread evenly across classes; 0 uses the whole split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="MERT only; the SVM is CPU-bound either way",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must not be negative")
    # MERT re-runs a 95M-parameter backbone for every window at every level. A full-split pilot
    # over a dense grid is hours of GPU time, so refuse it rather than let it start silently.
    if args.model == "mert" and not args.limit:
        parser.error(
            "--limit 0 (whole split) with --model mert re-runs the backbone on every validation "
            "window at every SNR. Pass an explicit --limit, e.g. 240."
        )
    return args


def main() -> None:
    args = parse_args()
    run_pilot(
        model=args.model,
        model_path=args.model_path,
        stats_path=args.stats,
        noise_types=list(args.noise_types),
        snrs=list(args.snrs),
        limit=None if not args.limit else args.limit,
        seed=args.seed,
        output=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
