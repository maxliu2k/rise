"""The single permitted test evaluation for the CNN ensemble.

    python -m instrument_robustness.finalize_cnn

Mirrors finalize_svm and finalize_mert. Nothing is tuned or selected here: the combiner was chosen
on validation by train_cnn, and this reports it.

FOUR GATES, in this order. Each exists because the corresponding mistake is silent:

  1. Refuse if any final artifact already exists. Re-running test after seeing a disappointing
     number is the most natural form of test-set overfitting and leaves no trace.
  2. Require validation_summary.json with test_evaluated == false. If the summary does not
     describe a sealed test set, the selection it records cannot be trusted.
  3. Re-hash the train/val feature arrays and compare against what validation recorded. Otherwise
     features could have been regenerated between selection and evaluation, and the reported score
     would belong to a different dataset than the one the combiner was chosen on.
  4. Write the status record with open("x") BEFORE loading test. An exclusive create means a crash
     partway through still blocks a second attempt, rather than leaving the door open.

Checkpoints are loaded under the architecture they recorded, and their config fingerprint is
asserted, so a checkpoint from a different label set or window length fails loudly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from instrument_robustness.cnn_data import CNN as CNN_FEATURE_DIR
from instrument_robustness.cnn_data import load_cnn
from instrument_robustness.cnn_model import MediumCNN, hard_vote, predict_probs, soft_vote
from instrument_robustness.config import (
    ARTIFACTS,
    TARGET_LABELS,
    assert_fingerprint,
    config_fingerprint,
)
from instrument_robustness.train_cnn import sha256

COMBINERS = {"soft_vote": soft_vote, "hard_vote": hard_vote}


def load_checkpoint(path: Path, expected_arch: str, device: str):
    """Load one checkpoint under its recorded architecture, fingerprint-checked.

    Raises: ValueError on an architecture or label-order mismatch; StaleArtifactError on a config
    mismatch; FileNotFoundError if absent.
    """
    from instrument_robustness.crnn_model import MediumCRNN
    arches = {"MediumCNN": MediumCNN, "MediumCRNN": MediumCRNN}
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture")
    if arch != expected_arch:
        raise ValueError(f"{path} records architecture {arch!r}, expected {expected_arch!r}")
    if ckpt.get("label_order") != list(TARGET_LABELS):
        raise ValueError(f"{path} was trained on a different label order")
    assert_fingerprint(ckpt.get("config_fingerprint"), str(path))
    model = arches[arch]().to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def run_finalize(model_cls, output_dir: Path, device: str) -> dict:
    """Perform the one permitted test evaluation for `model_cls`.

    Postcondition: writes final_evaluation_status.json, test_summary.json and
    test_confusion_matrix.csv under output_dir, and returns the summary.
    Raises: RuntimeError if a final artifact already exists, or the validation summary is missing,
    unsealed, or describes different input data.
    """
    status_path = output_dir / "final_evaluation_status.json"
    summary_path = output_dir / "test_summary.json"
    confusion_path = output_dir / "test_confusion_matrix.csv"
    validation_path = output_dir / "validation_summary.json"

    existing = [p.name for p in (status_path, summary_path, confusion_path) if p.exists()]
    if existing:
        raise RuntimeError(
            f"Final {model_cls.__name__} evaluation has already started or completed; refusing to "
            f"evaluate test again. Existing: {', '.join(existing)}")
    if not validation_path.exists():
        raise RuntimeError(f"Missing {validation_path}; run the trainer first.")

    validation = json.loads(validation_path.read_text())
    if validation.get("test_evaluated") is not False:
        raise RuntimeError("The validation summary does not describe a sealed test set")
    if validation.get("architecture") != model_cls.__name__:
        raise RuntimeError(f"{validation_path} was written for "
                           f"{validation.get('architecture')!r}, not {model_cls.__name__!r}")
    for split, rec in validation.get("inputs", {}).items():
        current = sha256(Path(rec["path"]))
        if current != rec["sha256"]:
            raise RuntimeError(
                f"{split} features changed since validation ({rec['sha256'][:12]} -> "
                f"{current[:12]}). The selection was made on different data; retrain.")

    combiner_name = validation["selected_combiner"]
    seeds = validation["seeds"]
    print(f"{model_cls.__name__}: seeds {seeds}, combiner {combiner_name} (pre-committed)")

    # Status first, exclusively: a crash below must still block a second attempt.
    output_dir.mkdir(parents=True, exist_ok=True)
    with status_path.open("x", encoding="utf-8") as fh:
        json.dump({"architecture": model_cls.__name__, "seeds": seeds,
                   "selected_combiner": combiner_name, "state": "started"}, fh, indent=2)

    Xte, yte = load_cnn("test")
    models = [load_checkpoint(output_dir / f"model_s{s}.pt", model_cls.__name__, device)
              for s in seeds]
    probs = np.stack([predict_probs(m, Xte, device) for m in models])

    singles = [float(balanced_accuracy_score(yte, p.argmax(axis=1))) for p in probs]
    preds = COMBINERS[combiner_name](probs)
    labels = list(range(len(TARGET_LABELS)))

    # macro-F1 is recorded even though the combiner was selected on balanced accuracy. It is the
    # project's primary cross-model metric and the quantity noise_eval_common's clean-parity gate
    # compares against, so a summary without it cannot enter the noise sweep at all. Keeping
    # balanced accuracy and MCC alongside it costs nothing and preserves what selection used.
    macro_f1 = float(f1_score(yte, preds, labels=labels, average="macro", zero_division=0))
    summary = {
        "architecture": model_cls.__name__,
        "seeds": seeds,
        "combiner": combiner_name,
        "label_order": list(TARGET_LABELS),
        "config_fingerprint": config_fingerprint(),
        "selection_metric": "validation_balanced_accuracy",
        "n_test": int(len(yte)),
        "test_examples": int(len(yte)),
        "test_inputs": {"path": str(CNN_FEATURE_DIR / "test.npz"),
                        "sha256": sha256(CNN_FEATURE_DIR / "test.npz")},
        "single_seed": {"per_seed": singles, "mean": float(np.mean(singles)),
                        "std": float(np.std(singles, ddof=1)) if len(singles) > 1 else 0.0},
        "test_metrics": {
            "accuracy": float(accuracy_score(yte, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, preds)),
            "macro_f1": macro_f1,
            "mcc": float(matthews_corrcoef(yte, preds)),
        },
        "ensemble_balanced_accuracy": float(balanced_accuracy_score(yte, preds)),
        "ensemble_mcc": float(matthews_corrcoef(yte, preds)),
        "per_class": classification_report(yte, preds, labels=labels,
                                           target_names=list(TARGET_LABELS), output_dict=True,
                                           zero_division=0),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    cm = pd.crosstab(pd.Series([TARGET_LABELS[i] for i in yte], name="true_instrument"),
                     pd.Series([TARGET_LABELS[i] for i in preds], name="predicted_instrument"))
    cm = cm.reindex(index=TARGET_LABELS, columns=TARGET_LABELS, fill_value=0)
    cm.to_csv(confusion_path)

    status_path.write_text(json.dumps({"architecture": model_cls.__name__, "seeds": seeds,
                                       "selected_combiner": combiner_name,
                                       "state": "completed"}, indent=2))

    print(f"\nsingle seed  {summary['single_seed']['mean']:.4f} "
          f"+/- {summary['single_seed']['std']:.4f}")
    print(f"ensemble     {summary['ensemble_balanced_accuracy']:.4f} "
          f"| MCC {summary['ensemble_mcc']:.4f} "
          f"| macro-F1 {summary['test_metrics']['macro_f1']:.4f}")
    print(f"\nwrote {summary_path}, {confusion_path}, {status_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One permitted CNN test evaluation.")
    p.add_argument("--output-dir", type=Path, default=ARTIFACTS / "cnn")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_finalize(MediumCNN, args.output_dir, args.device)


if __name__ == "__main__":
    main()
