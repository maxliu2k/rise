"""Fine-tune pretrained AST on the Step-5 normalized window splits."""
import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from instrument_robustness.ast_data import make_ast_dataloader, resolve_ast_labels
from instrument_robustness.config import (
    DATA_ROOT,
    INSTRUMENT_FAMILY,
    WINDOWS_CSV,
    assert_fingerprint,
    config_fingerprint,
)
from instrument_robustness.pretrained_extractors import build_ast_extractor, build_ast_model


def _macro_f1(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    num_labels: int,
) -> float:
    scores = []
    for index in range(num_labels):
        actual = true_labels == index
        predicted = predicted_labels == index
        true_positive = int((actual & predicted).sum())
        predicted_count = int(predicted.sum())
        support = int(actual.sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def _balanced_class_weights(labels, num_labels: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_labels)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Cannot balance AST classes with no training examples: {missing}")
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(
    model,
    loader,
    device: torch.device,
    num_labels: int,
    optimizer=None,
    collect_predictions=False,
    phase="evaluation",
    class_weights: Optional[torch.Tensor] = None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    true_labels = []
    predicted_labels = []
    total_batches = len(loader)
    print(f"{phase}: {total_batches} batches", flush=True)

    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)

            labels = batch["labels"]
            model_inputs = {key: value for key, value in batch.items() if key != "labels"}
            outputs = model(**model_inputs)
            loss = F.cross_entropy(
                outputs.logits,
                labels,
                weight=class_weights if training else None,
            )
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            count += batch_size
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
                print(
                    f"{phase}: batch {batch_index}/{total_batches} "
                    f"loss {total_loss / count:.4f} acc {correct / count:.3f}",
                    flush=True,
                )

    accuracy = correct / count
    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted_labels)
    macro_f1 = _macro_f1(true_array, predicted_array, num_labels)
    metrics = {
        "loss": total_loss / count,
        "accuracy": accuracy,
        "accuracy_pct": round(100 * accuracy, 2),
        "macro_f1": macro_f1,
        "macro_f1_pct": round(100 * macro_f1, 2),
    }
    if collect_predictions:
        return metrics, true_array, predicted_array
    return metrics


def _percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _write_test_reports(
    output_dir: Path,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    label_names: Sequence[str],
):
    per_instrument = []
    per_instrument_f1 = []
    for index, instrument in enumerate(label_names):
        actual = true_labels == index
        predicted = predicted_labels == index
        support = int(actual.sum())
        predicted_count = int(predicted.sum())
        true_positive = int((actual & predicted).sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_instrument_f1.append(f1)
        per_instrument.append(
            {
                "instrument": instrument,
                "support": support,
                "correct": true_positive,
                "accuracy_pct": _percentage(true_positive, support),
                "precision_pct": round(100 * precision, 2),
                "recall_pct": round(100 * recall, 2),
                "f1_pct": round(100 * f1, 2),
            }
        )
    pd.DataFrame(per_instrument).to_csv(output_dir / "test_by_instrument.csv", index=False)

    accuracy = float((true_labels == predicted_labels).mean())
    macro_f1 = float(np.mean(per_instrument_f1))
    summary = {
        "test_clips": int(true_labels.size),
        "accuracy": accuracy,
        "accuracy_pct": round(100 * accuracy, 2),
        "macro_f1": macro_f1,
        "macro_f1_pct": round(100 * macro_f1, 2),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "test_summary.csv", index=False)

    family_to_instruments = {}
    for instrument in label_names:
        family = INSTRUMENT_FAMILY.get(instrument, "other")
        family_to_instruments.setdefault(family, []).append(instrument)

    label_to_index = {label: index for index, label in enumerate(label_names)}
    per_family = []
    for family, instruments in family_to_instruments.items():
        indices = [label_to_index[instrument] for instrument in instruments]
        actual = np.isin(true_labels, indices)
        predicted = np.isin(predicted_labels, indices)
        support = int(actual.sum())
        correct = int((actual & predicted).sum())
        per_family.append(
            {
                "family": family,
                "instruments": ", ".join(instruments),
                "support": support,
                "correct": correct,
                "accuracy_pct": _percentage(correct, support),
            }
        )
    pd.DataFrame(per_family).to_csv(output_dir / "test_by_family.csv", index=False)

    confusion = np.zeros((len(label_names), len(label_names)), dtype=int)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    pd.DataFrame(
        confusion,
        index=label_names,
        columns=[f"predicted_{instrument}" for instrument in label_names],
    ).to_csv(output_dir / "test_confusion_matrix.csv", index_label="true_instrument")

    return {"summary": summary, "per_instrument": per_instrument, "per_family": per_family}


def train(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
    seed: int,
    device: Optional[str] = None,
    use_class_weights: bool = True,
) -> Dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest_path = WINDOWS_CSV
    labels = resolve_ast_labels(manifest_path)
    num_labels = len(labels)
    print(f"AST classes ({num_labels}): {', '.join(labels)}", flush=True)

    extractor = build_ast_extractor()
    loader_args = {
        "batch_size": batch_size,
        "extractor": extractor,
        "pin_memory": target_device.type == "cuda",
        "label_names": labels,
        "manifest_path": manifest_path,
    }
    train_loader = make_ast_dataloader("train", **loader_args, shuffle=True)
    val_loader = make_ast_dataloader("val", **loader_args, shuffle=False)
    test_loader = make_ast_dataloader("test", **loader_args, shuffle=False)

    model = build_ast_model(labels).to(target_device)
    model.config.instrument_robustness_fingerprint = config_fingerprint()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_weights = None
    if use_class_weights:
        class_weights = _balanced_class_weights(train_loader.dataset.labels, num_labels).to(
            target_device
        )
        print(
            "class weights: "
            + ", ".join(
                f"{label}={weight:.3f}"
                for label, weight in zip(labels, class_weights.cpu().tolist())
            ),
            flush=True,
        )

    best_macro_f1 = float("-inf")
    best_accuracy = float("-inf")
    best_epoch = None
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            target_device,
            num_labels,
            optimizer=optimizer,
            phase=f"epoch {epoch} train",
            class_weights=class_weights,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            target_device,
            num_labels,
            phase=f"epoch {epoch} validation",
        )
        result = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(result)
        print(
            f"epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.3f} macro-F1 {train_metrics['macro_f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} "
            f"macro-F1 {val_metrics['macro_f1']:.3f}"
        )

        is_better = (
            val_metrics["macro_f1"] > best_macro_f1
            or (
                val_metrics["macro_f1"] == best_macro_f1
                and val_metrics["accuracy"] > best_accuracy
            )
        )
        if is_better:
            best_macro_f1 = val_metrics["macro_f1"]
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            model.save_pretrained(output_dir)
            extractor.save_pretrained(output_dir)

    from transformers import ASTForAudioClassification

    best_model = ASTForAudioClassification.from_pretrained(output_dir).to(target_device)
    assert_fingerprint(
        getattr(best_model.config, "instrument_robustness_fingerprint", None),
        str(output_dir),
    )
    test_metrics, true_labels, predicted_labels = _run_epoch(
        best_model,
        test_loader,
        target_device,
        num_labels,
        collect_predictions=True,
        phase="test",
    )
    reports = _write_test_reports(output_dir, true_labels, predicted_labels, labels)
    metrics = {
        "labels": labels,
        "num_labels": num_labels,
        "selection_metric": "validation_macro_f1",
        "best_epoch": best_epoch,
        "class_weights": (
            {
                label: weight
                for label, weight in zip(labels, class_weights.cpu().tolist())
            }
            if class_weights is not None
            else None
        ),
        "class_counts": {
            "train": train_loader.dataset.class_counts,
            "val": val_loader.dataset.class_counts,
            "test": test_loader.dataset.class_counts,
        },
        "config_fingerprint": config_fingerprint(),
        "history": history,
        "test": test_metrics,
        "per_instrument": reports["per_instrument"],
        "per_family": reports["per_family"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"test loss {test_metrics['loss']:.4f} | acc {test_metrics['accuracy']:.3f} "
        f"| macro-F1 {test_metrics['macro_f1']:.3f}"
    )
    print(f"wrote test reports to {output_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "models" / "ast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="Torch device, such as cuda or cpu")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        use_class_weights=not args.no_class_weights,
    )


if __name__ == "__main__":
    main()
