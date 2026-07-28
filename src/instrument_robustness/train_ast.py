"""Fine-tune pretrained AST on the Step-5 normalized window splits."""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from instrument_robustness.ast_data import make_ast_dataloader
from instrument_robustness.config import DATA_ROOT, TARGET_LABELS
from instrument_robustness.pretrained_extractors import build_ast_extractor, build_ast_model

INSTRUMENT_FAMILIES = {
    "strings": ("violin", "viola", "cello"),
    "woodwinds": ("flute", "clarinet", "bassoon"),
    "brass": ("trumpet", "tuba", "trombone"),
}


def _run_epoch(
    model,
    loader,
    device: torch.device,
    optimizer=None,
    collect_predictions=False,
    phase="evaluation",
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

            outputs = model(**batch)
            loss = outputs.loss
            if training:
                loss.backward()
                optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            count += batch_size
            if collect_predictions:
                true_labels.extend(batch["labels"].cpu().tolist())
                predicted_labels.extend(predictions.cpu().tolist())
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == total_batches:
                print(
                    f"{phase}: batch {batch_index}/{total_batches} "
                    f"loss {total_loss / count:.4f} acc {correct / count:.3f}",
                    flush=True,
                )

    accuracy = correct / count
    metrics = {"loss": total_loss / count, "accuracy": accuracy, "accuracy_pct": round(100 * accuracy, 2)}
    if collect_predictions:
        return metrics, np.asarray(true_labels), np.asarray(predicted_labels)
    return metrics


def _percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _write_test_reports(output_dir: Path, true_labels: np.ndarray, predicted_labels: np.ndarray):
    per_instrument = []
    per_instrument_f1 = []
    for index, instrument in enumerate(TARGET_LABELS):
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

    label_to_index = {label: index for index, label in enumerate(TARGET_LABELS)}
    per_family = []
    for family, instruments in INSTRUMENT_FAMILIES.items():
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

    confusion = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=int)
    np.add.at(confusion, (true_labels, predicted_labels), 1)
    pd.DataFrame(
        confusion,
        index=TARGET_LABELS,
        columns=[f"predicted_{instrument}" for instrument in TARGET_LABELS],
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
) -> Dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    extractor = build_ast_extractor()
    loader_args = {
        "batch_size": batch_size,
        "extractor": extractor,
        "pin_memory": target_device.type == "cuda",
    }
    train_loader = make_ast_dataloader("train", **loader_args, shuffle=True)
    val_loader = make_ast_dataloader("val", **loader_args, shuffle=False)
    test_loader = make_ast_dataloader("test", **loader_args, shuffle=False)

    model = build_ast_model().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_accuracy = float("-inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            target_device,
            optimizer,
            phase=f"epoch {epoch} train",
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            target_device,
            phase=f"epoch {epoch} validation",
        )
        result = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(result)
        print(
            f"epoch {epoch}/{epochs} | train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.3f} | val loss {val_metrics['loss']:.4f} "
            f"acc {val_metrics['accuracy']:.3f}"
        )

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            model.save_pretrained(output_dir)
            extractor.save_pretrained(output_dir)

    from transformers import ASTForAudioClassification

    best_model = ASTForAudioClassification.from_pretrained(output_dir).to(target_device)
    test_metrics, true_labels, predicted_labels = _run_epoch(
        best_model,
        test_loader,
        target_device,
        collect_predictions=True,
        phase="test",
    )
    reports = _write_test_reports(output_dir, true_labels, predicted_labels)
    test_metrics.update(
        {
            "macro_f1": reports["summary"]["macro_f1"],
            "macro_f1_pct": reports["summary"]["macro_f1_pct"],
        }
    )
    metrics = {
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
    )


if __name__ == "__main__":
    main()
