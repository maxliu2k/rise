from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from instrument_robustness.config import TARGET_LABELS
from instrument_robustness.mert_data import (
    MERT_FEATURE_DIR,
    load_mert_embedding_metadata,
    load_mert_embeddings,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "mert"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a linear probe on frozen MERT train/validation embeddings. "
            "This command never loads test data."
        )
    )
    parser.add_argument("--feature-dir", type=Path, default=MERT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if any(rate <= 0 for rate in args.learning_rates):
        parser.error("learning rates must be greater than zero")
    for name in ("batch_size", "max_epochs", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    return args


def choose_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict(model, X, *, batch_size: int, device: str, torch) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def score(y: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=list(range(len(TARGET_LABELS))),
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=list(range(len(TARGET_LABELS))),
            target_names=TARGET_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_candidate(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: str,
    torch,
    MERTProbe,
):
    seed_everything(seed, torch)
    model = MERTProbe(len(TARGET_LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    best_state = None
    best_metrics = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=generator).numpy()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            inputs = torch.from_numpy(X_train[indices]).to(device)
            targets = torch.from_numpy(y_train[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        predictions = predict(
            model,
            X_val,
            batch_size=batch_size,
            device=device,
            torch=torch,
        )
        metrics = score(y_val, predictions)
        current = (metrics["macro_f1"], metrics["accuracy"])
        previous = (
            (-1.0, -1.0)
            if best_metrics is None
            else (best_metrics["macro_f1"], best_metrics["accuracy"])
        )
        if current > previous:
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Probe training did not produce a checkpoint")

    model.load_state_dict(best_state)
    return model, best_state, best_metrics, best_epoch, epoch


def main() -> None:
    args = parse_args()
    try:
        import torch
        from instrument_robustness.mert_probe import MERTProbe
    except ImportError as error:
        raise RuntimeError(
            "MERT probe training requires PyTorch: pip install -e '.[mert]'"
        ) from error

    device = choose_device(args.device, torch)
    X_train, y_train = load_mert_embeddings("train", feature_dir=args.feature_dir)
    X_val, y_val = load_mert_embeddings("val", feature_dir=args.feature_dir)
    train_metadata = load_mert_embedding_metadata(
        "train", feature_dir=args.feature_dir
    )
    val_metadata = load_mert_embedding_metadata("val", feature_dir=args.feature_dir)
    if train_metadata != val_metadata:
        raise ValueError("Train and validation MERT embeddings use different extractors")

    results = []
    checkpoints = []
    for candidate, learning_rate in enumerate(args.learning_rates, start=1):
        started = perf_counter()
        model, state, metrics, best_epoch, epochs_run = train_candidate(
            X_train,
            y_train,
            X_val,
            y_val,
            learning_rate=learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
            device=device,
            torch=torch,
            MERTProbe=MERTProbe,
        )
        elapsed = perf_counter() - started
        results.append(
            {
                "candidate": candidate,
                "learning_rate": learning_rate,
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_weighted_f1": metrics["weighted_f1"],
                "fit_seconds": elapsed,
            }
        )
        checkpoints.append((model, state, metrics))
        print(
            f"[{candidate}/{len(args.learning_rates)}] lr={learning_rate:g}, "
            f"macro-F1={metrics['macro_f1']:.6f}"
        )

    frame = pd.DataFrame(results).sort_values(
        by=["validation_macro_f1", "validation_accuracy", "candidate"],
        ascending=[False, False, True],
        kind="stable",
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    best_row = frame.iloc[0]
    best_index = int(best_row["candidate"]) - 1
    best_model, best_state, best_metrics = checkpoints[best_index]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    search_path = output_dir / "validation_search.csv"
    confusion_path = output_dir / "validation_confusion_matrix.csv"
    model_path = output_dir / "best_probe.pt"
    summary_path = output_dir / "validation_summary.json"
    frame.to_csv(search_path, index=False)

    confusion = pd.DataFrame(
        best_metrics["confusion_matrix"],
        index=TARGET_LABELS,
        columns=TARGET_LABELS,
    )
    confusion.index.name = "actual"
    confusion.to_csv(confusion_path)

    torch.save(
        {
            "state_dict": best_state,
            "num_classes": len(TARGET_LABELS),
            "label_order": TARGET_LABELS,
            "learning_rate": float(best_row["learning_rate"]),
            "layer_weights": best_model.layer_weights(),
        },
        model_path,
    )

    feature_dir = Path(args.feature_dir)
    train_path = feature_dir / "train.npz"
    val_path = feature_dir / "val.npz"
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "frozen MERT-v1-95M layer-weighted linear probe",
        "selection_metric": "validation_macro_f1",
        "best_config": {
            "learning_rate": float(best_row["learning_rate"]),
            "batch_size": args.batch_size,
            "best_epoch": int(best_row["best_epoch"]),
            "seed": args.seed,
        },
        "validation_metrics": best_metrics,
        "learned_layer_weights": best_model.layer_weights(),
        "label_order": TARGET_LABELS,
        "training_examples": int(len(y_train)),
        "validation_examples": int(len(y_val)),
        "model_fit_splits": ["train"],
        "backbone_frozen": True,
        "embedding_schema": train_metadata,
        "test_evaluated": False,
        "input_files": {
            "train": {
                "path": str(train_path.resolve()),
                "sha256": sha256(train_path),
            },
            "val": {
                "path": str(val_path.resolve()),
                "sha256": sha256(val_path),
            },
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "output_files": {
            "validation_search": {
                "path": str(search_path.resolve()),
                "sha256": sha256(search_path),
            },
            "validation_confusion_matrix": {
                "path": str(confusion_path.resolve()),
                "sha256": sha256(confusion_path),
            },
            "model": {
                "path": str(model_path.resolve()),
                "sha256": sha256(model_path),
            },
        },
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Best learning rate:", summary["best_config"]["learning_rate"])
    print("Validation macro-F1:", best_metrics["macro_f1"])
    print(f"Saved outputs under {output_dir}")


if __name__ == "__main__":
    main()
