"""Trains Age Classifier PyTorch models across multiple architectures:
- Baseline Linear (192 -> 4)
- MLP Head (192 -> 64 -> 4)
- Deep MLP Head (192 -> 128 -> 64 -> 4)
- Ordinal Classifier (192 -> 64 -> 3 cumulative threshold heads)

Usage:
    uv run python training/train_age.py --experiment [baseline|mlp|deep_mlp|ordinal] [--epochs 100] [--lr 0.001]
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.models.age_classifier import (
    AgeClassifierHead,
    BaselineLinearAgeHead,
    DeepMLPAgeHead,
    OrdinalAgeHead,
)


def compute_age_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> dict:
    """Compute exact accuracy, balanced accuracy, adjacent accuracy, catastrophic error rate, and MABE."""
    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = correct / max(total, 1)

    # Per-class recall & Balanced accuracy
    per_class_recall = []
    for c in range(n_classes):
        mask = (y_true == c)
        c_total = int(mask.sum())
        if c_total > 0:
            c_corr = int(((y_true == c) & (y_pred == c)).sum())
            per_class_recall.append(c_corr / c_total)
        else:
            per_class_recall.append(0.0)

    balanced_acc = float(np.mean(per_class_recall))

    # Adjacent bracket accuracy (|y_true - y_pred| <= 1)
    adjacent_correct = int((np.abs(y_true - y_pred) <= 1).sum())
    adj_acc = adjacent_correct / max(total, 1)

    # Catastrophic error rate (|y_true - y_pred| >= 3: 18-30 <-> 60+)
    catastrophic_errors = int((np.abs(y_true - y_pred) >= 3).sum())
    catastrophic_rate = catastrophic_errors / max(total, 1)

    # Mean Absolute Bracket Error (MABE)
    mabe = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "adjacent_accuracy": float(adj_acc),
        "catastrophic_rate": float(catastrophic_rate),
        "mabe": float(mabe),
        "per_class_recall": [float(r) for r in per_class_recall],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Age Classification Models")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Directory with .npz files")
    parser.add_argument("--output-dir", type=str, default="./model_weights", help="Directory to save .pt weights")
    parser.add_argument("--log-dir", type=str, default="./training_logs", help="Directory to save CSV logs")
    parser.add_argument("--experiment", type=str, default="mlp", choices=["baseline", "mlp", "deep_mlp", "ordinal"], help="Model architecture experiment")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--save-primary", action="store_true", default=True, help="Save to primary age_head.pt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load cached embeddings
    train_data = np.load(data_dir / "train_embeddings.npz")
    val_data = np.load(data_dir / "val_embeddings.npz")

    train_mask = train_data["age_labels"] >= 0
    val_mask = val_data["age_labels"] >= 0

    x_train = torch.tensor(train_data["embeddings"][train_mask], dtype=torch.float32)
    y_train = torch.tensor(train_data["age_labels"][train_mask], dtype=torch.long)
    train_spks = set(train_data["speaker_ids"][train_mask])

    x_val = torch.tensor(val_data["embeddings"][val_mask], dtype=torch.float32)
    y_val = torch.tensor(val_data["age_labels"][val_mask], dtype=torch.long)
    val_spks = set(val_data["speaker_ids"][val_mask])

    class_names = ["18-30", "31-45", "46-60", "60+"]
    print("\n" + "=" * 65)
    print(f"AGE CLASSIFICATION TRAINING SETUP: EXPERIMENT [{args.experiment.upper()}]")
    print("=" * 65)
    print(f"Training set:   {len(x_train)} samples across {len(train_spks)} unique speakers")
    print(f"Validation set: {len(x_val)} samples across {len(val_spks)} unique speakers")

    # Compute class counts and inverse-frequency class weights exclusively from training data
    class_counts = [int((y_train == i).sum()) for i in range(4)]
    total_train = len(y_train)
    class_weights = [
        (total_train / (4.0 * max(c, 1))) for c in class_counts
    ]
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print("\nClass Distribution:")
    for i, name in enumerate(class_names):
        val_count = int((y_val == i).sum())
        print(f"  Class {i} ({name:<6}): Train={class_counts[i]:<4} | Val={val_count:<3} | Weight={class_weights[i]:.3f}")

    # Instantiate model according to experiment
    if args.experiment == "baseline":
        model = BaselineLinearAgeHead(embedding_dim=192)
        is_ordinal = False
    elif args.experiment == "deep_mlp":
        model = DeepMLPAgeHead(embedding_dim=192)
        is_ordinal = False
    elif args.experiment == "ordinal":
        model = OrdinalAgeHead(embedding_dim=192, hidden_dim=64)
        is_ordinal = True
    else:
        model = AgeClassifierHead(embedding_dim=192, hidden_dim=64)
        is_ordinal = False

    # Datasets and Loaders
    if is_ordinal:
        # Construct 3 binary threshold targets: y > 0, y > 1, y > 2
        y_train_ord = torch.stack([
            (y_train > 0).float(),
            (y_train > 1).float(),
            (y_train > 2).float(),
        ], dim=-1)

        y_val_ord = torch.stack([
            (y_val > 0).float(),
            (y_val > 1).float(),
            (y_val > 2).float(),
        ], dim=-1)

        train_dataset = TensorDataset(x_train, y_train_ord, y_train)
        val_dataset = TensorDataset(x_val, y_val_ord, y_val)
        criterion = nn.BCEWithLogitsLoss()
    else:
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_balanced_acc = 0.0
    best_state_dict = None
    best_epoch = 0

    log_rows = []

    print(f"\nStarting model training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        if is_ordinal:
            for batch_x, batch_y_ord, batch_y_orig in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y_ord)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_y_ord)
                # Compute predictions via threshold probability decomposition
                probs = model.predict_probabilities(batch_x)
                train_preds.extend(probs.argmax(axis=-1).tolist())
                train_targets.extend(batch_y_orig.cpu().numpy().tolist())
        else:
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_y)
                train_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                train_targets.extend(batch_y.cpu().numpy().tolist())

        scheduler.step()
        train_loss /= len(train_targets)
        train_m = compute_age_metrics(np.array(train_targets), np.array(train_preds))

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            if is_ordinal:
                for batch_x, batch_y_ord, batch_y_orig in val_loader:
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y_ord)
                    val_loss += loss.item() * len(batch_y_ord)
                    probs = model.predict_probabilities(batch_x)
                    val_preds.extend(probs.argmax(axis=-1).tolist())
                    val_targets.extend(batch_y_orig.cpu().numpy().tolist())
            else:
                for batch_x, batch_y in val_loader:
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * len(batch_y)
                    val_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                    val_targets.extend(batch_y.cpu().numpy().tolist())

        val_loss /= len(val_targets)
        val_m = compute_age_metrics(np.array(val_targets), np.array(val_preds))

        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_m["accuracy"], 4),
            "train_balanced_accuracy": round(train_m["balanced_accuracy"], 4),
            "train_adjacent_accuracy": round(train_m["adjacent_accuracy"], 4),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_m["accuracy"], 4),
            "val_balanced_accuracy": round(val_m["balanced_accuracy"], 4),
            "val_adjacent_accuracy": round(val_m["adjacent_accuracy"], 4),
            "val_catastrophic_rate": round(val_m["catastrophic_rate"], 4),
            "val_mabe": round(val_m["mabe"], 4),
            "learning_rate": current_lr,
        })

        # Track best checkpoint by validation balanced accuracy (and loss as tiebreaker)
        if val_m["balanced_accuracy"] > best_val_balanced_acc or (
            val_m["balanced_accuracy"] == best_val_balanced_acc and val_loss < best_val_loss
        ):
            best_val_loss = val_loss
            best_val_balanced_acc = val_m["balanced_accuracy"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_m['accuracy']*100:.1f}% BAcc: {train_m['balanced_accuracy']*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_m['accuracy']*100:.1f}% BAcc: {val_m['balanced_accuracy']*100:.1f}% AdjAcc: {val_m['adjacent_accuracy']*100:.1f}%"
            )

    # Save CSV training log
    csv_path = log_dir / f"age_training_log_{args.experiment}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[SAVED] Training log to: {csv_path}")

    # Save experiment-specific checkpoint
    exp_weights_path = out_dir / f"age_head_{args.experiment}.pt"
    torch.save(best_state_dict, exp_weights_path)
    print(f"[SAVED] Experiment checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f}, val_bacc={best_val_balanced_acc*100:.2f}%) to: {exp_weights_path}")

    if args.save_primary:
        primary_weights_path = out_dir / "age_head.pt"
        torch.save(best_state_dict, primary_weights_path)
        print(f"[SAVED] Primary active checkpoint to: {primary_weights_path}")

    # Evaluate best checkpoint on validation set
    model.load_state_dict(best_state_dict)
    model.eval()
    val_preds_list = []
    with torch.no_grad():
        if is_ordinal:
            for batch_x, _, _ in val_loader:
                probs = model.predict_probabilities(batch_x)
                val_preds_list.extend(probs.argmax(axis=-1).tolist())
        else:
            for batch_x, _ in val_loader:
                preds = model(batch_x).argmax(dim=-1)
                val_preds_list.extend(preds.cpu().numpy().tolist())

    val_preds_np = np.array(val_preds_list)
    val_targets_np = y_val.numpy()
    final_val_m = compute_age_metrics(val_targets_np, val_preds_np)

    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = val_targets_np == i
        if mask.sum() > 0:
            per_class_acc[name] = round(float((val_preds_np[mask] == i).mean()), 4)
        else:
            per_class_acc[name] = 0.0

    meta = {
        "model_name": f"AgeClassifierHead_{args.experiment}",
        "experiment": args.experiment,
        "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": 192,
        "classes": class_names,
        "class_mapping": {str(i): name for i, name in enumerate(class_names)},
        "dataset": "Mozilla Common Voice + Global Voices (Genuine Labels)",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(x_train),
        "train_speakers": len(train_spks),
        "val_samples": len(x_val),
        "val_speakers": len(val_spks),
        "class_distribution_train": {name: class_counts[i] for i, name in enumerate(class_names)},
        "class_weights": [round(w, 4) for w in class_weights],
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(final_val_m["accuracy"], 4),
        "best_val_balanced_accuracy": round(final_val_m["balanced_accuracy"], 4),
        "best_val_adjacent_accuracy": round(final_val_m["adjacent_accuracy"], 4),
        "best_val_catastrophic_rate": round(final_val_m["catastrophic_rate"], 4),
        "best_val_mabe": round(final_val_m["mabe"], 4),
        "val_per_class_accuracy": per_class_acc,
        "hyperparameters": {
            "experiment": args.experiment,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
    }

    meta_path = out_dir / "age_head_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] Training metadata to: {meta_path}")


if __name__ == "__main__":
    main()
