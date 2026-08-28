"""Trains the GenderClassifierHead PyTorch model on extracted genuine ECAPA embeddings.

Usage:
    uv run python training/train_gender.py [--epochs 60] [--lr 0.001]
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.models.gender_classifier import GenderClassifierHead


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gender Classification Head")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Directory with .npz files")
    parser.add_argument("--output-dir", type=str, default="./model_weights", help="Directory to save .pt weights")
    parser.add_argument("--log-dir", type=str, default="./training_logs", help="Directory to save CSV logs")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
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

    train_mask = train_data["gender_labels"] >= 0
    val_mask = val_data["gender_labels"] >= 0

    x_train = torch.tensor(train_data["embeddings"][train_mask], dtype=torch.float32)
    y_train = torch.tensor(train_data["gender_labels"][train_mask], dtype=torch.long)
    train_spks = set(train_data["speaker_ids"][train_mask])

    x_val = torch.tensor(val_data["embeddings"][val_mask], dtype=torch.float32)
    y_val = torch.tensor(val_data["gender_labels"][val_mask], dtype=torch.long)
    val_spks = set(val_data["speaker_ids"][val_mask])

    print("\n" + "=" * 60)
    print("GENDER CLASSIFICATION TRAINING SETUP (GENUINE LABELS)")
    print("=" * 60)
    print(f"Training set:   {len(x_train)} samples across {len(train_spks)} unique speakers")
    print(f"Validation set: {len(x_val)} samples across {len(val_spks)} unique speakers")
    print(f"Class distribution (train): Male={int((y_train==0).sum())}, Female={int((y_train==1).sum())}")
    print(f"Class distribution (val):   Male={int((y_val==0).sum())}, Female={int((y_val==1).sum())}")

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = GenderClassifierHead(embedding_dim=192, hidden_dim=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    log_rows = []

    print("\nStarting model training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = logits.argmax(dim=-1)
            train_correct += int((preds == batch_y).sum())
            total_train += len(batch_y)

        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = logits.argmax(dim=-1)
                val_correct += int((preds == batch_y).sum())
                total_val += len(batch_y)

        val_loss /= total_val
        val_acc = val_correct / total_val

        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": current_lr,
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%"
            )

    # Save CSV training log
    csv_path = log_dir / "gender_training_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[SAVED] Training log to: {csv_path}")

    # Save best model checkpoint
    weights_path = out_dir / "gender_head.pt"
    torch.save(best_state_dict, weights_path)
    print(f"[SAVED] Best checkpoint (epoch {best_epoch}, val_acc={best_val_acc*100:.2f}%) to: {weights_path}")

    # Save training metadata
    meta = {
        "model_name": "GenderClassifierHead",
        "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": 192,
        "hidden_dim": 64,
        "classes": ["male", "female"],
        "class_mapping": {"0": "male", "1": "female"},
        "dataset": "Mozilla Common Voice + FLEURS (Genuine Labels)",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(x_train),
        "train_speakers": len(train_spks),
        "val_samples": len(x_val),
        "val_speakers": len(val_spks),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
        },
    }

    meta_path = out_dir / "gender_head_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] Training metadata to: {meta_path}")


if __name__ == "__main__":
    main()
