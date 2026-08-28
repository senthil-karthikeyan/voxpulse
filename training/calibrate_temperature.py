"""Confidence Calibration via Temperature Scaling on Validation Set.

Learns temperature parameter T > 0 on held-out validation logits by minimizing
Negative Log-Likelihood (NLL). Computes and reports Expected Calibration Error (ECE),
Brier Score, and reliability diagrams before and after temperature scaling.

Usage:
    uv run python training/calibrate_temperature.py [--weights-dir ./model_weights] [--data-dir ./data/processed]
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from app.models.age_classifier import (
    AgeClassifierHead,
    BaselineLinearAgeHead,
    DeepMLPAgeHead,
    OrdinalAgeHead,
)
from app.models.gender_classifier import GenderClassifierHead


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE) and bin statistics."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = int(np.sum(mask))

        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)
            bin_stats.append({
                "bin": f"{bin_lower:.2f}-{bin_upper:.2f}",
                "samples": bin_size,
                "avg_confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4),
            })
        else:
            bin_stats.append({
                "bin": f"{bin_lower:.2f}-{bin_upper:.2f}",
                "samples": 0,
                "avg_confidence": round((bin_lower + bin_upper) / 2.0, 4),
                "accuracy": 0.0,
            })

    return round(float(ece), 4), bin_stats


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int = 4) -> float:
    """Compute multi-class Brier Score: (1/N) * sum((probs - one_hot)^2)."""
    one_hot = np.zeros_like(probs)
    for idx, lbl in enumerate(labels):
        one_hot[idx, lbl] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return round(float(brier), 4)


def calibrate_model(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    model_name: str,
    output_path: Path,
) -> float:
    """Optimize temperature T > 0 on validation set using L-BFGS."""
    model.eval()
    y_val_np = y_val.cpu().numpy()

    if isinstance(model, OrdinalAgeHead):
        # Ordinal model probability extraction with temperature
        with torch.no_grad():
            raw_probs = model.predict_probabilities(x_val, temperature=1.0)

        ece_before, bins_before = compute_ece(raw_probs, y_val_np)
        brier_before = compute_brier_score(raw_probs, y_val_np)

        # Optimize temperature on threshold logits
        with torch.no_grad():
            raw_logits = model(x_val)

        temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        y_val_ord = torch.stack([
            (y_val > 0).float(),
            (y_val > 1).float(),
            (y_val > 2).float(),
        ], dim=-1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)

        def eval_loss():
            optimizer.zero_grad()
            clamped_temp = torch.clamp(temperature, min=0.1, max=10.0)
            loss = criterion(raw_logits / clamped_temp, y_val_ord)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        optimal_t = float(torch.clamp(temperature, min=0.1, max=10.0).item())

        with torch.no_grad():
            calibrated_probs = model.predict_probabilities(x_val, temperature=optimal_t)
        ece_after, bins_after = compute_ece(calibrated_probs, y_val_np)
        brier_after = compute_brier_score(calibrated_probs, y_val_np)

    else:
        with torch.no_grad():
            logits = model(x_val)

        raw_probs = F.softmax(logits, dim=-1).cpu().numpy()
        ece_before, bins_before = compute_ece(raw_probs, y_val_np)
        brier_before = compute_brier_score(raw_probs, y_val_np, n_classes=raw_probs.shape[-1])

        # Temperature parameter (initialized to 1.0)
        temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)

        def eval_loss():
            optimizer.zero_grad()
            clamped_temp = torch.clamp(temperature, min=0.1, max=10.0)
            loss = criterion(logits / clamped_temp, y_val)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        optimal_t = float(torch.clamp(temperature, min=0.1, max=10.0).item())

        with torch.no_grad():
            calibrated_probs = F.softmax(logits / optimal_t, dim=-1).cpu().numpy()
        ece_after, bins_after = compute_ece(calibrated_probs, y_val_np)
        brier_after = compute_brier_score(calibrated_probs, y_val_np, n_classes=raw_probs.shape[-1])

    calib_data = {
        "model_name": model_name,
        "temperature": round(optimal_t, 4),
        "validation_samples": len(x_val),
        "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "uncalibrated_avg_confidence": round(float(np.mean(np.max(raw_probs, axis=-1))), 4),
        "calibrated_avg_confidence": round(float(np.mean(np.max(calibrated_probs, axis=-1))), 4),
        "bins_before": bins_before,
        "bins_after": bins_after,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print("\n" + "=" * 65)
    print(f"TEMPERATURE SCALING CALIBRATION: {model_name}")
    print("=" * 65)
    print(f"Optimal Learned Temperature (T): {optimal_t:.4f}")
    print(f"Expected Calibration Error (ECE):")
    print(f"  Before Calibration (T=1.000):   {ece_before * 100:.2f}%")
    print(f"  After Calibration (T={optimal_t:.3f}):   {ece_after * 100:.2f}%")
    print(f"Brier Score:")
    print(f"  Before:                         {brier_before:.4f}")
    print(f"  After:                          {brier_after:.4f}")
    print(f"Average Model Confidence:")
    print(f"  Before:                         {calib_data['uncalibrated_avg_confidence'] * 100:.2f}%")
    print(f"  After:                          {calib_data['calibrated_avg_confidence'] * 100:.2f}%")
    print(f"Saved calibration parameters to:  {output_path}")
    print("=" * 65)

    return optimal_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate classifier confidence via Temperature Scaling")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to .npz embeddings")
    parser.add_argument("--weights-dir", type=str, default="./model_weights", help="Path to model weights")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    weights_dir = Path(args.weights_dir)

    val_file = data_dir / "val_embeddings.npz"
    if not val_file.exists():
        print(f"Error: {val_file} not found. Run extract_embeddings.py first.")
        return

    val_data = np.load(val_file)
    embeddings = val_data["embeddings"]
    y_age_raw = val_data["age_labels"]
    y_gender_raw = val_data["gender_labels"]

    # 1. Calibrate Age Head
    age_mask = y_age_raw >= 0
    if np.any(age_mask):
        x_val_age = torch.tensor(embeddings[age_mask], dtype=torch.float32)
        y_val_age = torch.tensor(y_age_raw[age_mask], dtype=torch.long)

        age_weights = weights_dir / "age_head.pt"
        if age_weights.exists():
            state_dict = torch.load(age_weights, weights_only=True)
            last_key = [k for k in state_dict.keys() if "weight" in k][-1]
            out_dim = state_dict[last_key].shape[0]

            if out_dim == 3:
                age_model = OrdinalAgeHead(embedding_dim=192, hidden_dim=64)
            elif len(state_dict) <= 2:
                age_model = BaselineLinearAgeHead(embedding_dim=192)
            elif any("2.weight" in k for k in state_dict.keys()):
                age_model = DeepMLPAgeHead(embedding_dim=192)
            else:
                age_model = AgeClassifierHead(embedding_dim=192, hidden_dim=64)

            age_model.load_state_dict(state_dict)
            calibrate_model(age_model, x_val_age, y_val_age, "AgeClassifierHead", weights_dir / "age_calibration.json")

    # 2. Calibrate Gender Head
    gender_mask = y_gender_raw >= 0
    if np.any(gender_mask):
        x_val_gen = torch.tensor(embeddings[gender_mask], dtype=torch.float32)
        y_val_gen = torch.tensor(y_gender_raw[gender_mask], dtype=torch.long)

        gender_weights = weights_dir / "gender_head.pt"
        if gender_weights.exists():
            gender_model = GenderClassifierHead(embedding_dim=192, hidden_dim=64)
            gender_model.load_state_dict(torch.load(gender_weights, weights_only=True))
            calibrate_model(gender_model, x_val_gen, y_val_gen, "GenderClassifierHead", weights_dir / "gender_calibration.json")


if __name__ == "__main__":
    main()
