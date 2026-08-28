"""Evaluates trained demographic models on the held-out test split (unseen speakers).
Calculates exact accuracy, balanced accuracy, per-class metrics, adjacent bracket accuracy,
catastrophic error rate, confusion matrices, and exports misclassified samples to evaluation_errors.json.

Usage:
    uv run python training/evaluate_models.py [--data-dir ./data/processed] [--weights-dir ./model_weights] [--output evaluation_errors.json]
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from app.models.age_classifier import (
    AgeClassifierHead,
    BaselineLinearAgeHead,
    DeepMLPAgeHead,
    OrdinalAgeHead,
)
from app.models.gender_classifier import GenderClassifierHead
from training.calibrate_temperature import compute_brier_score, compute_ece


def evaluate_threshold_tradeoffs(
    probs: np.ndarray, targets: np.ndarray, threshold_steps: list[float]
) -> list[dict]:
    """Evaluate coverage, known-only accuracy, and unknown rate across confidence thresholds."""
    confidences = np.max(probs, axis=-1)
    raw_preds = np.argmax(probs, axis=-1)
    total = len(targets)

    table = []
    for thresh in threshold_steps:
        known_mask = confidences >= thresh
        known_count = int(np.sum(known_mask))
        unknown_count = total - known_count
        coverage = known_count / max(total, 1)
        unknown_rate = unknown_count / max(total, 1)

        if known_count > 0:
            known_acc = float((raw_preds[known_mask] == targets[known_mask]).mean())
        else:
            known_acc = 0.0

        table.append({
            "threshold": round(thresh, 2),
            "coverage": round(coverage * 100, 1),
            "known_accuracy": round(known_acc * 100, 1),
            "unknown_rate": round(unknown_rate * 100, 1),
            "known_samples": known_count,
            "unknown_samples": unknown_count,
        })
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained heads on held-out test split")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to .npz embeddings")
    parser.add_argument("--weights-dir", type=str, default="./model_weights", help="Path to .pt weights")
    parser.add_argument("--output-errors", type=str, default="evaluation_errors.json", help="Path to save misclassified samples")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    weights_dir = Path(args.weights_dir)

    test_file = data_dir / "test_embeddings.npz"
    val_file = data_dir / "val_embeddings.npz"

    test_data = np.load(test_file)
    x_test_all = test_data["embeddings"]
    y_gender_all = test_data["gender_labels"]
    y_age_all = test_data["age_labels"]
    test_spks = set(test_data["speaker_ids"])
    test_filenames = test_data["filenames"] if "filenames" in test_data else [f"sample_{i:04d}.wav" for i in range(len(x_test_all))]
    test_sources = test_data["sources"] if "sources" in test_data else ["test" for _ in range(len(x_test_all))]

    val_data = np.load(val_file)
    x_val_all = val_data["embeddings"]
    y_age_val_all = val_data["age_labels"]

    print("=" * 65)
    print("HELD-OUT TEST SPLIT EVALUATION (GENUINE LABELS & ZERO SPEAKER OVERLAP)")
    print("=" * 65)
    print(f"Total test samples:  {len(x_test_all)}")
    print(f"Total test speakers: {len(test_spks)} (strictly zero overlap with train/val)")

    # -------------------------------------------------------------
    # 1. Evaluate Gender Head
    # -------------------------------------------------------------
    gen_mask = y_gender_all >= 0
    if np.any(gen_mask):
        x_test_gen = torch.tensor(x_test_all[gen_mask], dtype=torch.float32)
        y_gender = y_gender_all[gen_mask]

        gender_weights = weights_dir / "gender_head.pt"
        if gender_weights.exists():
            gender_model = GenderClassifierHead(embedding_dim=192, hidden_dim=64)
            gender_model.load_state_dict(torch.load(gender_weights, weights_only=True))
            gender_model.eval()

            with torch.no_grad():
                logits = gender_model(x_test_gen)
                probs = F.softmax(logits, dim=-1).numpy()
                preds = probs.argmax(axis=-1)

            correct = int((preds == y_gender).sum())
            total = len(y_gender)
            acc = correct / float(total)

            cm_gender = np.zeros((2, 2), dtype=int)
            for t, p in zip(y_gender, preds):
                cm_gender[t, p] += 1

            print("\n" + "-" * 65)
            print("GENDER CLASSIFIER EVALUATION")
            print("-" * 65)
            print(f"Test Samples:          {total}")
            print(f"Test Accuracy:         {acc * 100:.2f}% ({correct}/{total} correct)")
            print(f"Average Confidence:    {float(np.mean(np.max(probs, axis=-1))) * 100:.2f}%")
            print("Confusion Matrix:")
            print("                Predicted Male   Predicted Female")
            print(f"  Actual Male       {cm_gender[0, 0]:<14} {cm_gender[0, 1]:<14}")
            print(f"  Actual Female     {cm_gender[1, 0]:<14} {cm_gender[1, 1]:<14}")

    # -------------------------------------------------------------
    # 2. Evaluate Age Head
    # -------------------------------------------------------------
    age_mask = y_age_all >= 0
    if np.any(age_mask):
        x_test_age = torch.tensor(x_test_all[age_mask], dtype=torch.float32)
        y_age = y_age_all[age_mask]
        test_age_spks = [s for s, m in zip(test_data["speaker_ids"], age_mask) if m]
        test_age_fnames = [f for f, m in zip(test_filenames, age_mask) if m]
        test_age_srcs = [src for src, m in zip(test_sources, age_mask) if m]

        val_age_mask = y_age_val_all >= 0
        x_val_age = torch.tensor(x_val_all[val_age_mask], dtype=torch.float32)
        y_age_val = y_age_val_all[val_age_mask]

        age_weights = weights_dir / "age_head.pt"
        if age_weights.exists():
            state_dict = torch.load(age_weights, weights_only=True)
            last_key = [k for k in state_dict.keys() if "weight" in k][-1]
            out_dim = state_dict[last_key].shape[0]

            if out_dim == 3:
                age_model = OrdinalAgeHead(embedding_dim=192, hidden_dim=64)
                is_ordinal = True
            elif len(state_dict) <= 2:
                age_model = BaselineLinearAgeHead(embedding_dim=192)
                is_ordinal = False
            elif any("2.weight" in k for k in state_dict.keys()):
                age_model = DeepMLPAgeHead(embedding_dim=192)
                is_ordinal = False
            else:
                age_model = AgeClassifierHead(embedding_dim=192, hidden_dim=64)
                is_ordinal = False

            age_model.load_state_dict(state_dict)
            age_model.eval()

            # Load learned temperature
            age_calib_path = weights_dir / "age_calibration.json"
            temperature = 1.0
            if age_calib_path.exists():
                with open(age_calib_path, "r", encoding="utf-8") as f:
                    temperature = float(json.load(f).get("temperature", 1.0))

            with torch.no_grad():
                if is_ordinal:
                    uncalib_probs = age_model.predict_probabilities(x_test_age, temperature=1.0)
                    calib_probs = age_model.predict_probabilities(x_test_age, temperature=temperature)
                    preds = calib_probs.argmax(axis=-1)
                    val_calib_probs = age_model.predict_probabilities(x_val_age, temperature=temperature)
                else:
                    logits_test = age_model(x_test_age)
                    uncalib_probs = F.softmax(logits_test, dim=-1).numpy()
                    calib_probs = F.softmax(logits_test / temperature, dim=-1).numpy()
                    preds = calib_probs.argmax(axis=-1)

                    logits_val = age_model(x_val_age)
                    val_calib_probs = F.softmax(logits_val / temperature, dim=-1).numpy()

            correct = int((preds == y_age).sum())
            total = len(y_age)
            acc = correct / float(total)

            class_names = ["18-30", "31-45", "46-60", "60+"]
            cm_age = np.zeros((4, 4), dtype=int)
            for t, p in zip(y_age, preds):
                cm_age[t, p] += 1

            # Metrics
            adjacent_correct = int((np.abs(y_age - preds) <= 1).sum())
            adj_acc = adjacent_correct / float(total)

            catastrophic_errors = int((np.abs(y_age - preds) >= 3).sum())
            catastrophic_rate = catastrophic_errors / float(total)

            mabe = float(np.mean(np.abs(y_age - preds)))

            # Per-class recall & balanced accuracy
            per_class_rec = []
            for c in range(4):
                c_mask = y_age == c
                c_tot = int(c_mask.sum())
                c_corr = int(((y_age == c) & (preds == c)).sum())
                per_class_rec.append(c_corr / max(c_tot, 1) if c_tot > 0 else 0.0)
            balanced_acc = float(np.mean(per_class_rec))

            ece_uncalib, _ = compute_ece(uncalib_probs, y_age)
            ece_calib, _ = compute_ece(calib_probs, y_age)
            brier_uncalib = compute_brier_score(uncalib_probs, y_age)
            brier_calib = compute_brier_score(calib_probs, y_age)

            print("\n" + "-" * 65)
            print(f"AGE BRACKET CLASSIFIER EVALUATION (Temperature T={temperature:.3f})")
            print("-" * 65)
            print(f"Test Samples:                 {total}")
            print(f"Exact Accuracy:               {acc * 100:.2f}% ({correct}/{total} correct)")
            print(f"Balanced Accuracy:            {balanced_acc * 100:.2f}%")
            print(f"Adjacent Bracket Accuracy:    {adj_acc * 100:.2f}% ({adjacent_correct}/{total} within ±1 bracket)")
            print(f"Catastrophic Error Rate:      {catastrophic_rate * 100:.2f}% ({catastrophic_errors}/{total} 18-30 <-> 60+)")
            print(f"Mean Absolute Bracket Error:  {mabe:.3f} brackets")
            print(f"Average Confidence (Uncalib): {float(np.mean(np.max(uncalib_probs, axis=-1))) * 100:.2f}% (ECE: {ece_uncalib*100:.2f}%, Brier: {brier_uncalib:.4f})")
            print(f"Average Confidence (Calib):   {float(np.mean(np.max(calib_probs, axis=-1))) * 100:.2f}% (ECE: {ece_calib*100:.2f}%, Brier: {brier_calib:.4f})")

            print("\nPer-Class Breakdown:")
            for i, name in enumerate(class_names):
                class_total = int((y_age == i).sum())
                class_corr = int(((y_age == i) & (preds == i)).sum())
                class_acc = (class_corr / class_total * 100.0) if class_total > 0 else 0.0
                print(f"  Class {name:<6}: {class_corr:>3}/{class_total:<3} correct ({class_acc:>5.1f}%)")

            print("\nConfusion Matrix:")
            header = " " * 16 + " ".join([f"Pred {n:<6}" for n in class_names])
            print(header)
            for i, name in enumerate(class_names):
                row_str = f"  Actual {name:<6}  " + "  ".join([f"{cm_age[i, j]:<8}" for j in range(4)])
                print(row_str)

            print("\n" + "-" * 65)
            print("VALIDATION SET CONFIDENCE THRESHOLD TRADEOFF TABLE")
            print("-" * 65)
            val_tradeoffs = evaluate_threshold_tradeoffs(
                val_calib_probs, y_age_val, [0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
            )
            print("  Threshold | Coverage | Accuracy on Known | Unknown Rate | Samples (K/U)")
            print("  " + "-" * 60)
            for row in val_tradeoffs:
                print(f"  {row['threshold']:<9.2f} | {row['coverage']:>6.1f}%  | {row['known_accuracy']:>15.1f}%  | {row['unknown_rate']:>10.1f}%  | {row['known_samples']:>4}/{row['unknown_samples']:<4}")

            # Export misclassified samples
            error_records = []
            for idx in range(total):
                t_idx = int(y_age[idx])
                p_idx = int(preds[idx])
                conf = float(calib_probs[idx, p_idx])
                if t_idx != p_idx:
                    error_records.append({
                        "speaker_id": str(test_age_spks[idx]),
                        "true_age": class_names[t_idx],
                        "predicted_age": class_names[p_idx],
                        "confidence": round(conf, 4),
                        "is_catastrophic": abs(t_idx - p_idx) >= 3,
                        "bracket_distance": abs(t_idx - p_idx),
                        "source": str(test_age_srcs[idx]),
                        "filename": str(test_age_fnames[idx]),
                        "audio_path": f"data/audio/{test_age_fnames[idx]}",
                    })

            err_path = Path(args.output_errors)
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(error_records, f, indent=2)
            print(f"\n[SAVED] {len(error_records)} misclassified error samples exported to: {err_path}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
