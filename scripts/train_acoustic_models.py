"""Experiment B: openSMILE eGeMAPSv02 Acoustic Feature Extraction & Classical ML Model Training.

Extracts 88 standard acoustic features (F0/pitch, jitter, shimmer, HNR, formants F1-F3,
spectral flux, loudness, MFCCs) and benchmarks:
- Logistic Regression (Class-Weighted)
- Random Forest
- Support Vector Classifier (Linear & RBF)
- LightGBM Gradient Boosted Decision Trees

Usage:
    uv run python scripts/train_acoustic_models.py [--features-dir ./data/features]
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import opensmile
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import lightgbm as lgb


def load_manifest(jsonl_path: Path) -> list:
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples


def extract_or_load_features(features_dir: Path, data_dir: Path) -> dict:
    features_dir.mkdir(parents=True, exist_ok=True)
    cache_path = features_dir / "egemaps_features.npz"

    if cache_path.exists():
        print(f"Loading cached eGeMAPS features from {cache_path}...")
        data = np.load(cache_path)
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_val": data["X_val"],
            "y_val": data["y_val"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
        }

    print("Initializing openSMILE eGeMAPSv02 functional extractor...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    age_map = {"18-30": 0, "31-45": 1, "46-60": 2, "60+": 3}
    splits = ["train", "validation", "test"]
    split_arrays = {}

    for s in splits:
        manifest = data_dir / f"{s}.jsonl"
        samples = load_manifest(manifest)
        print(f"Extracting eGeMAPS features for {len(samples)} samples in {s}...")

        feats = []
        labels = []
        t0 = time.perf_counter()

        for idx, item in enumerate(samples):
            age_str = item.get("age_bracket")
            if age_str not in age_map:
                continue

            wav_path = Path(item["audio_path"])
            if not wav_path.exists():
                continue

            sig, sr = sf.read(wav_path)
            if sig.ndim > 1:
                sig = np.mean(sig, axis=-1)

            feat_df = smile.process_signal(sig, sr)
            feats.append(feat_df.values.flatten())
            labels.append(age_map[age_str])

            if (idx + 1) % 500 == 0 or idx + 1 == len(samples):
                print(f"  [{idx + 1:>4}/{len(samples)}] processed ({time.perf_counter() - t0:.1f}s)")

        split_arrays[f"X_{s[:2]}"] = np.array(feats, dtype=np.float32)
        split_arrays[f"y_{s[:2]}"] = np.array(labels, dtype=np.int64)

    np.savez_compressed(
        cache_path,
        X_train=split_arrays["X_tr"],
        y_train=split_arrays["y_tr"],
        X_val=split_arrays["X_va"],
        y_val=split_arrays["y_va"],
        X_test=split_arrays["X_te"],
        y_test=split_arrays["y_te"],
    )
    print(f"Saved extracted acoustic features to: {cache_path}")

    return {
        "X_train": split_arrays["X_tr"],
        "y_train": split_arrays["y_tr"],
        "X_val": split_arrays["X_va"],
        "y_val": split_arrays["y_va"],
        "X_test": split_arrays["X_te"],
        "y_test": split_arrays["y_te"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate classical ML models on eGeMAPS acoustic features")
    parser.add_argument("--features-dir", type=str, default="./data/features", help="Path to cached features")
    parser.add_argument("--data-dir", type=str, default="./data/processed", help="Path to jsonl manifests")
    args = parser.parse_args()

    data = extract_or_load_features(Path(args.features_dir), Path(args.data_dir))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"\nFeature Matrix Dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classifiers = {
        "Logistic Regression (Class Weighted)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Linear SVM": SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42),
        "RBF SVM": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "LightGBM (150 trees)": lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1),
    }

    print("\n" + "=" * 90)
    print("EXPERIMENT B: openSMILE eGeMAPSv02 (88 Acoustic Features) + Classical ML Models")
    print("=" * 90)
    print(f"{'Model Name':<35} | {'Val Acc':<9} | {'Val BAcc':<9} | {'Test Acc':<9} | {'Test BAcc':<9} | {'Test AdjAcc':<11} | {'Test MABE':<9}")
    print("-" * 98)

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)

        val_preds = clf.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds)
        val_bacc = balanced_accuracy_score(y_val, val_preds)

        test_preds = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_preds)
        test_bacc = balanced_accuracy_score(y_test, test_preds)
        test_adj_acc = float(np.mean(np.abs(y_test - test_preds) <= 1))
        test_mabe = float(np.mean(np.abs(y_test - test_preds)))

        print(f"{name:<35} | {val_acc * 100:>7.2f}% | {val_bacc * 100:>7.2f}% | {test_acc * 100:>7.2f}% | {test_bacc * 100:>7.2f}% | {test_adj_acc * 100:>9.2f}% | {test_mabe:>7.3f}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
