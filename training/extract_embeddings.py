"""Extracts and caches 192-d SpeechBrain ECAPA embeddings for train/val/test splits.

Usage:
    uv run python training/extract_embeddings.py [--data-dir ./data]
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from app.core.config import settings
from app.services.feature_extractor import feature_extractor


def extract_split_embeddings(
    json_path: Path, output_npz: Path, device: torch.device, batch_print: int = 100
) -> None:
    """Extract 192-d embeddings for a split and save to compressed npz."""
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"\nExtracting embeddings for {len(samples)} samples in {json_path.name}...")

    embeddings_list = []
    gender_labels = []  # 0: male, 1: female, -1: unlabeled
    age_labels = []  # 0: 18-30, 1: 31-45, 2: 46-60, 3: 60+, -1: unlabeled
    speaker_ids = []
    filenames = []
    durations = []
    sources = []

    gender_map = {"male": 0, "female": 1}
    age_map = {"18-30": 0, "31-45": 1, "46-60": 2, "60+": 3}

    with torch.inference_mode():
        for i, s in enumerate(samples, start=1):
            audio_path = s["audio_path"]
            wave, sr = sf.read(audio_path, dtype="float32")
            if wave.ndim > 1:
                wave = np.mean(wave, axis=1)

            emb = feature_extractor.extract_embedding(wave, sample_rate=sr)
            assert emb.shape[-1] == 192, f"Expected 192-dim embedding, got {emb.shape}"
            emb_np = emb.squeeze().cpu().numpy()

            embeddings_list.append(emb_np)
            g_lbl = gender_map.get(s.get("gender"), -1)
            a_lbl = age_map.get(s.get("age_bracket"), -1)

            gender_labels.append(g_lbl)
            age_labels.append(a_lbl)
            speaker_ids.append(s["speaker_id"])
            filenames.append(s["filename"])
            durations.append(s.get("duration_seconds", 0.0))
            sources.append(s.get("source", ""))

            if i % batch_print == 0 or i == len(samples):
                print(f"  [{i:>4}/{len(samples)}] Extracted embeddings (shape: {emb_np.shape})")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        embeddings=np.array(embeddings_list, dtype=np.float32),
        gender_labels=np.array(gender_labels, dtype=np.int64),
        age_labels=np.array(age_labels, dtype=np.int64),
        speaker_ids=np.array(speaker_ids),
        filenames=np.array(filenames),
        durations=np.array(durations, dtype=np.float32),
        sources=np.array(sources),
    )
    print(f"Saved {output_npz} with shape {np.array(embeddings_list).shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ECAPA embeddings for training")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    splits_dir = data_root / "splits"
    proc_dir = data_root / "processed"
    features_dir = data_root / "features"
    proc_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing feature extractor on device: {device}")
    feature_extractor.load_model()

    for split in ["train", "val", "test"]:
        json_file = splits_dir / f"{split}.json"
        out_npz = proc_dir / f"{split}_embeddings.npz"
        extract_split_embeddings(json_file, out_npz, device)

        # Also save or link to features dir
        feat_npz = features_dir / f"{split}_embeddings.npz"
        if feat_npz.resolve() != out_npz.resolve():
            import shutil
            shutil.copyfile(str(out_npz), str(feat_npz))

    print("\nEmbedding extraction complete for all splits.")


if __name__ == "__main__":
    main()
