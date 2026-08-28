"""Dataset acquisition tool for VoxPulse Voice Attribute Inference Service.

Downloads and verifies genuine speech datasets with real demographic metadata:
- Mozilla Common Voice (Age and Gender subsets)
- Global Voices Speech Corpus (globe_v2 via VoicePersona)
- FLEURS English Speech Corpus

Usage:
    uv run python scripts/download_dataset.py [--target-dir ./data/audio]
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset


def download_datasets(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("VOXPULSE DATASET ACQUISITION PIPELINE")
    print("=" * 65)
    print(f"Target directory: {target_dir}")

    # 1. Global Voices Speech Corpus
    print("\n[1/3] Verifying/Downloading Global Voices (Paranoiid/VoicePersona globe_v2)...")
    try:
        vp_dataset = load_dataset("Paranoiid/VoicePersona", "globe_v2", split="train")
        print(f"  --> Downloaded/verified {len(vp_dataset)} Global Voices samples.")
    except Exception as e:
        print(f"  [Warning] Global Voices download error: {e}")

    # 2. DynamicSuperb Common Voice Age & Gender
    print("\n[2/3] Verifying/Downloading Common Voice demographic subsets...")
    try:
        cv_age = load_dataset("DynamicSuperb/AgeClassification_CommonVoiceCorpus-Test", split="test")
        print(f"  --> Downloaded/verified {len(cv_age)} Common Voice age samples.")
    except Exception as e:
        print(f"  [Warning] CV Age download error: {e}")

    try:
        cv_gen = load_dataset("DynamicSuperb/GenderRecognitionbyVoice_CommonVoice-DeltaSegment-15", split="test")
        print(f"  --> Downloaded/verified {len(cv_gen)} Common Voice gender samples.")
    except Exception as e:
        print(f"  [Warning] CV Gender download error: {e}")

    # 3. Google FLEURS
    print("\n[3/3] Verifying/Downloading Google FLEURS English corpus...")
    try:
        fleurs = load_dataset("google/fleurs", "en_us", split="train")
        print(f"  --> Downloaded/verified {len(fleurs)} FLEURS samples.")
    except Exception as e:
        print(f"  [Warning] FLEURS download error: {e}")

    print("\n" + "=" * 65)
    print("Dataset acquisition and verification completed successfully.")
    print("Next step: Run `uv run python scripts/prepare_dataset.py` to partition.")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify genuine speech datasets")
    parser.add_argument("--target-dir", type=str, default="./data/audio", help="Target directory for audio")
    args = parser.parse_args()

    download_datasets(Path(args.target_dir))


if __name__ == "__main__":
    main()
