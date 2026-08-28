"""Automated downloader and integrity verifier for SpeechBrain ECAPA-TDNN model."""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Required components for SpeechBrain spkrec-ecapa-voxceleb
REQUIRED_COMPONENTS = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "classifier.ckpt",
    "mean_var_norm_emb.ckpt",
    "label_encoder.ckpt",
]

DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MODEL_SUBDIR = "spkrec-ecapa-voxceleb"


def check_model_integrity(model_dir: Path) -> Tuple[bool, List[str]]:
    """Check if all required SpeechBrain model artifacts exist and have non-zero size.

    Args:
        model_dir: Directory containing the model checkpoint and yaml files.

    Returns:
        Tuple of (is_valid, list_of_missing_or_corrupt_files).
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return False, REQUIRED_COMPONENTS

    missing_or_invalid = []
    for filename in REQUIRED_COMPONENTS:
        file_path = model_dir / filename
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_or_invalid.append(filename)

    return len(missing_or_invalid) == 0, missing_or_invalid


def download_ecapa_model(
    cache_dir: Path,
    source: str = DEFAULT_MODEL_SOURCE,
    force: bool = False,
) -> bool:
    """Download and prepare the SpeechBrain ECAPA-TDNN pretrained model.

    Args:
        cache_dir: Base directory for storing pretrained models.
        source: HuggingFace repository ID.
        force: If True, re-downloads even if the model already exists.

    Returns:
        True if the model is ready and verified, False on failure.
    """
    target_dir = cache_dir / DEFAULT_MODEL_SUBDIR
    print("=" * 70)
    print("        VOXPULSE PRETRAINED ECAPA-TDNN MODEL SETUP")
    print("=" * 70)
    print(f"Target Directory: {target_dir.resolve()}")
    print(f"Model Source:     {source}")

    is_valid, missing = check_model_integrity(target_dir)

    if is_valid and not force:
        print("\n[INFO] Checking ECAPA-TDNN model...")
        print("[INFO] Model already exists and is fully verified on disk.")
        print("[INFO] All required components present:")
        for comp in REQUIRED_COMPONENTS:
            size_kb = (target_dir / comp).stat().st_size / 1024
            print(f"  - {comp:<25} ({size_kb:8.1f} KB)")
        print("\n[SUCCESS] ECAPA-TDNN model is ready for local inference and Docker.")
        print("=" * 70)
        return True

    if force:
        print("\n[INFO] Force flag supplied. Re-downloading ECAPA-TDNN model...")
    else:
        print("\n[INFO] Checking ECAPA-TDNN model...")
        print(f"[INFO] Model not found or incomplete. Missing/empty components: {missing}")
        print(f"[INFO] Downloading pretrained model from HuggingFace ({source})...")

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from speechbrain.inference.speaker import EncoderClassifier

        # Instantiate EncoderClassifier to download & populate target_dir
        model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(target_dir),
            run_opts={"device": "cpu"},
        )

        # Verify integrity after download
        is_valid_after, missing_after = check_model_integrity(target_dir)
        if not is_valid_after:
            print(f"\n[ERROR] Download completed but components are missing: {missing_after}", file=sys.stderr)
            return False

        print("\n[SUCCESS] Model downloaded and verified successfully.")
        print(f"[INFO] Model location: {target_dir.resolve()}")
        print("[INFO] Verified components:")
        for comp in REQUIRED_COMPONENTS:
            size_kb = (target_dir / comp).stat().st_size / 1024
            print(f"  - {comp:<25} ({size_kb:8.1f} KB)")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to download SpeechBrain ECAPA-TDNN model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    # Resolve default cache directory from app settings if available
    default_cache = Path("pretrained_models")
    try:
        from app.core.config import settings
        default_cache = Path(settings.SPEECHBRAIN_CACHE_DIR)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Download and verify SpeechBrain ECAPA-TDNN pretrained model for VoxPulse."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help=f"Base directory for storing pretrained models (default: {default_cache})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_MODEL_SOURCE,
        help=f"HuggingFace model source (default: {DEFAULT_MODEL_SOURCE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model files already exist",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify model integrity without downloading",
    )

    args = parser.parse_args()

    if args.verify_only:
        target_dir = args.cache_dir / DEFAULT_MODEL_SUBDIR
        is_valid, missing = check_model_integrity(target_dir)
        if is_valid:
            print(f"[VERIFIED] ECAPA-TDNN model at {target_dir} is complete and valid.")
            sys.exit(0)
        else:
            print(f"[FAILED] ECAPA-TDNN model at {target_dir} is missing: {missing}", file=sys.stderr)
            sys.exit(1)

    success = download_ecapa_model(
        cache_dir=args.cache_dir,
        source=args.source,
        force=args.force,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
