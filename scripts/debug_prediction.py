"""Audio Prediction Debugger CLI for VoxPulse Voice Attribute Service.

Inspects an audio file in detail:
- Decodes and resamples audio (16kHz mono)
- Evaluates audio quality and VAD metrics
- Enforces audio quality rejection gates (insufficient audio -> unknown)
- Extracts 192-d SpeechBrain ECAPA embedding (or force inference with --force-inference)
- Computes raw logits and probabilities for Age and Gender
- Applies validation temperature calibration
- Evaluates confidence thresholding and final API prediction outcome

Usage:
    uv run python scripts/debug_prediction.py --audio path/to/audio.wav [--force-inference]
"""

import argparse
import io
import json
import os
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.age_classifier import AgeClassifier, OrdinalAgeHead
from app.models.gender_classifier import GenderClassifier
from app.services.audio_processor import AudioProcessor
from app.services.audio_quality import AudioQualityService
from app.services.feature_extractor import FeatureExtractor
from app.schemas.response import AgeBracketEnum, AudioQualityEnum, GenderPredictionEnum


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug voice attribute predictions for an audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--age-threshold", type=float, default=settings.AGE_CONFIDENCE_THRESHOLD, help="Age confidence threshold")
    parser.add_argument("--gender-threshold", type=float, default=settings.GENDER_CONFIDENCE_THRESHOLD, help="Gender confidence threshold")
    parser.add_argument("--force-inference", action="store_true", help="Force ML inference even if audio quality is insufficient")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return

    # 1. Initialize Services
    processor = AudioProcessor()
    quality_service = AudioQualityService()
    feature_extractor = FeatureExtractor()
    feature_extractor.load_model()

    age_cls = AgeClassifier(confidence_threshold=args.age_threshold)
    gender_cls = GenderClassifier(confidence_threshold=args.gender_threshold)

    # 2. Decode Audio
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    waveform, sr = processor.decode_and_normalize(audio_bytes)
    total_samples = len(waveform)
    duration_s = total_samples / float(sr)

    # 3. Audio Quality & VAD
    quality_res = quality_service.evaluate(waveform, sample_rate=sr)

    print("\n" + "=" * 58)
    print("AUDIO DEBUG REPORT")
    print("=" * 58)
    print(f"File:               {audio_path.name}")
    print(f"Duration:           {duration_s:.2f} s ({total_samples} samples)")
    print(f"Sample rate:        {sr} Hz")
    print(f"Speech duration:    {quality_res.speech_duration_seconds:.2f} s")
    print(f"Silence ratio:      {(1.0 - quality_res.speech_ratio) * 100:.1f}%")
    print(f"SNR:                {quality_res.snr_db:.1f} dB")
    print(f"Clipping ratio:     {quality_res.clipping_ratio * 100:.2f}%")
    print(f"Audio quality:      {quality_res.quality.value.upper()}")
    if quality_res.reasons:
        print(f"Quality reasons:    {', '.join(quality_res.reasons)}")

    if quality_res.quality == AudioQualityEnum.INSUFFICIENT and not args.force_inference:
        print("\n" + "-" * 58)
        print("PRODUCTION API QUALITY GATE: REJECTED (INSUFFICIENT SPEECH)")
        print("-" * 58)
        print("Final API Gender:     unknown (confidence: 0.00%)")
        print("Final API Age:        unknown (confidence: 0.00%)")
        print("Note: To inspect underlying neural model logits anyway, pass --force-inference")
        print("=" * 58 + "\n")
        return

    # 4. Feature Extraction
    embedding = feature_extractor.extract_embedding(waveform, sample_rate=sr)

    # 5. Model Inference Details
    age_classes = [
        AgeBracketEnum.AGE_18_30,
        AgeBracketEnum.AGE_31_45,
        AgeBracketEnum.AGE_46_60,
        AgeBracketEnum.AGE_60_PLUS,
    ]
    with torch.no_grad():
        if isinstance(age_cls.model, OrdinalAgeHead):
            raw_probs = age_cls.model.predict_probabilities(embedding, temperature=1.0).squeeze(0)
            calib_probs = age_cls.model.predict_probabilities(embedding, temperature=age_cls.temperature).squeeze(0)
        else:
            raw_logits = age_cls.model(embedding)
            raw_probs = F.softmax(raw_logits, dim=-1).squeeze(0).cpu().numpy()
            calib_probs = F.softmax(raw_logits / max(age_cls.temperature, 0.1), dim=-1).squeeze(0).cpu().numpy()

    age_max_idx = int(calib_probs.argmax())
    age_raw_pred = age_classes[age_max_idx].value
    age_conf = float(calib_probs[age_max_idx])
    age_final = age_raw_pred if age_conf >= args.age_threshold else "unknown"

    gender_classes = [GenderPredictionEnum.MALE, GenderPredictionEnum.FEMALE]
    with torch.no_grad():
        gen_logits = gender_cls.model(embedding)
        gen_probs = F.softmax(gen_logits, dim=-1).squeeze(0).cpu().numpy()

    gen_max_idx = int(gen_probs.argmax())
    gen_raw_pred = gender_classes[gen_max_idx].value
    gen_conf = float(gen_probs[gen_max_idx])
    gen_final = gen_raw_pred if gen_conf >= args.gender_threshold else "unknown"

    print("\nAGE PREDICTION")
    print(f"Model Type:         {age_cls.model_type}")
    print(f"Temperature (T):    {age_cls.temperature:.4f}")
    print("\nRaw probabilities (uncalibrated):")
    for cls, p in zip(age_classes, raw_probs):
        print(f"  {cls.value:<7}: {p * 100:>6.2f}%")

    print("\nCalibrated probabilities:")
    for cls, p in zip(age_classes, calib_probs):
        print(f"  {cls.value:<7}: {p * 100:>6.2f}%")

    print(f"\nSelected prediction: {age_raw_pred}")
    print(f"Confidence:          {age_conf * 100:.2f}%")
    print(f"Threshold:           {args.age_threshold * 100:.1f}%")
    print(f"Final API prediction: {age_final if quality_res.quality != AudioQualityEnum.INSUFFICIENT else 'unknown (insufficient quality)'}")

    print("\nGENDER PREDICTION")
    print("Raw probabilities:")
    for cls, p in zip(gender_classes, gen_probs):
        print(f"  {cls.value:<7}: {p * 100:>6.2f}%")

    print(f"\nSelected prediction: {gen_raw_pred}")
    print(f"Confidence:          {gen_conf * 100:.2f}%")
    print(f"Threshold:           {args.gender_threshold * 100:.1f}%")
    print(f"Final API prediction: {gen_final if quality_res.quality != AudioQualityEnum.INSUFFICIENT else 'unknown (insufficient quality)'}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
