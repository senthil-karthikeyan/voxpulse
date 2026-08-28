"""Unit tests for demographic model inference, safe fallback behavior, temperature calibration, and VAD audio quality."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from app.models.age_classifier import AgeClassifier, AgeClassifierHead
from app.models.gender_classifier import GenderClassifier, GenderClassifierHead
from app.schemas.response import AgeBracketEnum, AudioQualityEnum, GenderPredictionEnum
from app.services.audio_quality import AudioQualityService


def test_gender_classifier_missing_weights_fallback(tmp_path: Path) -> None:
    """When weights are missing, GenderClassifier must return unknown with 0.0 confidence."""
    non_existent = tmp_path / "non_existent_gender.pt"
    classifier = GenderClassifier(weights_path=non_existent)

    assert classifier.model_available is False
    assert classifier.model_source == "unavailable"

    dummy_emb = torch.randn(1, 192)
    result = classifier.predict(dummy_emb)

    assert result.prediction == GenderPredictionEnum.UNKNOWN
    assert result.confidence == 0.0


def test_age_classifier_missing_weights_fallback(tmp_path: Path) -> None:
    """When weights are missing, AgeClassifier must return unknown with 0.0 confidence."""
    non_existent = tmp_path / "non_existent_age.pt"
    classifier = AgeClassifier(weights_path=non_existent)

    assert classifier.model_available is False
    assert classifier.model_source == "unavailable"

    dummy_emb = torch.randn(1, 192)
    result = classifier.predict(dummy_emb)

    assert result.prediction == AgeBracketEnum.UNKNOWN
    assert result.confidence == 0.0


def test_gender_classifier_with_trained_weights(tmp_path: Path) -> None:
    """When trained weights exist, GenderClassifier loads them and performs inference."""
    weights_path = tmp_path / "test_gender.pt"
    head = GenderClassifierHead(embedding_dim=192, hidden_dim=64)
    torch.save(head.state_dict(), weights_path)

    classifier = GenderClassifier(weights_path=weights_path)
    assert classifier.model_available is True
    assert classifier.model_source == "trained"

    dummy_emb = torch.randn(1, 192)
    result = classifier.predict(dummy_emb)

    assert result.prediction in (GenderPredictionEnum.MALE, GenderPredictionEnum.FEMALE, GenderPredictionEnum.UNKNOWN)
    assert 0.0 <= result.confidence <= 1.0


def test_age_classifier_with_trained_weights(tmp_path: Path) -> None:
    """When trained weights exist, AgeClassifier loads them and performs inference."""
    weights_path = tmp_path / "test_age.pt"
    head = AgeClassifierHead(embedding_dim=192, hidden_dim=64)
    torch.save(head.state_dict(), weights_path)

    classifier = AgeClassifier(weights_path=weights_path)
    assert classifier.model_available is True
    assert classifier.model_source == "trained"

    dummy_emb = torch.randn(1, 192)
    result = classifier.predict(dummy_emb)

    assert result.prediction in classifier.classes or result.prediction == AgeBracketEnum.UNKNOWN
    assert 0.0 <= result.confidence <= 1.0


def test_age_classifier_temperature_scaling_calibration(tmp_path: Path) -> None:
    """AgeClassifier applies learned temperature scaling to soften overconfident logits."""
    weights_path = tmp_path / "test_age.pt"
    calib_path = tmp_path / "test_age_calibration.json"

    head = AgeClassifierHead(embedding_dim=192, hidden_dim=64)
    torch.save(head.state_dict(), weights_path)

    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": 2.0}, f)

    classifier = AgeClassifier(weights_path=weights_path, calibration_path=calib_path)
    assert classifier.temperature == 2.0

    dummy_emb = torch.randn(1, 192)
    res = classifier.predict(dummy_emb)
    assert 0.0 <= res.confidence <= 1.0


def test_speaker_split_zero_overlap() -> None:
    """Train, validation, and test split JSON files must contain zero speaker overlap."""
    splits_dir = Path("data/splits")
    if not (splits_dir / "train.json").exists():
        pytest.skip("Splits directory not yet populated")

    with open(splits_dir / "train.json", "r") as f:
        train_spks = {s["speaker_id"] for s in json.load(f)}
    with open(splits_dir / "val.json", "r") as f:
        val_spks = {s["speaker_id"] for s in json.load(f)}
    with open(splits_dir / "test.json", "r") as f:
        test_spks = {s["speaker_id"] for s in json.load(f)}

    assert len(train_spks & val_spks) == 0, "Speaker overlap detected between Train and Val!"
    assert len(train_spks & test_spks) == 0, "Speaker overlap detected between Train and Test!"
    assert len(val_spks & test_spks) == 0, "Speaker overlap detected between Val and Test!"


def test_vad_audio_quality_checks() -> None:
    """Audio quality service correctly distinguishes speech from silence."""
    service = AudioQualityService(min_duration_seconds=2.0, min_speech_duration_seconds=1.5)

    # 1. Total duration too short (< 2.0s)
    short_wave = np.zeros(16000 * 1, dtype=np.float32)  # 1.0s
    res_short = service.evaluate(short_wave, sample_rate=16000)
    assert res_short.quality == AudioQualityEnum.INSUFFICIENT

    # 2. 5.0s audio with only 0.5s speech (insufficient active speech)
    mostly_silent = np.zeros(16000 * 5, dtype=np.float32)
    mostly_silent[: 16000 // 2] = np.random.uniform(-0.5, 0.5, 16000 // 2).astype(np.float32)
    res_silent = service.evaluate(mostly_silent, sample_rate=16000)
    assert res_silent.quality == AudioQualityEnum.INSUFFICIENT

    # 3. 5.0s modulated harmonic voice-like signal with speech activity
    t = np.linspace(0, 5, 16000 * 5, endpoint=False, dtype=np.float32)
    carrier = 0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t)
    modulator = (np.sin(2 * np.pi * 1.5 * t) > -0.2).astype(np.float32)
    speech_wave = (carrier * modulator).astype(np.float32)

    res_speech = service.evaluate(speech_wave, sample_rate=16000)
    assert res_speech.quality == AudioQualityEnum.GOOD
    assert res_speech.speech_duration_seconds >= 1.5
