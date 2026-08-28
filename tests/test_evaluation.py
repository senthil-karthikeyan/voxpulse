"""Tests for Mozilla Common Voice evaluation harness, mappings, metrics, and calibration."""

from pathlib import Path
import pytest
from app.evaluation.dataset import CommonVoiceDataset, create_mock_common_voice_fixture
from app.evaluation.evaluator import Evaluator
from app.evaluation.mappings import normalize_age, normalize_gender
from app.evaluation.metrics import (
    SampleResult,
    compute_audio_quality_distribution,
    compute_classification_metrics,
    compute_confidence_calibration,
    compute_latency_metrics,
)


def test_gender_normalization_valid() -> None:
    """Verify valid gender strings are normalized to male/female."""
    assert normalize_gender("male") == "male"
    assert normalize_gender("Male") == "male"
    assert normalize_gender("MALE") == "male"
    assert normalize_gender("m") == "male"
    assert normalize_gender("man") == "male"

    assert normalize_gender("female") == "female"
    assert normalize_gender("Female") == "female"
    assert normalize_gender("FEMALE") == "female"
    assert normalize_gender("f") == "female"
    assert normalize_gender("woman") == "female"


def test_gender_normalization_excluded() -> None:
    """Verify non-binary or unsupported gender categories return None."""
    assert normalize_gender("other") is None
    assert normalize_gender("unknown") is None
    assert normalize_gender("non-binary") is None
    assert normalize_gender("") is None
    assert normalize_gender(None) is None
    assert normalize_gender("   ") is None


def test_age_normalization_brackets() -> None:
    """Verify Common Voice age deciles are mapped to VoxPulse brackets."""
    # 18-30
    assert normalize_age("twenties") == "18-30"
    assert normalize_age("20s") == "18-30"
    assert normalize_age("20-29") == "18-30"

    # 31-45
    assert normalize_age("thirties") == "31-45"
    assert normalize_age("30s") == "31-45"
    assert normalize_age("forties") == "31-45"
    assert normalize_age("40s") == "31-45"

    # 46-60
    assert normalize_age("fifties") == "46-60"
    assert normalize_age("50s") == "46-60"

    # 60+
    assert normalize_age("sixties") == "60+"
    assert normalize_age("seventies") == "60+"
    assert normalize_age("eighties") == "60+"
    assert normalize_age("nineties") == "60+"
    assert normalize_age("elderly") == "60+"


def test_age_normalization_teens_exclusion() -> None:
    """Verify 'teens' are strictly excluded from ground truth evaluation."""
    assert normalize_age("teens") is None
    assert normalize_age("teen") is None
    assert normalize_age("teenager") is None
    assert normalize_age("<20") is None


def test_age_normalization_unsupported() -> None:
    """Verify empty/unsupported age labels return None."""
    assert normalize_age("") is None
    assert normalize_age(None) is None
    assert normalize_age("unknown") is None
    assert normalize_age("invalid_label") is None


def test_classification_metrics_calculation() -> None:
    """Test strict accuracy, known-only accuracy, and unknown rates."""
    samples = [
        SampleResult(
            filename="s1.wav",
            gender_ground_truth="male",
            gender_prediction="male",
            gender_confidence=0.85,
            gender_correct=True,
        ),
        SampleResult(
            filename="s2.wav",
            gender_ground_truth="female",
            gender_prediction="female",
            gender_confidence=0.90,
            gender_correct=True,
        ),
        SampleResult(
            filename="s3.wav",
            gender_ground_truth="female",
            gender_prediction="male",
            gender_confidence=0.70,
            gender_correct=False,
        ),
        SampleResult(
            filename="s4.wav",
            gender_ground_truth="male",
            gender_prediction="unknown",
            gender_confidence=0.45,
            gender_correct=False,
        ),
        SampleResult(
            filename="s5.wav",
            gender_ground_truth=None,  # Unlabeled sample, must be excluded
            gender_prediction="male",
            gender_confidence=0.80,
            gender_correct=None,
        ),
    ]

    metrics = compute_classification_metrics(samples, attribute="gender")

    assert metrics["labeled_samples"] == 4
    assert metrics["correct_predictions"] == 2
    assert metrics["incorrect_predictions"] == 1
    assert metrics["unknown_predictions"] == 1

    # Strict accuracy: 2 correct / 4 labeled = 0.50
    assert metrics["strict_accuracy"] == 0.50

    # Known-only accuracy: 2 correct / (2 correct + 1 incorrect) = 0.6667
    assert metrics["known_only_accuracy"] == round(2 / 3.0, 4)

    # Unknown rate: 1 unknown / 4 labeled = 0.25
    assert metrics["unknown_rate"] == 0.25


def test_confidence_calibration_bucketing() -> None:
    """Test confidence calibration table generation across buckets."""
    samples = [
        SampleResult(
            filename="s1.wav",
            gender_ground_truth="male",
            gender_prediction="male",
            gender_confidence=0.55,
            gender_correct=True,
        ),
        SampleResult(
            filename="s2.wav",
            gender_ground_truth="female",
            gender_prediction="male",
            gender_confidence=0.58,
            gender_correct=False,
        ),
        SampleResult(
            filename="s3.wav",
            gender_ground_truth="female",
            gender_prediction="female",
            gender_confidence=0.85,
            gender_correct=True,
        ),
    ]

    calibration = compute_confidence_calibration(samples, attribute="gender")
    assert len(calibration) == 5

    # 0.50-0.60 bucket
    b1 = calibration[0]
    assert b1["bucket"] == "0.50-0.60"
    assert b1["samples"] == 2
    assert b1["accuracy"] == 0.50
    assert b1["avg_confidence"] == round((0.55 + 0.58) / 2, 4)

    # 0.80-0.90 bucket
    b4 = calibration[3]
    assert b4["bucket"] == "0.80-0.90"
    assert b4["samples"] == 1
    assert b4["accuracy"] == 1.00


def test_latency_and_quality_metrics() -> None:
    """Test latency percentile calculation and audio quality counts."""
    samples = [
        SampleResult(
            filename="s1.wav",
            processing_ms=100.0,
            request_latency_ms=120.0,
            audio_quality="good",
        ),
        SampleResult(
            filename="s2.wav",
            processing_ms=200.0,
            request_latency_ms=230.0,
            audio_quality="degraded",
        ),
        SampleResult(
            filename="s3.wav",
            processing_ms=300.0,
            request_latency_ms=340.0,
            audio_quality="good",
        ),
    ]

    lat = compute_latency_metrics(samples)
    assert lat["api_processing_ms"]["mean"] == 200.0
    assert lat["api_processing_ms"]["min"] == 100.0
    assert lat["api_processing_ms"]["max"] == 300.0
    assert lat["client_request_latency_ms"]["mean"] == 230.0

    qual = compute_audio_quality_distribution(samples)
    assert qual["good"] == 2
    assert qual["degraded"] == 1
    assert qual["insufficient"] == 0


def test_mock_common_voice_fixture_loading(tmp_path: Path) -> None:
    """Test creating and reading a mock Common Voice dataset fixture."""
    fixture_dir = create_mock_common_voice_fixture(tmp_path / "cv_mock", n_samples=5)
    dataset = CommonVoiceDataset(dataset_dir=fixture_dir, split="test")

    samples = dataset.load_samples()
    assert len(samples) == 5

    # Check sample attributes
    s0 = samples[0]
    assert s0.audio_path.exists()
    assert s0.raw_age == "twenties"
    assert s0.age_ground_truth == "18-30"
    assert s0.raw_gender == "male"
    assert s0.gender_ground_truth == "male"

    # Check teens exclusion in sample 4
    s4 = samples[4]
    assert s4.raw_age == "teens"
    assert s4.age_ground_truth is None  # Excluded
