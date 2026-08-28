"""Mozilla Common Voice evaluation framework for VoxPulse."""

from app.evaluation.dataset import CommonVoiceDataset, DatasetSample, create_mock_common_voice_fixture
from app.evaluation.evaluator import Evaluator
from app.evaluation.mappings import normalize_age, normalize_gender
from app.evaluation.metrics import (
    SampleResult,
    compute_audio_quality_distribution,
    compute_classification_metrics,
    compute_confidence_calibration,
    compute_latency_metrics,
)

__all__ = [
    "CommonVoiceDataset",
    "DatasetSample",
    "create_mock_common_voice_fixture",
    "Evaluator",
    "normalize_age",
    "normalize_gender",
    "SampleResult",
    "compute_audio_quality_distribution",
    "compute_classification_metrics",
    "compute_confidence_calibration",
    "compute_latency_metrics",
]
