"""Statistical metrics calculation and confidence calibration for evaluation results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class SampleResult:
    """Individual sample evaluation outcome."""

    filename: str
    gender_ground_truth: Optional[str] = None
    gender_prediction: Optional[str] = None
    gender_confidence: Optional[float] = None
    gender_correct: Optional[bool] = None

    age_ground_truth: Optional[str] = None
    age_prediction: Optional[str] = None
    age_confidence: Optional[float] = None
    age_correct: Optional[bool] = None

    audio_quality: Optional[str] = None
    processing_ms: Optional[float] = None
    request_latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary omitting None values where appropriate."""
        data = {
            "filename": self.filename,
            "gender_ground_truth": self.gender_ground_truth,
            "gender_prediction": self.gender_prediction,
            "gender_confidence": round(self.gender_confidence, 4) if self.gender_confidence is not None else None,
            "gender_correct": self.gender_correct,
            "age_ground_truth": self.age_ground_truth,
            "age_prediction": self.age_prediction,
            "age_confidence": round(self.age_confidence, 4) if self.age_confidence is not None else None,
            "age_correct": self.age_correct,
            "audio_quality": self.audio_quality,
            "processing_ms": round(self.processing_ms, 2) if self.processing_ms is not None else None,
            "request_latency_ms": round(self.request_latency_ms, 2) if self.request_latency_ms is not None else None,
        }
        if self.error:
            data["error"] = self.error
        return data


def compute_classification_metrics(
    samples: List[SampleResult], attribute: str = "gender"
) -> Dict[str, Any]:
    """Calculate classification metrics (strict accuracy, known-only accuracy, unknown rate, confidence stats).

    Args:
        samples: List of SampleResult objects.
        attribute: Either 'gender' or 'age'.

    Returns:
        Dictionary of classification metrics.
    """
    gt_key = f"{attribute}_ground_truth"
    pred_key = f"{attribute}_prediction"
    conf_key = f"{attribute}_confidence"
    corr_key = f"{attribute}_correct"

    # Filter to samples with valid ground truth
    labeled = [s for s in samples if getattr(s, gt_key) is not None and getattr(s, pred_key) is not None]

    if not labeled:
        return {
            "labeled_samples": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "unknown_predictions": 0,
            "strict_accuracy": 0.0,
            "known_only_accuracy": 0.0,
            "unknown_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_confidence_correct": 0.0,
            "avg_confidence_incorrect": 0.0,
        }

    correct_count = sum(1 for s in labeled if getattr(s, corr_key) is True)
    unknown_count = sum(1 for s in labeled if getattr(s, pred_key) == "unknown")
    incorrect_count = sum(
        1 for s in labeled if getattr(s, corr_key) is False and getattr(s, pred_key) != "unknown"
    )
    total_labeled = len(labeled)

    known_count = correct_count + incorrect_count
    strict_acc = correct_count / float(total_labeled)
    known_acc = (correct_count / float(known_count)) if known_count > 0 else 0.0
    unknown_rate = unknown_count / float(total_labeled)

    # Confidence statistics
    all_confs = [getattr(s, conf_key) for s in labeled if getattr(s, conf_key) is not None]
    corr_confs = [
        getattr(s, conf_key) for s in labeled if getattr(s, corr_key) is True and getattr(s, conf_key) is not None
    ]
    incorr_confs = [
        getattr(s, conf_key)
        for s in labeled
        if getattr(s, corr_key) is False
        and getattr(s, pred_key) != "unknown"
        and getattr(s, conf_key) is not None
    ]

    return {
        "labeled_samples": total_labeled,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "unknown_predictions": unknown_count,
        "strict_accuracy": round(strict_acc, 4),
        "known_only_accuracy": round(known_acc, 4),
        "unknown_rate": round(unknown_rate, 4),
        "avg_confidence": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
        "avg_confidence_correct": round(float(np.mean(corr_confs)), 4) if corr_confs else 0.0,
        "avg_confidence_incorrect": round(float(np.mean(incorr_confs)), 4) if incorr_confs else 0.0,
    }


def compute_confidence_calibration(
    samples: List[SampleResult], attribute: str = "gender"
) -> List[Dict[str, Any]]:
    """Compute empirical calibration across standard confidence buckets.

    Buckets adapt to class count:
    - Gender (2 classes): [0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90, 0.90-1.00]
    - Age (4 classes):    [0.20-0.40, 0.40-0.55, 0.55-0.70, 0.70-0.85, 0.85-1.00]
    """
    gt_key = f"{attribute}_ground_truth"
    pred_key = f"{attribute}_prediction"
    conf_key = f"{attribute}_confidence"
    corr_key = f"{attribute}_correct"

    # Only calibrate predictions with ground truth and non-unknown predictions
    valid_samples = [
        s
        for s in samples
        if getattr(s, gt_key) is not None
        and getattr(s, pred_key) is not None
        and getattr(s, pred_key) != "unknown"
        and getattr(s, conf_key) is not None
    ]

    if attribute == "gender":
        bucket_ranges = [
            (0.50, 0.60),
            (0.60, 0.70),
            (0.70, 0.80),
            (0.80, 0.90),
            (0.90, 1.00),
        ]
    else:
        bucket_ranges = [
            (0.20, 0.40),
            (0.40, 0.55),
            (0.55, 0.70),
            (0.70, 0.85),
            (0.85, 1.00),
        ]

    calibration_table = []
    for low, high in bucket_ranges:
        if high == 1.00:
            bucket_samples = [s for s in valid_samples if low <= getattr(s, conf_key) <= high]
        else:
            bucket_samples = [s for s in valid_samples if low <= getattr(s, conf_key) < high]

        n_samples = len(bucket_samples)
        if n_samples > 0:
            confs = [getattr(s, conf_key) for s in bucket_samples]
            n_correct = sum(1 for s in bucket_samples if getattr(s, corr_key) is True)
            avg_conf = float(np.mean(confs))
            acc = n_correct / float(n_samples)
        else:
            avg_conf = (low + high) / 2.0
            acc = 0.0

        calibration_table.append(
            {
                "bucket": f"{low:.2f}-{high:.2f}",
                "samples": n_samples,
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(acc, 4),
            }
        )

    return calibration_table


def compute_latency_metrics(samples: List[SampleResult]) -> Dict[str, Any]:
    """Compute summary statistics for API processing latency and client request latency."""
    proc_times = [s.processing_ms for s in samples if s.processing_ms is not None]
    req_times = [s.request_latency_ms for s in samples if s.request_latency_ms is not None]

    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        np_arr = np.array(arr)
        return {
            "mean": round(float(np.mean(np_arr)), 2),
            "median": round(float(np.median(np_arr)), 2),
            "p50": round(float(np.percentile(np_arr, 50)), 2),
            "p95": round(float(np.percentile(np_arr, 95)), 2),
            "min": round(float(np.min(np_arr)), 2),
            "max": round(float(np.max(np_arr)), 2),
        }

    return {
        "api_processing_ms": _stats(proc_times),
        "client_request_latency_ms": _stats(req_times),
    }


def compute_audio_quality_distribution(samples: List[SampleResult]) -> Dict[str, int]:
    """Compute counts for audio quality classifications."""
    counts = {"good": 0, "degraded": 0, "insufficient": 0}
    for s in samples:
        if s.audio_quality in counts:
            counts[s.audio_quality] += 1
    return counts
