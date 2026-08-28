"""Evaluation engine orchestrating API execution, metric aggregation, and report generation."""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from app.evaluation.dataset import CommonVoiceDataset, DatasetSample
from app.evaluation.metrics import (
    SampleResult,
    compute_audio_quality_distribution,
    compute_classification_metrics,
    compute_confidence_calibration,
    compute_latency_metrics,
)


class Evaluator:
    """Evaluates VoxPulse Voice Attribute API performance against Common Voice ground truth."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def evaluate_sample(
        self, client: httpx.Client, sample: DatasetSample
    ) -> SampleResult:
        """Send a single sample to POST /analyze and evaluate predictions against ground truth."""
        contact_id = str(uuid.uuid4())
        audio_filename = sample.audio_path.name

        # Determine MIME type
        suffix = sample.audio_path.suffix.lower()
        content_type = "audio/mpeg" if suffix == ".mp3" else "audio/wav"

        start_time = time.perf_counter()
        try:
            with open(sample.audio_path, "rb") as af:
                files = {"audio": (audio_filename, af, content_type)}
                data = {"contact_id": contact_id}
                resp = client.post(
                    f"{self.api_url}/analyze",
                    data=data,
                    files=files,
                    timeout=self.timeout,
                )
            req_latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                return SampleResult(
                    filename=audio_filename,
                    gender_ground_truth=sample.gender_ground_truth,
                    age_ground_truth=sample.age_ground_truth,
                    request_latency_ms=req_latency_ms,
                    error=f"HTTP {resp.status_code}: {resp.text[:150]}",
                )

            payload = resp.json()
            gender_data = payload.get("gender", {})
            age_data = payload.get("age_bracket", {})

            pred_gender = gender_data.get("prediction")
            conf_gender = gender_data.get("confidence")

            pred_age = age_data.get("prediction")
            conf_age = age_data.get("confidence")

            quality = payload.get("audio_quality")
            proc_ms = payload.get("processing_ms")

            # Gender correctness
            gender_correct: Optional[bool] = None
            if sample.gender_ground_truth is not None and pred_gender is not None:
                gender_correct = (pred_gender == sample.gender_ground_truth)

            # Age correctness
            age_correct: Optional[bool] = None
            if sample.age_ground_truth is not None and pred_age is not None:
                age_correct = (pred_age == sample.age_ground_truth)

            return SampleResult(
                filename=audio_filename,
                gender_ground_truth=sample.gender_ground_truth,
                gender_prediction=pred_gender,
                gender_confidence=conf_gender,
                gender_correct=gender_correct,
                age_ground_truth=sample.age_ground_truth,
                age_prediction=pred_age,
                age_confidence=conf_age,
                age_correct=age_correct,
                audio_quality=quality,
                processing_ms=proc_ms,
                request_latency_ms=req_latency_ms,
            )

        except Exception as e:
            req_latency_ms = (time.perf_counter() - start_time) * 1000.0
            return SampleResult(
                filename=audio_filename,
                gender_ground_truth=sample.gender_ground_truth,
                age_ground_truth=sample.age_ground_truth,
                request_latency_ms=req_latency_ms,
                error=str(e)[:150],
            )

    def run_evaluation(
        self,
        dataset: CommonVoiceDataset,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute evaluation across all samples in dataset split."""
        samples = dataset.load_samples()
        total_requested = len(samples)

        results: List[SampleResult] = []
        with httpx.Client(timeout=self.timeout) as client:
            for idx, sample in enumerate(samples, start=1):
                res = self.evaluate_sample(client, sample)
                results.append(res)
                if progress_callback:
                    progress_callback(idx, total_requested, res)

        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        report = {
            "summary": {
                "dataset_path": str(dataset.dataset_dir),
                "split": dataset.split,
                "samples_requested": total_requested,
                "samples_processed": len(successful),
                "samples_failed": len(failed),
            },
            "gender": compute_classification_metrics(successful, attribute="gender"),
            "gender_calibration": compute_confidence_calibration(successful, attribute="gender"),
            "age_bracket": compute_classification_metrics(successful, attribute="age"),
            "age_calibration": compute_confidence_calibration(successful, attribute="age"),
            "latency": compute_latency_metrics(successful),
            "audio_quality": compute_audio_quality_distribution(successful),
            "samples": [r.to_dict() for r in results],
        }

        return report

    @staticmethod
    def format_console_report(report: Dict[str, Any]) -> str:
        """Format evaluation report as an aesthetic ASCII console table."""
        lines = []
        lines.append("=" * 50)
        lines.append("        VOXPULSE VOICE EVALUATION RESULTS")
        lines.append("=" * 50)

        s = report.get("summary", {})
        lines.append(f"Dataset:            {s.get('dataset_path')}")
        lines.append(f"Split:              {s.get('split')}")
        lines.append(f"Samples Requested:  {s.get('samples_requested')}")
        lines.append(f"Samples Processed:  {s.get('samples_processed')}")
        lines.append(f"Samples Failed:     {s.get('samples_failed')}")

        # Gender
        lines.append("-" * 50)
        lines.append("GENDER CLASSIFICATION")
        lines.append("-" * 50)
        g = report.get("gender", {})
        lines.append(f"Labeled Samples:       {g.get('labeled_samples', 0)}")
        lines.append(f"Correct:               {g.get('correct_predictions', 0)}")
        lines.append(f"Incorrect:             {g.get('incorrect_predictions', 0)}")
        lines.append(f"Unknown Predictions:   {g.get('unknown_predictions', 0)}")
        lines.append(f"Strict Accuracy:       {g.get('strict_accuracy', 0.0):.2%}")
        lines.append(f"Known-Only Accuracy:   {g.get('known_only_accuracy', 0.0):.2%}")
        lines.append(f"Unknown Rate:          {g.get('unknown_rate', 0.0):.2%}")
        lines.append(f"Average Confidence:    {g.get('avg_confidence', 0.0):.2%}")

        # Gender Calibration
        lines.append("\nGender Calibration:")
        lines.append(f"  {'Bucket':<12} {'Samples':<10} {'Avg Conf':<12} {'Accuracy':<10}")
        lines.append(f"  {'-'*10:<12} {'-'*8:<10} {'-'*10:<12} {'-'*8:<10}")
        for b in report.get("gender_calibration", []):
            lines.append(
                f"  {b['bucket']:<12} {b['samples']:<10} {b['avg_confidence']:<12.2%} {b['accuracy']:<10.2%}"
            )

        # Age Bracket
        lines.append("-" * 50)
        lines.append("AGE BRACKET CLASSIFICATION")
        lines.append("-" * 50)
        a = report.get("age_bracket", {})
        lines.append(f"Labeled Samples:       {a.get('labeled_samples', 0)}")
        lines.append(f"Correct:               {a.get('correct_predictions', 0)}")
        lines.append(f"Incorrect:             {a.get('incorrect_predictions', 0)}")
        lines.append(f"Unknown Predictions:   {a.get('unknown_predictions', 0)}")
        lines.append(f"Strict Accuracy:       {a.get('strict_accuracy', 0.0):.2%}")
        lines.append(f"Known-Only Accuracy:   {a.get('known_only_accuracy', 0.0):.2%}")
        lines.append(f"Unknown Rate:          {a.get('unknown_rate', 0.0):.2%}")
        lines.append(f"Average Confidence:    {a.get('avg_confidence', 0.0):.2%}")

        # Age Calibration
        lines.append("\nAge Calibration:")
        lines.append(f"  {'Bucket':<12} {'Samples':<10} {'Avg Conf':<12} {'Accuracy':<10}")
        lines.append(f"  {'-'*10:<12} {'-'*8:<10} {'-'*10:<12} {'-'*8:<10}")
        for b in report.get("age_calibration", []):
            lines.append(
                f"  {b['bucket']:<12} {b['samples']:<10} {b['avg_confidence']:<12.2%} {b['accuracy']:<10.2%}"
            )

        # Latency
        lines.append("-" * 50)
        lines.append("LATENCY PROFILING")
        lines.append("-" * 50)
        lat = report.get("latency", {})
        api_lat = lat.get("api_processing_ms", {})
        req_lat = lat.get("client_request_latency_ms", {})
        lines.append("API Processing (Server):")
        lines.append(f"  Mean:   {api_lat.get('mean', 0.0)} ms | Median: {api_lat.get('median', 0.0)} ms")
        lines.append(f"  P50:    {api_lat.get('p50', 0.0)} ms | P95:    {api_lat.get('p95', 0.0)} ms")
        lines.append(f"  Min:    {api_lat.get('min', 0.0)} ms | Max:    {api_lat.get('max', 0.0)} ms")
        lines.append("\nClient Request (Round-Trip):")
        lines.append(f"  Mean:   {req_lat.get('mean', 0.0)} ms | Median: {req_lat.get('median', 0.0)} ms")
        lines.append(f"  P50:    {req_lat.get('p50', 0.0)} ms | P95:    {req_lat.get('p95', 0.0)} ms")

        # Audio Quality
        lines.append("-" * 50)
        lines.append("AUDIO QUALITY DISTRIBUTION")
        lines.append("-" * 50)
        aq = report.get("audio_quality", {})
        total_q = sum(aq.values()) if aq else 1
        for k, v in aq.items():
            lines.append(f"  {k.capitalize():<14}: {v:<5} ({v/total_q:.1%})")

        lines.append("=" * 50)
        return "\n".join(lines)
