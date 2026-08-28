"""Comprehensive Latency Benchmark Suite for VoxPulse Voice Attribute Service.

Profiles each pipeline stage (cold start vs warm inference) across multiple audio durations (1s, 3s, 5s).
Measures:
- Audio decoding (SoundFile/FFmpeg)
- Resampling & normalization
- Audio quality assessment & VAD
- SpeechBrain ECAPA embedding extraction
- Gender classifier head
- Age classifier head + temperature calibration
- Total end-to-end processing

Reports: Mean, Median (P50), P95, Minimum, Maximum latencies.

Usage:
    uv run python scripts/benchmark.py [--iterations 25] [--warmup 5]
"""

import argparse
import io
import time
from typing import Dict, List, Tuple
from uuid import uuid4
import numpy as np
import soundfile as sf
import torch

from app.core.config import settings
from app.models.age_classifier import AgeClassifier
from app.models.gender_classifier import GenderClassifier
from app.services.audio_processor import AudioProcessor
from app.services.audio_quality import AudioQualityService
from app.services.feature_extractor import FeatureExtractor
from app.services.inference_service import AttributeInferenceService
from app.utils.timing import StageTimer


def generate_synthetic_audio(duration_seconds: float, sample_rate: int = 16000) -> Tuple[bytes, np.ndarray]:
    """Generate authentic synthetic human vocal tone audio for benchmarking."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    # Fundamental frequency + formants to simulate human speech energy
    f0 = 150.0  # 150 Hz base pitch
    wave = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.05 * np.sin(2 * np.pi * 5 * f0 * t)
    )
    # Modulation
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    wave = (wave * mod).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), wave


def run_detailed_stage_benchmark(
    audio_bytes: bytes,
    processor: AudioProcessor,
    quality_service: AudioQualityService,
    extractor: FeatureExtractor,
    gender_cls: GenderClassifier,
    age_cls: AgeClassifier,
) -> Dict[str, float]:
    """Profile individual pipeline components with microsecond accuracy."""
    stages: Dict[str, float] = {}

    # 1. Decode & Resample
    t0 = time.perf_counter()
    waveform, sr = processor.decode_and_normalize(audio_bytes)
    t1 = time.perf_counter()
    stages["audio_decode_and_resample_ms"] = (t1 - t0) * 1000.0

    # 2. Quality & VAD
    t0 = time.perf_counter()
    quality = quality_service.evaluate(waveform, sample_rate=sr)
    t1 = time.perf_counter()
    stages["audio_quality_and_vad_ms"] = (t1 - t0) * 1000.0

    # 3. ECAPA Feature Extraction
    t0 = time.perf_counter()
    embedding = extractor.extract_embedding(waveform, sample_rate=sr)
    t1 = time.perf_counter()
    stages["ecapa_embedding_ms"] = (t1 - t0) * 1000.0

    # 4. Gender Classifier Head
    t0 = time.perf_counter()
    gender_res = gender_cls.predict(embedding)
    t1 = time.perf_counter()
    stages["gender_head_ms"] = (t1 - t0) * 1000.0

    # 5. Age Classifier Head + Temperature Scaling
    t0 = time.perf_counter()
    age_res = age_cls.predict(embedding)
    t1 = time.perf_counter()
    stages["age_head_calibrated_ms"] = (t1 - t0) * 1000.0

    # Total Pipeline Latency
    stages["total_pipeline_ms"] = sum(stages.values())
    return stages


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a latency metric."""
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxPulse Performance & Latency Benchmark")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("      VOXPULSE VOICE ATTRIBUTE SERVICE — LATENCY BENCHMARK")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Cold Start Benchmark
    # -------------------------------------------------------------
    print("\n[1/3] Measuring Cold Start Latency (Uncached Model Initialization)...")
    cold_start_t0 = time.perf_counter()

    proc = AudioProcessor()
    qual = AudioQualityService()
    feat = FeatureExtractor()
    feat.load_model()
    gen_cls = GenderClassifier()
    age_cls = AgeClassifier()

    cold_start_init_ms = (time.perf_counter() - cold_start_t0) * 1000.0

    # Cold first inference request (5-second audio)
    test_5s_bytes, _ = generate_synthetic_audio(5.0)
    cold_first_stages = run_detailed_stage_benchmark(test_5s_bytes, proc, qual, feat, gen_cls, age_cls)

    print(f"  Cold Model Load & Startup Time:  {cold_start_init_ms:.2f} ms")
    print(f"  First Cold Request Total Latency: {cold_first_stages['total_pipeline_ms']:.2f} ms")
    print(f"    - Decoding & Resampling:        {cold_first_stages['audio_decode_and_resample_ms']:.2f} ms")
    print(f"    - Quality & VAD:                {cold_first_stages['audio_quality_and_vad_ms']:.2f} ms")
    print(f"    - ECAPA Embedding:              {cold_first_stages['ecapa_embedding_ms']:.2f} ms")
    print(f"    - Gender Head:                  {cold_first_stages['gender_head_ms']:.2f} ms")
    print(f"    - Age Head (Calibrated):        {cold_first_stages['age_head_calibrated_ms']:.2f} ms")

    # -------------------------------------------------------------
    # 2. Warm Inference Benchmark Across Durations
    # -------------------------------------------------------------
    durations = [1.0, 3.0, 5.0]
    print(f"\n[2/3] Running Warm Inference Benchmark ({args.warmup} warmup, {args.iterations} benchmark runs)...")

    benchmark_results = {}

    for dur in durations:
        print(f"\n--- Testing Audio Duration: {dur:.1f} seconds ---")
        audio_bytes, _ = generate_synthetic_audio(dur)

        # Warmup runs
        for _ in range(args.warmup):
            _ = run_detailed_stage_benchmark(audio_bytes, proc, qual, feat, gen_cls, age_cls)

        # Timed benchmark runs
        stage_records: Dict[str, List[float]] = {
            "audio_decode_and_resample_ms": [],
            "audio_quality_and_vad_ms": [],
            "ecapa_embedding_ms": [],
            "gender_head_ms": [],
            "age_head_calibrated_ms": [],
            "total_pipeline_ms": [],
        }

        for _ in range(args.iterations):
            run_stages = run_detailed_stage_benchmark(audio_bytes, proc, qual, feat, gen_cls, age_cls)
            for k, v in run_stages.items():
                stage_records[k].append(v)

        dur_stats = {k: compute_stats(v) for k, v in stage_records.items()}
        benchmark_results[f"{dur}s"] = dur_stats

        print(f"  {'Stage Name':<32} | {'Mean (ms)':>9} | {'P50 (ms)':>8} | {'P95 (ms)':>8} | {'Min (ms)':>8} | {'Max (ms)':>8}")
        print("  " + "-" * 85)
        for stage_name, stats in dur_stats.items():
            print(f"  {stage_name:<32} | {stats['mean']:>9.2f} | {stats['p50']:>8.2f} | {stats['p95']:>8.2f} | {stats['min']:>8.2f} | {stats['max']:>8.2f}")

    # -------------------------------------------------------------
    # 3. Overall Target Compliance Assessment
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                 BENCHMARK SUMMARY & SLA TARGET COMPLIANCE")
    print("=" * 70)
    p50_5s = benchmark_results["5.0s"]["total_pipeline_ms"]["p50"]
    p95_5s = benchmark_results["5.0s"]["total_pipeline_ms"]["p95"]
    mean_5s = benchmark_results["5.0s"]["total_pipeline_ms"]["mean"]

    p50_gender = benchmark_results["5.0s"]["gender_head_ms"]["p50"]
    p50_age = benchmark_results["5.0s"]["age_head_calibrated_ms"]["p50"]

    meets_target = p95_5s < 500.0

    print(f"Target SLA:                       < 500.00 ms (CPU Warm Inference)")
    print(f"5-Second Audio Mean Latency:       {mean_5s:.2f} ms")
    print(f"5-Second Audio P50 (Median):       {p50_5s:.2f} ms")
    print(f"5-Second Audio P95 Latency:       {p95_5s:.2f} ms")
    print(f"Gender Head P50:                   {p50_gender:.2f} ms")
    print(f"Age Head (Calibrated) P50:         {p50_age:.2f} ms")
    print(f"SLA Target Status (<500ms):        {'[PASSED - MEETS TARGET]' if meets_target else '[EXCEEDED TARGET]'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
