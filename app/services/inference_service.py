"""Inference service coordinating audio decoding, quality checks, embeddings, and attribute heads."""

from typing import Optional
from uuid import UUID
import numpy as np

from app.core.config import settings
from app.core.logging import logger, mask_contact_id
from app.models.age_classifier import AgeClassifier
from app.models.gender_classifier import GenderClassifier
from app.schemas.response import (
    AgeBracketEnum,
    AgeResult,
    AnalyzeResponse,
    AudioQualityEnum,
    GenderPredictionEnum,
    GenderResult,
)
from app.services.audio_processor import AudioProcessor, audio_processor
from app.services.audio_quality import AudioQualityService, audio_quality_service
from app.services.feature_extractor import FeatureExtractor, feature_extractor
from app.utils.timing import StageTimer


class AttributeInferenceService:
    """Coordinates the end-to-end voice attribute inference pipeline."""

    def __init__(
        self,
        processor: Optional[AudioProcessor] = None,
        quality_service: Optional[AudioQualityService] = None,
        extractor: Optional[FeatureExtractor] = None,
        gender_classifier: Optional[GenderClassifier] = None,
        age_classifier: Optional[AgeClassifier] = None,
    ) -> None:
        self.processor = processor or audio_processor
        self.quality_service = quality_service or audio_quality_service
        self.extractor = extractor or feature_extractor
        self.gender_classifier = gender_classifier or GenderClassifier()
        self.age_classifier = age_classifier or AgeClassifier()
        self.is_ready = False

    def initialize(self) -> None:
        """Load and warm up models during application startup."""
        logger.info(
            "Initializing voice attribute inference models...",
            extra={"event": "models_loading"},
        )
        self.extractor.load_model()
        self.is_ready = True
        logger.info(
            "Voice attribute inference models initialized and ready.",
            extra={"event": "models_loaded"},
        )

    def analyze_waveform(
        self,
        waveform: np.ndarray,
        contact_id: UUID,
        timer: Optional[StageTimer] = None,
    ) -> AnalyzeResponse:
        """Run attribute inference directly on a normalized 16kHz mono waveform.

        Args:
            waveform: 1D Float32 numpy array.
            contact_id: UUID of the caller.
            timer: Optional StageTimer for latency tracking.

        Returns:
            AnalyzeResponse with predictions, confidence, quality, and processing latency.
        """
        if timer is None:
            timer = StageTimer()

        masked_id = mask_contact_id(contact_id)

        # 1. Quality Check
        with timer.measure("quality_check_ms"):
            quality_result = self.quality_service.evaluate(
                waveform, sample_rate=settings.TARGET_SAMPLE_RATE
            )

        logger.info(
            f"Evaluated audio quality: {quality_result.quality.value}",
            extra={
                "event": "audio_quality_checked",
                "contact_id_hash": masked_id,
                "audio_quality": quality_result.quality.value,
                "audio_duration_seconds": quality_result.duration_seconds,
            },
        )

        # 2. Insufficient Audio Handling
        if quality_result.quality == AudioQualityEnum.INSUFFICIENT:
            gender_res = GenderResult(
                prediction=GenderPredictionEnum.UNKNOWN, confidence=0.0
            )
            age_res = AgeResult(prediction=AgeBracketEnum.UNKNOWN, confidence=0.0)

            total_ms = timer.total_elapsed_ms
            logger.info(
                "Audio quality insufficient for inference, returning unknown.",
                extra={
                    "event": "inference_completed",
                    "contact_id_hash": masked_id,
                    "audio_quality": quality_result.quality.value,
                    "gender_prediction": gender_res.prediction.value,
                    "age_prediction": age_res.prediction.value,
                    "processing_ms": total_ms,
                    "stage_timings": timer.stages,
                },
            )

            return AnalyzeResponse(
                contact_id=contact_id,
                gender=gender_res,
                age_bracket=age_res,
                processing_ms=total_ms,
                audio_quality=quality_result.quality,
            )

        # 3. Extract Shared Voice Embedding
        with timer.measure("embedding_ms"):
            embedding = self.extractor.extract_embedding(
                waveform, sample_rate=settings.TARGET_SAMPLE_RATE
            )

        # 4. Predict Gender
        with timer.measure("gender_inference_ms"):
            gender_res = self.gender_classifier.predict(embedding)

        # 5. Predict Age Bracket
        with timer.measure("age_inference_ms"):
            age_res = self.age_classifier.predict(embedding)

        total_ms = timer.total_elapsed_ms

        logger.info(
            "Attribute inference completed successfully.",
            extra={
                "event": "inference_completed",
                "contact_id_hash": masked_id,
                "audio_quality": quality_result.quality.value,
                "gender_prediction": gender_res.prediction.value,
                "gender_confidence": gender_res.confidence,
                "age_prediction": age_res.prediction.value,
                "age_confidence": age_res.confidence,
                "processing_ms": total_ms,
                "stage_timings": timer.stages,
            },
        )

        return AnalyzeResponse(
            contact_id=contact_id,
            gender=gender_res,
            age_bracket=age_res,
            processing_ms=total_ms,
            audio_quality=quality_result.quality,
        )

    def analyze_bytes(
        self, audio_bytes: bytes, contact_id: UUID
    ) -> AnalyzeResponse:
        """Decode raw audio bytes and run complete analysis pipeline.

        Args:
            audio_bytes: Raw binary audio data.
            contact_id: UUID of caller.

        Returns:
            AnalyzeResponse.
        """
        timer = StageTimer()
        masked_id = mask_contact_id(contact_id)

        logger.info(
            "Starting audio analysis request.",
            extra={"event": "analysis_started", "contact_id_hash": masked_id},
        )

        # 1. Decode and Normalize
        with timer.measure("decode_ms"):
            waveform, _ = self.processor.decode_and_normalize(audio_bytes)

        logger.info(
            f"Decoded and normalized audio: {len(waveform)} samples",
            extra={
                "event": "audio_normalized",
                "contact_id_hash": masked_id,
                "audio_duration_seconds": round(len(waveform) / settings.TARGET_SAMPLE_RATE, 2),
            },
        )

        # 2. Run Inference on Normalized Waveform
        response = self.analyze_waveform(waveform, contact_id=contact_id, timer=timer)

        logger.info(
            "Analysis pipeline finished.",
            extra={
                "event": "analysis_completed",
                "contact_id_hash": masked_id,
                "processing_ms": response.processing_ms,
            },
        )

        return response


inference_service = AttributeInferenceService()
