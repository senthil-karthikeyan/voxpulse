"""HTTP routes for health check and voice attribute analysis."""

from uuid import UUID
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logging import logger, mask_contact_id
from app.schemas.response import AnalyzeResponse, HealthResponse
from app.services.audio_processor import AudioProcessingError
from app.services.inference_service import inference_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check whether the service is running and ML models are loaded in memory.",
)
async def health_check() -> HealthResponse:
    """Return service health and model readiness status."""
    return HealthResponse(
        status="healthy",
        models_loaded=inference_service.is_ready,
        version=settings.APP_VERSION,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze caller audio for gender, age, and audio quality",
    description="Accepts multipart/form-data audio and returns gender, age bracket, confidence, and audio quality.",
)
async def analyze_audio(
    contact_id: UUID = Form(..., description="Unique caller UUID identifier"),
    audio: UploadFile = File(..., description="Audio file payload (WAV, MP3, M4A, OGG, Opus, etc.)"),
) -> AnalyzeResponse:
    """Analyze incoming audio file and predict demographic attributes."""
    masked_id = mask_contact_id(contact_id)

    # 1. Read audio bytes from upload stream
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        logger.error(
            f"Failed to read uploaded audio stream: {e}",
            extra={"event": "analysis_failed", "contact_id_hash": masked_id, "error_type": "read_error"},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read uploaded audio file stream.",
        )
    finally:
        await audio.close()

    if not audio_bytes or len(audio_bytes) == 0:
        logger.warning(
            "Empty audio upload received.",
            extra={"event": "analysis_failed", "contact_id_hash": masked_id, "error_type": "empty_upload"},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio upload is empty. Please provide a valid non-empty audio recording.",
        )

    # 2. Run attribute inference pipeline
    try:
        response = inference_service.analyze_bytes(audio_bytes, contact_id=contact_id)
        return response
    except AudioProcessingError as ape:
        logger.warning(
            f"Audio processing error: {ape}",
            extra={"event": "analysis_failed", "contact_id_hash": masked_id, "error_type": "audio_processing_error"},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing error: {str(ape)}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during analysis: {e}",
            exc_info=True,
            extra={"event": "analysis_failed", "contact_id_hash": masked_id, "error_type": "internal_error"},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing voice attributes.",
        )
