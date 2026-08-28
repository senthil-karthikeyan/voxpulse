"""Response schemas and enums for VoxPulse Voice Attribute Inference Service."""

from enum import Enum
from typing import Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class AudioQualityEnum(str, Enum):
    """Audio quality classification levels."""

    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class GenderPredictionEnum(str, Enum):
    """Gender prediction categories."""

    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeBracketEnum(str, Enum):
    """Age bracket prediction categories."""

    AGE_18_30 = "18-30"
    AGE_31_45 = "31-45"
    AGE_46_60 = "46-60"
    AGE_60_PLUS = "60+"
    UNKNOWN = "unknown"


class PredictionResult(BaseModel):
    """Generic prediction result containing label and confidence score."""

    prediction: str = Field(..., description="Predicted label or 'unknown'")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score between 0.0 and 1.0",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class GenderResult(PredictionResult):
    """Gender prediction outcome."""

    prediction: GenderPredictionEnum = Field(
        ..., description="Gender label: male, female, or unknown"
    )


class AgeResult(PredictionResult):
    """Age bracket prediction outcome."""

    prediction: AgeBracketEnum = Field(
        ..., description="Age bracket label: 18-30, 31-45, 46-60, 60+, or unknown"
    )


class AnalyzeResponse(BaseModel):
    """Response returned by the /analyze endpoint."""

    contact_id: UUID = Field(..., description="Unique caller identifier")
    gender: GenderResult = Field(..., description="Gender prediction outcome")
    age_bracket: AgeResult = Field(..., description="Age bracket prediction outcome")
    processing_ms: float = Field(
        ..., description="End-to-end processing latency in milliseconds"
    )
    audio_quality: AudioQualityEnum = Field(
        ..., description="Evaluated audio quality: good, degraded, or insufficient"
    )

    @field_validator("processing_ms")
    @classmethod
    def round_processing_ms(cls, v: float) -> float:
        return round(v, 2)


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Service health status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded in memory")
    version: str = Field(..., description="Service version string")


class StreamPredictionMessage(BaseModel):
    """WebSocket streaming prediction message."""

    type: Literal["partial_prediction", "final_prediction", "error"] = Field(
        ..., description="Message type"
    )
    gender: Optional[GenderResult] = Field(None, description="Gender prediction")
    age_bracket: Optional[AgeResult] = Field(None, description="Age bracket prediction")
    audio_quality: Optional[AudioQualityEnum] = Field(
        None, description="Audio quality classification"
    )
    is_final: bool = Field(
        default=False, description="Whether this is the final prediction"
    )
    processing_ms: Optional[float] = Field(
        None, description="Processing time for this chunk/session in milliseconds"
    )
    error: Optional[str] = Field(None, description="Error message if type is error")
