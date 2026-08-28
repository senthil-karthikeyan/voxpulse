"""Configuration settings for VoxPulse Voice Attribute Inference Service."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and ML inference configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    APP_NAME: str = "VoxPulse Voice Attribute Inference Service"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Audio Normalization Settings
    TARGET_SAMPLE_RATE: int = 16000
    MAX_AUDIO_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MIN_AUDIO_DURATION_SECONDS: float = 0.5
    MAX_AUDIO_DURATION_SECONDS: float = 300.0  # 5 minutes

    # Audio Quality & VAD Pipeline Thresholds
    MIN_SPEECH_DURATION_SECONDS: float = 2.0
    MIN_SPEECH_RATIO: float = 0.35
    MIN_RMS_ENERGY: float = 0.005
    MAX_CLIPPING_RATIO: float = 0.05
    MIN_SNR_DB: float = 6.0
    MAX_SILENCE_RATIO: float = 0.65

    # Inference & Model Settings
    GENDER_CONFIDENCE_THRESHOLD: float = 0.60
    AGE_CONFIDENCE_THRESHOLD: float = 0.40
    DEVICE: str = "cpu"
    MODEL_WEIGHTS_DIR: Path = Path("model_weights")
    SPEECHBRAIN_CACHE_DIR: Path = Path("pretrained_models")
    ENABLE_SPEECHBRAIN_DOWNLOAD: bool = True
    REQUIRE_TRAINED_MODELS: bool = False

    # Streaming / WebSocket settings
    STREAM_CHUNK_BYTES: int = 4096
    STREAM_MIN_INFERENCE_SECONDS: float = 1.5
    STREAM_INFERENCE_INTERVAL_SECONDS: float = 1.5
    STREAM_MAX_BUFFER_SECONDS: float = 60.0


settings = Settings()
