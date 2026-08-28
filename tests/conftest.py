"""Pytest test fixtures and synthetic audio generators."""

import io
import uuid
from typing import Generator
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app


def generate_synthetic_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 180.0,
    amplitude: float = 0.5,
    noise_level: float = 0.01,
    clipping: bool = False,
) -> bytes:
    """Generate in-memory WAV audio bytes with harmonic structure."""
    num_samples = int(duration * sample_rate)
    if num_samples == 0:
        return b""

    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Base harmonic series imitating vocal cords
    waveform = (
        0.50 * np.sin(2 * np.pi * frequency * t)
        + 0.25 * np.sin(2 * np.pi * frequency * 2 * t)
        + 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
        + 0.10 * np.sin(2 * np.pi * frequency * 4 * t)
    )

    # Apply speech-like amplitude modulation (syllables)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))
    waveform = waveform * envelope * amplitude

    # Add Gaussian background noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, num_samples)
        waveform = waveform + noise

    if clipping:
        waveform = np.clip(waveform * 10.0, -1.0, 1.0)
    else:
        waveform = np.clip(waveform, -0.95, 0.95)

    buffer = io.BytesIO()
    sf.write(buffer, waveform.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """TestClient fixture with application lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_contact_id() -> str:
    """Random valid UUID string."""
    return str(uuid.uuid4())


@pytest.fixture
def clean_audio_wav() -> bytes:
    """3-second clean speech-like synthetic WAV file."""
    return generate_synthetic_audio(duration=3.0, frequency=190.0, noise_level=0.005)


@pytest.fixture
def short_insufficient_audio_wav() -> bytes:
    """0.1-second audio file, below minimum duration threshold."""
    return generate_synthetic_audio(duration=0.1)


@pytest.fixture
def silent_audio_wav() -> bytes:
    """2.0-second silent audio file."""
    return generate_synthetic_audio(duration=2.0, amplitude=0.0, noise_level=0.0)


@pytest.fixture
def clipped_audio_wav() -> bytes:
    """2.5-second heavily clipped/saturated audio file."""
    return generate_synthetic_audio(duration=2.5, amplitude=1.0, clipping=True)


@pytest.fixture
def corrupt_audio_bytes() -> bytes:
    """Invalid non-audio binary payload."""
    return b"RIFF....CORRUPTED_NON_AUDIO_PAYLOAD_1234567890"
