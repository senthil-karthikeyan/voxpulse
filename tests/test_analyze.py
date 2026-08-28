"""Tests for the POST /analyze endpoint."""

from fastapi.testclient import TestClient


def test_analyze_valid_audio(
    client: TestClient, valid_contact_id: str, clean_audio_wav: bytes
) -> None:
    """Test analyzing valid audio payload returns complete demographic response."""
    response = client.post(
        "/analyze",
        data={"contact_id": valid_contact_id},
        files={"audio": ("test.wav", clean_audio_wav, "audio/wav")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["contact_id"] == valid_contact_id

    # Gender assertion
    assert "gender" in data
    assert data["gender"]["prediction"] in ["male", "female", "unknown"]
    assert 0.0 <= data["gender"]["confidence"] <= 1.0

    # Age bracket assertion
    assert "age_bracket" in data
    assert data["age_bracket"]["prediction"] in [
        "18-30",
        "31-45",
        "46-60",
        "60+",
        "unknown",
    ]
    assert 0.0 <= data["age_bracket"]["confidence"] <= 1.0

    # Audio quality assertion
    assert data["audio_quality"] in ["good", "degraded", "insufficient"]
    assert data["audio_quality"] == "good"

    # Latency tracking
    assert "processing_ms" in data
    assert data["processing_ms"] > 0.0


def test_analyze_insufficient_audio_duration(
    client: TestClient, valid_contact_id: str, short_insufficient_audio_wav: bytes
) -> None:
    """Test very short audio is flagged as insufficient and demographic predictions are unknown."""
    response = client.post(
        "/analyze",
        data={"contact_id": valid_contact_id},
        files={"audio": ("short.wav", short_insufficient_audio_wav, "audio/wav")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["audio_quality"] == "insufficient"
    assert data["gender"]["prediction"] == "unknown"
    assert data["gender"]["confidence"] == 0.0
    assert data["age_bracket"]["prediction"] == "unknown"
    assert data["age_bracket"]["confidence"] == 0.0


def test_analyze_silent_audio(
    client: TestClient, valid_contact_id: str, silent_audio_wav: bytes
) -> None:
    """Test pure silence is flagged as insufficient with unknown predictions."""
    response = client.post(
        "/analyze",
        data={"contact_id": valid_contact_id},
        files={"audio": ("silent.wav", silent_audio_wav, "audio/wav")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["audio_quality"] == "insufficient"
    assert data["gender"]["prediction"] == "unknown"
    assert data["age_bracket"]["prediction"] == "unknown"


def test_analyze_degraded_clipped_audio(
    client: TestClient, valid_contact_id: str, clipped_audio_wav: bytes
) -> None:
    """Test clipped/distorted audio is detected as degraded."""
    response = client.post(
        "/analyze",
        data={"contact_id": valid_contact_id},
        files={"audio": ("clipped.wav", clipped_audio_wav, "audio/wav")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["audio_quality"] in ["degraded", "good"]


def test_analyze_empty_audio_upload(
    client: TestClient, valid_contact_id: str
) -> None:
    """Test uploading an empty audio file returns HTTP 400 Bad Request."""
    response = client.post(
        "/analyze",
        data={"contact_id": valid_contact_id},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_invalid_audio_bytes(
    client: TestClient, valid_contact_id: str, corrupt_audio_bytes: bytes
) -> None:
    """Test corrupted non-audio bytes return HTTP 400 Bad Request safely."""
    response = client.post(
        "/analyze",
        data={"contact_id": valid_contact_id},
        files={"audio": ("corrupt.wav", corrupt_audio_bytes, "audio/wav")},
    )
    assert response.status_code == 400
    assert "error" in response.json()["detail"].lower()


def test_analyze_invalid_uuid(client: TestClient, clean_audio_wav: bytes) -> None:
    """Test submitting an invalid UUID string triggers a 422 Unprocessable Entity."""
    response = client.post(
        "/analyze",
        data={"contact_id": "not-a-valid-uuid"},
        files={"audio": ("test.wav", clean_audio_wav, "audio/wav")},
    )
    assert response.status_code == 422
