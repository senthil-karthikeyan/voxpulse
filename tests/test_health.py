"""Tests for the GET /health check endpoint."""

from fastapi.testclient import TestClient


def test_health_check_returns_200_and_healthy(client: TestClient) -> None:
    """Verify health endpoint returns status 200 and expected health schema."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], bool)
    assert data["models_loaded"] is True
    assert "version" in data
