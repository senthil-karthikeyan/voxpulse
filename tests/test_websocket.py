"""Tests for WebSocket streaming voice attribute inference."""

import json
from uuid import uuid4
import numpy as np
from fastapi.testclient import TestClient


def test_websocket_streaming_flow(client: TestClient) -> None:
    """Test full WebSocket streaming flow: connect, send chunks, finalize, disconnect."""
    contact_id = str(uuid4())

    # Generate 2 seconds of 16kHz 16-bit PCM synthetic audio
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    waveform = (0.6 * np.sin(2 * np.pi * 180.0 * t) * 32767).astype(np.int16)
    pcm_bytes = waveform.tobytes()

    with client.websocket_connect(f"/stream/analyze/{contact_id}") as ws:
        # Stream in 0.5-second chunks
        chunk_size = sr * 2 // 2  # 16000 bytes = 0.5s
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            ws.send_bytes(chunk)

        # Send finalize control command
        ws.send_text(json.dumps({"action": "finalize"}))

        # Receive messages until final prediction
        final_received = False
        for _ in range(5):
            data = ws.receive_json()
            if data["type"] == "final_prediction":
                assert data["is_final"] is True
                assert "audio_quality" in data
                assert data["audio_quality"] in ["good", "degraded", "insufficient"]
                final_received = True
                break

        assert final_received, "Did not receive final prediction message"


def test_websocket_control_messages(client: TestClient) -> None:
    """Test WebSocket ping and reset commands."""
    contact_id = str(uuid4())

    with client.websocket_connect(f"/stream/analyze/{contact_id}") as ws:
        # Test ping
        ws.send_text(json.dumps({"action": "ping"}))
        resp = ws.receive_json()
        assert resp["type"] == "pong"

        # Test reset
        ws.send_text(json.dumps({"action": "reset"}))
        resp2 = ws.receive_json()
        assert resp2["type"] == "info"
