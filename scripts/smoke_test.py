"""Smoke test script for VoxPulse Voice Attribute Inference Service.

Demonstrates calling both REST and WebSocket endpoints using synthetic or local audio.

Usage:
    uv run python scripts/smoke_test.py [--url http://localhost:8000] [--file path/to/audio.wav]
"""

import argparse
import asyncio
import io
import json
import sys
import time
import uuid
import numpy as np
import soundfile as sf
import httpx
import websockets


def generate_sample_wav(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate a clean synthetic harmonic voice-like audio in memory."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    # Fundamental pitch at 180 Hz with speech harmonics and envelope
    waveform = (
        0.50 * np.sin(2 * np.pi * 180.0 * t)
        + 0.25 * np.sin(2 * np.pi * 360.0 * t)
        + 0.15 * np.sin(2 * np.pi * 540.0 * t)
    )
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))
    audio_data = (waveform * envelope * 0.7).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_health(base_url: str) -> bool:
    """Check GET /health."""
    print(f"\n[1/3] Testing GET {base_url}/health ...")
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{base_url}/health")
            elapsed = (time.perf_counter() - start) * 1000.0

        if resp.status_code == 200:
            data = resp.json()
            print(f"  [PASS] Health Status: {data.get('status')}")
            print(f"  [PASS] Models Loaded: {data.get('models_loaded')}")
            print(f"  [PASS] Version: {data.get('version')}")
            print(f"  [PASS] Latency: {elapsed:.2f} ms")
            return True
        else:
            print(f"  [FAIL] Health check failed with status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] Failed to connect to {base_url}/health: {e}")
        return False


def test_analyze(base_url: str, audio_bytes: bytes) -> bool:
    """Check POST /analyze."""
    print(f"\n[2/3] Testing POST {base_url}/analyze ...")
    contact_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{base_url}/analyze",
                data={"contact_id": contact_id},
                files={"audio": ("smoke_sample.wav", audio_bytes, "audio/wav")},
            )
            elapsed = (time.perf_counter() - start) * 1000.0

        if resp.status_code == 200:
            data = resp.json()
            print(f"  [PASS] Contact ID: {data.get('contact_id')}")
            print(f"  [PASS] Gender: {data.get('gender', {}).get('prediction')} (confidence: {data.get('gender', {}).get('confidence'):.2%})")
            print(f"  [PASS] Age Bracket: {data.get('age_bracket', {}).get('prediction')} (confidence: {data.get('age_bracket', {}).get('confidence'):.2%})")
            print(f"  [PASS] Audio Quality: {data.get('audio_quality')}")
            print(f"  [PASS] Server Processing Time: {data.get('processing_ms')} ms")
            print(f"  [PASS] Total Round-Trip: {elapsed:.2f} ms")
            return True
        else:
            print(f"  [FAIL] Analysis failed with status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] Analysis request failed: {e}")
        return False


async def test_websocket_stream(ws_url: str, audio_bytes: bytes) -> bool:
    """Check WS /stream/analyze/{contact_id}."""
    contact_id = str(uuid.uuid4())
    target_url = f"{ws_url}/stream/analyze/{contact_id}"
    print(f"\n[3/3] Testing WebSocket {target_url} ...")

    try:
        async with websockets.connect(target_url) as ws:
            # Send audio in 16KB chunks
            chunk_size = 16000
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.05)

            # Finalize stream
            await ws.send(json.dumps({"action": "finalize"}))
            response = await ws.recv()
            data = json.loads(response)

            print(f"  [PASS] Stream Type: {data.get('type')}")
            print(f"  [PASS] Is Final: {data.get('is_final')}")
            print(f"  [PASS] Audio Quality: {data.get('audio_quality')}")
            if data.get("gender"):
                print(f"  [PASS] Gender: {data['gender'].get('prediction')} (conf: {data['gender'].get('confidence')})")
            if data.get("age_bracket"):
                print(f"  [PASS] Age Bracket: {data['age_bracket'].get('prediction')} (conf: {data['age_bracket'].get('confidence')})")
            return True
    except Exception as e:
        print(f"  [FAIL] WebSocket streaming test failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxPulse Voice Attribute Service Smoke Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base HTTP URL")
    parser.add_argument("--file", default=None, help="Path to local audio file (optional)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")

    if args.file:
        print(f"Loading user audio file from {args.file}...")
        with open(args.file, "rb") as f:
            audio_bytes = f.read()
    else:
        print("Generating 3.0-second synthetic voice audio for testing...")
        audio_bytes = generate_sample_wav()

    h_ok = test_health(base_url)
    if not h_ok:
        print("\nHealth check failed! Ensure server is running (`uv run uvicorn app.main:app`).")
        sys.exit(1)

    a_ok = test_analyze(base_url, audio_bytes)
    ws_ok = asyncio.run(test_websocket_stream(ws_url, audio_bytes))

    print("\n" + "=" * 50)
    if h_ok and a_ok and ws_ok:
        print("ALL SMOKE TESTS PASSED SUCCESSFULLY!")
        print("=" * 50)
        sys.exit(0)
    else:
        print("SOME SMOKE TESTS FAILED!")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
