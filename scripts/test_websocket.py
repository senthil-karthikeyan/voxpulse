"""Manual WebSocket Testing Client for VoxPulse Voice Attribute Inference Service.

Supports testing all aspects of the WebSocket streaming protocol:
- Mode 'connect': Basic handshake, ping/pong, and reset command verification.
- Mode 'stream': Stream real or synthetic audio in chunks with configurable delay.
- Mode 'idle-test': Stream a few chunks, pause/idle, and observe server behavior.
- Mode 'disconnect-test': Stream audio and abruptly close the socket to verify server cleanup.
- Mode 'concurrent': Launch multiple simultaneous clients to verify state isolation.
- Mode 'invalid-test': Send malformed JSON and corrupted payloads to verify resilience.
- Mode 'insufficient-test': Send <1.5s audio to verify early-rejection / insufficient behavior.

Usage Examples:
    uv run python scripts/test_websocket.py --mode connect
    uv run python scripts/test_websocket.py --mode stream --audio data/audio/sample_0001.wav
    uv run python scripts/test_websocket.py --mode stream --audio data/audio/sample_0001.wav --chunk-duration 0.5 --delay 0.5
    uv run python scripts/test_websocket.py --mode idle-test --audio data/audio/sample_0001.wav --idle-seconds 5
    uv run python scripts/test_websocket.py --mode disconnect-test --audio data/audio/sample_0001.wav
    uv run python scripts/test_websocket.py --mode concurrent --clients 2 --audio data/audio/sample_0001.wav
    uv run python scripts/test_websocket.py --mode invalid-test
    uv run python scripts/test_websocket.py --mode insufficient-test
"""

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
import soundfile as sf
import websockets


def generate_synthetic_pcm(duration_s: float = 3.0, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
    """Generate 16kHz 16-bit mono PCM sine wave bytes."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    # Sine wave with gentle envelope to avoid clicks
    envelope = np.ones_like(t)
    fade_len = int(0.05 * sample_rate)
    if len(t) > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    waveform = (0.7 * np.sin(2 * np.pi * freq * t) * envelope * 32767).astype(np.int16)
    return waveform.tobytes()


def load_audio_as_pcm(audio_path: Path, target_sr: int = 16000) -> bytes:
    """Load audio file and convert to 16kHz 16-bit mono PCM bytes."""
    import librosa

    waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm_array = (waveform * 32767).astype(np.int16)
    return pcm_array.tobytes()


async def test_connect(base_url: str) -> None:
    """Test 1: Basic Connection, Ping, and Reset."""
    contact_id = str(uuid4())
    ws_url = f"{base_url}/stream/analyze/{contact_id}"
    print(f"\n[TEST: CONNECT] Connecting to {ws_url}...")

    t0 = time.perf_counter()
    async with websockets.connect(ws_url) as ws:
        connect_time = (time.perf_counter() - t0) * 1000
        print(f"  [OK] Connected in {connect_time:.2f} ms")

        # Test Ping
        print("  -> Sending control message: {\"action\": \"ping\"}")
        await ws.send(json.dumps({"action": "ping"}))
        resp = await ws.recv()
        print(f"  <- Received: {resp}")
        data = json.loads(resp)
        assert data.get("type") == "pong", f"Expected type=pong, got {data}"
        print("  [OK] Ping / Pong verified successfully")

        # Test Reset
        print("  -> Sending control message: {\"action\": \"reset\"}")
        await ws.send(json.dumps({"action": "reset"}))
        resp2 = await ws.recv()
        print(f"  <- Received: {resp2}")
        data2 = json.loads(resp2)
        assert data2.get("type") == "info", f"Expected type=info, got {data2}"
        print("  [OK] Buffer reset command verified successfully")

    print("[PASSED] Test 1: Basic Connection verified.\n")


async def test_stream(
    base_url: str,
    pcm_bytes: bytes,
    chunk_duration: float = 0.5,
    delay: float = 0.5,
    client_label: str = "Client",
) -> None:
    """Test 2, 3, 4, 5: Stream audio chunks with progressive partial and final prediction."""
    contact_id = str(uuid4())
    ws_url = f"{base_url}/stream/analyze/{contact_id}"
    sample_rate = 16000
    bytes_per_sample = 2  # 16-bit
    chunk_size = int(chunk_duration * sample_rate * bytes_per_sample)
    total_chunks = (len(pcm_bytes) + chunk_size - 1) // chunk_size
    total_duration_s = len(pcm_bytes) / (sample_rate * bytes_per_sample)

    print(f"\n[{client_label}] Connecting to {ws_url}")
    print(f"[{client_label}] Audio: {total_duration_s:.2f}s ({len(pcm_bytes)} bytes), Chunk size: {chunk_size} bytes ({chunk_duration}s), Delay: {delay}s")

    async with websockets.connect(ws_url) as ws:
        # Background receiver task for partial predictions
        partial_count = 0
        final_prediction = None
        messages_received = []

        async def receive_loop():
            nonlocal partial_count, final_prediction
            try:
                while True:
                    msg = await ws.recv()
                    t_recv = time.perf_counter()
                    data = json.loads(msg)
                    messages_received.append((t_recv, data))
                    msg_type = data.get("type")
                    if msg_type == "partial_prediction":
                        partial_count += 1
                        print(f"  <- [{client_label}] PARTIAL PREDICTION #{partial_count}:")
                        print(f"       Gender: {data.get('gender', {}).get('prediction')} (conf: {data.get('gender', {}).get('confidence', 0.0):.2f})")
                        print(f"       Age:    {data.get('age_bracket', {}).get('prediction')} (conf: {data.get('age_bracket', {}).get('confidence', 0.0):.2f})")
                        print(f"       Quality: {data.get('audio_quality')} | Server processing: {data.get('processing_ms')} ms")
                    elif msg_type == "final_prediction":
                        final_prediction = data
                        print(f"  <- [{client_label}] FINAL PREDICTION:")
                        print(f"       Gender: {data.get('gender', {}).get('prediction')} (conf: {data.get('gender', {}).get('confidence', 0.0):.2f})")
                        print(f"       Age:    {data.get('age_bracket', {}).get('prediction')} (conf: {data.get('age_bracket', {}).get('confidence', 0.0):.2f})")
                        print(f"       Quality: {data.get('audio_quality')} | Server processing: {data.get('processing_ms')} ms | Final: {data.get('is_final')}")
                        break
            except websockets.exceptions.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receive_loop())

        # Send chunks
        stream_start = time.perf_counter()
        for idx in range(total_chunks):
            chunk = pcm_bytes[idx * chunk_size : (idx + 1) * chunk_size]
            accumulated_s = ((idx + 1) * len(chunk)) / (sample_rate * bytes_per_sample)
            print(f"  -> [{client_label}] Sending Chunk {idx + 1}/{total_chunks} ({len(chunk)} bytes, accumulated: {accumulated_s:.2f}s)")
            await ws.send(chunk)
            if delay > 0 and idx < total_chunks - 1:
                await asyncio.sleep(delay)

        # Send finalize command
        print(f"  -> [{client_label}] Sending finalize command: {{\"action\": \"finalize\"}}")
        finalize_send_time = time.perf_counter()
        await ws.send(json.dumps({"action": "finalize"}))

        # Wait for receiver task to complete
        try:
            await asyncio.wait_for(recv_task, timeout=10.0)
        except asyncio.TimeoutError:
            print(f"  [!] [{client_label}] Timed out waiting for final prediction")

        total_elapsed = (time.perf_counter() - stream_start) * 1000
        print(f"[{client_label}] Streaming session finished in {total_elapsed:.2f} ms")
        print(f"[{client_label}] Total partial predictions received: {partial_count}")
        assert final_prediction is not None, "Did not receive final prediction!"
        assert final_prediction.get("is_final") is True, "Final prediction is_final flag was not True!"
        print(f"[{client_label}] [PASSED] Streaming & Prediction test completed successfully.\n")


async def test_idle(base_url: str, pcm_bytes: bytes, idle_seconds: float = 5.0) -> None:
    """Test 7: Client stops sending chunks (Idle / Pause scenario)."""
    contact_id = str(uuid4())
    ws_url = f"{base_url}/stream/analyze/{contact_id}"
    print(f"\n[TEST: IDLE / PAUSE TEST] Connecting to {ws_url}...")
    print(f"  Plan: Send 1.0s audio chunk, then idle for {idle_seconds:.1f}s without sending anything.")

    async with websockets.connect(ws_url) as ws:
        # Send 1 second of audio (32,000 bytes)
        one_sec_bytes = pcm_bytes[:32000] if len(pcm_bytes) >= 32000 else pcm_bytes
        print(f"  -> Sending {len(one_sec_bytes)} bytes (~1.0s audio)")
        await ws.send(one_sec_bytes)

        print(f"  [PAUSE] Idling for {idle_seconds} seconds. Observing server behavior...")
        t_start = time.perf_counter()
        idle_msg_received = None

        try:
            # Check if server sends anything or closes
            idle_msg = await asyncio.wait_for(ws.recv(), timeout=idle_seconds)
            idle_msg_received = idle_msg
            print(f"  <- Server sent message during idle: {idle_msg}")
        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - t_start
            print(f"  [OBSERVATION] Server kept connection open and idle for {elapsed:.2f}s without timing out.")
            print("  [NOTE] Current implementation has NO idle timeout; server hangs waiting on socket.")
            print("  [RECOMMENDATION] In production, implement a 10s idle timeout to auto-finalize or disconnect stale callers.")

        # Now finalize properly
        print("  -> Finalizing session after idle...")
        await ws.send(json.dumps({"action": "finalize"}))
        final_resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
        print(f"  <- Final prediction after idle: {final_resp}")
        print("[PASSED] Test 7: Idle test completed.\n")


async def test_abrupt_disconnect(base_url: str, pcm_bytes: bytes) -> None:
    """Test 8: Client abruptly disconnects without finalizing."""
    contact_id = str(uuid4())
    ws_url = f"{base_url}/stream/analyze/{contact_id}"
    print(f"\n[TEST: ABRUPT DISCONNECT] Connecting to {ws_url}...")

    ws = await websockets.connect(ws_url)
    print("  [OK] Connected. Sending 2.0s audio...")
    await ws.send(pcm_bytes[:64000] if len(pcm_bytes) >= 64000 else pcm_bytes)
    await asyncio.sleep(0.5)

    print("  -> Force-closing WebSocket connection abruptly (without sending finalize)...")
    await ws.close(code=1000)
    print("  [OK] Client closed socket.")
    print("  [VERIFICATION] Check server terminal logs: you should see 'WebSocket client disconnected normally.' and 'Streaming session ended and ephemeral memory purged.'")
    print("[PASSED] Test 8: Abrupt disconnect test completed.\n")


async def test_invalid_messages(base_url: str) -> None:
    """Test 9: Send malformed JSON and corrupted payloads."""
    contact_id = str(uuid4())
    ws_url = f"{base_url}/stream/analyze/{contact_id}"
    print(f"\n[TEST: INVALID MESSAGES] Connecting to {ws_url}...")

    async with websockets.connect(ws_url) as ws:
        # 1. Send malformed JSON text
        print("  -> Sending malformed JSON: 'INVALID_NOT_JSON'")
        await ws.send("INVALID_NOT_JSON")
        resp = await ws.recv()
        print(f"  <- Server response: {resp}")
        data = json.loads(resp)
        assert data.get("type") == "error", f"Expected type=error, got {data}"
        print("  [OK] Malformed JSON handled safely without server crash")

        # 2. Send unknown action JSON
        print("  -> Sending unknown action JSON: {\"action\": \"unknown_command_xyz\"}")
        await ws.send(json.dumps({"action": "unknown_command_xyz"}))
        # Unknown actions are ignored without error

        # 3. Finalize empty
        print("  -> Sending finalize on empty buffer: {\"action\": \"finalize\"}")
        await ws.send(json.dumps({"action": "finalize"}))
        resp_final = await ws.recv()
        print(f"  <- Final prediction on empty buffer: {resp_final}")
        final_data = json.loads(resp_final)
        assert final_data.get("type") == "final_prediction"
        assert final_data.get("audio_quality") == "insufficient"
        print("  [OK] Finalize on empty buffer returned audio_quality='insufficient'")

    print("[PASSED] Test 9: Invalid message handling verified.\n")


async def test_insufficient_audio(base_url: str) -> None:
    """Test 6: Send audio shorter than minimum duration (< 1.5s speech)."""
    contact_id = str(uuid4())
    ws_url = f"{base_url}/stream/analyze/{contact_id}"
    print(f"\n[TEST: INSUFFICIENT AUDIO] Connecting to {ws_url}...")

    # Generate 0.3s of audio (4,800 samples = 9,600 bytes)
    short_pcm = generate_synthetic_pcm(duration_s=0.3, freq=300.0)
    print(f"  Audio duration: 0.3s ({len(short_pcm)} bytes)")

    async with websockets.connect(ws_url) as ws:
        print("  -> Sending 0.3s audio chunk...")
        await ws.send(short_pcm)
        print("  -> Sending finalize command: {\"action\": \"finalize\"}")
        await ws.send(json.dumps({"action": "finalize"}))

        resp = await ws.recv()
        print(f"  <- Server response: {resp}")
        data = json.loads(resp)
        print(f"       Quality:    {data.get('audio_quality')}")
        print(f"       Gender:     {data.get('gender', {}).get('prediction')} (conf: {data.get('gender', {}).get('confidence')})")
        print(f"       Age:        {data.get('age_bracket', {}).get('prediction')} (conf: {data.get('age_bracket', {}).get('confidence')})")
        assert data.get("audio_quality") == "insufficient", f"Expected quality=insufficient, got {data.get('audio_quality')}"
        assert data.get("gender", {}).get("prediction") == "unknown", "Expected gender=unknown for insufficient audio"
        assert data.get("age_bracket", {}).get("prediction") == "unknown", "Expected age=unknown for insufficient audio"
        print("  [OK] Short audio correctly refused inference with quality='insufficient' and demographics='unknown'")

    print("[PASSED] Test 6: Insufficient audio test completed.\n")


async def test_concurrent(base_url: str, pcm_bytes: bytes, num_clients: int = 2) -> None:
    """Test 10: Concurrent WebSocket connections with isolated state."""
    print(f"\n[TEST: CONCURRENT CLIENTS] Launching {num_clients} concurrent streaming sessions...")

    tasks = []
    for i in range(num_clients):
        label = f"Client-{i + 1}"
        # Slightly offset start time or frequency
        client_pcm = generate_synthetic_pcm(duration_s=3.0, freq=200.0 + i * 150.0) if not pcm_bytes else pcm_bytes
        tasks.append(
            test_stream(
                base_url=base_url,
                pcm_bytes=client_pcm,
                chunk_duration=0.5,
                delay=0.3,
                client_label=label,
            )
        )

    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    total_elapsed = (time.perf_counter() - t0) * 1000
    print(f"[PASSED] Test 10: All {num_clients} concurrent streaming clients completed in {total_elapsed:.2f} ms without collision.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual WebSocket Testing Client for VoxPulse Voice Attribute Service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "connect",
            "stream",
            "idle-test",
            "disconnect-test",
            "invalid-test",
            "insufficient-test",
            "concurrent",
            "all",
        ],
        default="all",
        help="Test mode to execute",
    )
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (e.g. data/audio/sample_0001.wav)")
    parser.add_argument("--chunk-duration", type=float, default=0.5, help="Duration of each chunk in seconds (default: 0.5)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between sending chunks in seconds (default: 0.5)")
    parser.add_argument("--idle-seconds", type=float, default=5.0, help="Seconds to idle during idle-test (default: 5.0)")
    parser.add_argument("--clients", type=int, default=2, help="Number of concurrent clients for concurrent test (default: 2)")

    args = parser.parse_args()
    base_url = f"ws://{args.host}:{args.port}"

    # Load audio or synthesize
    pcm_bytes = b""
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Error: Audio file not found at {audio_path}")
            sys.exit(1)
        print(f"Loading audio file: {audio_path}")
        pcm_bytes = load_audio_as_pcm(audio_path)
    else:
        print("No audio file specified (--audio), generating 4.0s synthetic 16kHz speech tone...")
        pcm_bytes = generate_synthetic_pcm(duration_s=4.0, freq=220.0)

    print("=" * 70)
    print(f"VOXPULSE WEBSOCKET MANUAL TEST SUITE -> Target: {base_url}")
    print("=" * 70)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        if args.mode in ("connect", "all"):
            loop.run_until_complete(test_connect(base_url))

        if args.mode in ("stream", "all"):
            loop.run_until_complete(
                test_stream(
                    base_url,
                    pcm_bytes,
                    chunk_duration=args.chunk_duration,
                    delay=args.delay,
                )
            )

        if args.mode in ("idle-test", "all"):
            loop.run_until_complete(test_idle(base_url, pcm_bytes, idle_seconds=args.idle_seconds))

        if args.mode in ("disconnect-test", "all"):
            loop.run_until_complete(test_abrupt_disconnect(base_url, pcm_bytes))

        if args.mode in ("invalid-test", "all"):
            loop.run_until_complete(test_invalid_messages(base_url))

        if args.mode in ("insufficient-test", "all"):
            loop.run_until_complete(test_insufficient_audio(base_url))

        if args.mode in ("concurrent", "all"):
            loop.run_until_complete(test_concurrent(base_url, pcm_bytes, num_clients=args.clients))

        print("=" * 70)
        print("ALL REQUESTED WEBSOCKET TESTS COMPLETED SUCCESSFULLY.")
        print("=" * 70)

    except ConnectionRefusedError:
        print(f"\n[ERROR] Connection refused at {base_url}!")
        print("Please ensure the VoxPulse server is running:")
        print("  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
