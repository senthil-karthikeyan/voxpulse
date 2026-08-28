"""WebSocket streaming endpoint for real-time voice attribute inference."""

import json
import time
from uuid import UUID
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging import logger, mask_contact_id
from app.schemas.response import AudioQualityEnum, StreamPredictionMessage
from app.services.inference_service import inference_service
from app.utils.timing import StageTimer

router = APIRouter()


@router.websocket("/stream/analyze/{contact_id}")
async def websocket_stream_analyze(websocket: WebSocket, contact_id: UUID) -> None:
    """Stream audio chunks over WebSocket and receive progressive attribute predictions.

    Protocol:
    - Client sends binary audio frames (16kHz 16-bit PCM or encoded audio chunks) or text JSON control frames.
    - Server emits StreamPredictionMessage ('partial_prediction' or 'final_prediction').
    - In-memory buffer is discarded immediately upon disconnect.
    """
    await websocket.accept()
    masked_id = mask_contact_id(contact_id)
    logger.info(
        "WebSocket streaming connection established.",
        extra={"event": "stream_connected", "contact_id_hash": masked_id},
    )

    # In-memory buffer allocated strictly for the duration of this session
    audio_buffer = bytearray()
    last_inference_time = time.perf_counter()
    samples_per_second = settings.TARGET_SAMPLE_RATE  # 16000 samples / sec
    # 16-bit PCM: 2 bytes per sample -> 32000 bytes/sec
    bytes_per_second = samples_per_second * 2
    min_inference_bytes = int(settings.STREAM_MIN_INFERENCE_SECONDS * bytes_per_second)
    max_buffer_bytes = int(settings.STREAM_MAX_BUFFER_SECONDS * bytes_per_second)

    try:
        while True:
            message = await websocket.receive()

            # Handle binary audio frame
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)

                # Prevent unbounded memory growth if client streams excessively long audio
                if len(audio_buffer) > max_buffer_bytes:
                    audio_buffer = bytearray(audio_buffer[-max_buffer_bytes:])

                # Check if we have accumulated enough audio and time interval has elapsed
                now = time.perf_counter()
                elapsed_since_last = now - last_inference_time

                if (
                    len(audio_buffer) >= min_inference_bytes
                    and elapsed_since_last >= settings.STREAM_INFERENCE_INTERVAL_SECONDS
                ):
                    last_inference_time = now
                    timer = StageTimer()
                    raw_bytes = bytes(audio_buffer)

                    # Convert 16-bit PCM buffer to Float32 normalized numpy array
                    try:
                        pcm_array = np.frombuffer(raw_bytes, dtype=np.int16).copy()
                        waveform = (pcm_array.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                    except Exception:
                        try:
                            waveform, _ = inference_service.processor.decode_and_normalize(raw_bytes)
                        except Exception as e:
                            logger.warning(f"Could not decode streaming audio chunk: {e}")
                            continue

                    try:
                        analysis = inference_service.analyze_waveform(
                            waveform, contact_id=contact_id, timer=timer
                        )
                        partial_msg = StreamPredictionMessage(
                            type="partial_prediction",
                            gender=analysis.gender,
                            age_bracket=analysis.age_bracket,
                            audio_quality=analysis.audio_quality,
                            is_final=False,
                            processing_ms=analysis.processing_ms,
                        )
                        await websocket.send_text(partial_msg.model_dump_json())
                    except Exception as e:
                        logger.error(f"Error during partial inference: {e}")

            # Handle text / JSON control messages (e.g. {"action": "finalize"})
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action", "").lower()

                    if action == "finalize":
                        # Compute final prediction
                        timer = StageTimer()
                        raw_bytes = bytes(audio_buffer)
                        if len(raw_bytes) > 0:
                            try:
                                pcm_array = np.frombuffer(raw_bytes, dtype=np.int16).copy()
                                waveform = (pcm_array.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                            except Exception:
                                waveform, _ = inference_service.processor.decode_and_normalize(raw_bytes)

                            analysis = inference_service.analyze_waveform(
                                waveform, contact_id=contact_id, timer=timer
                            )
                            final_msg = StreamPredictionMessage(
                                type="final_prediction",
                                gender=analysis.gender,
                                age_bracket=analysis.age_bracket,
                                audio_quality=analysis.audio_quality,
                                is_final=True,
                                processing_ms=analysis.processing_ms,
                            )
                        else:
                            final_msg = StreamPredictionMessage(
                                type="final_prediction",
                                audio_quality=AudioQualityEnum.INSUFFICIENT,
                                is_final=True,
                                processing_ms=0.0,
                            )
                        await websocket.send_text(final_msg.model_dump_json())
                        break

                    elif action == "reset":
                        audio_buffer.clear()
                        await websocket.send_text(
                            json.dumps({"type": "info", "message": "Buffer reset"})
                        )

                    elif action == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))

                except json.JSONDecodeError:
                    await websocket.send_text(
                        StreamPredictionMessage(
                            type="error",
                            error="Invalid JSON control message",
                        ).model_dump_json()
                    )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected normally.",
            extra={"event": "stream_disconnected", "contact_id_hash": masked_id},
        )
    except Exception as e:
        logger.error(
            f"WebSocket error: {e}",
            exc_info=True,
            extra={"event": "stream_error", "contact_id_hash": masked_id},
        )
    finally:
        # Zero storage guarantee: explicitly purge audio buffer from memory
        audio_buffer.clear()
        del audio_buffer
        logger.info(
            "Streaming session ended and ephemeral memory purged.",
            extra={"event": "stream_cleanup", "contact_id_hash": masked_id},
        )
