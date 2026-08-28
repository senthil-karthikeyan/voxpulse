"""Structured JSON logging with privacy protection and event tracking."""

import hashlib
import json
import logging
import sys
import time
from typing import Any, Dict, Optional
from uuid import UUID


def mask_contact_id(contact_id: Optional[UUID | str]) -> Optional[str]:
    """Return a privacy-safe truncated SHA-256 hash of a contact ID."""
    if contact_id is None:
        return None
    raw_str = str(contact_id)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:8]


class StructuredFormatter(logging.Formatter):
    """Formats log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured event attributes if present
        for key in (
            "event",
            "contact_id_hash",
            "processing_ms",
            "audio_quality",
            "audio_duration_seconds",
            "gender_prediction",
            "gender_confidence",
            "age_prediction",
            "age_confidence",
            "error_type",
            "error_detail",
            "stage_timings",
        ):
            if hasattr(record, key):
                log_payload[key] = getattr(record, key)

        if record.exc_info and not record.exc_text:
            log_payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_payload["exception"] = record.exc_text

        return json.dumps(log_payload, default=str)


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("voxpulse")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    # Set root logger level to avoid overly verbose 3rd party logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("speechbrain").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("torchaudio").setLevel(logging.WARNING)

    return logger


logger = setup_logging()
