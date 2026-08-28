"""Request schemas for VoxPulse Voice Attribute Inference Service."""

from uuid import UUID
from pydantic import BaseModel, Field


class StreamControlMessage(BaseModel):
    """Client control message for WebSocket streaming."""

    action: str = Field(
        ...,
        description="Action command: 'finalize', 'reset', or 'ping'",
    )
