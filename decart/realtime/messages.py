from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, TypeAdapter

try:
    from aiortc import RTCSessionDescription, RTCIceCandidate
except ImportError:
    RTCSessionDescription = None  # type: ignore
    RTCIceCandidate = None  # type: ignore


# Incoming Messages (from server)


class AnswerMessage(BaseModel):
    """WebRTC answer from server."""

    type: Literal["answer"]
    sdp: str


class IceCandidatePayload(BaseModel):
    """ICE candidate data."""

    candidate: str
    sdpMLineIndex: int
    sdpMid: str


class IceCandidateMessage(BaseModel):
    """ICE candidate message."""

    type: Literal["ice-candidate"]
    candidate: IceCandidatePayload


class SessionIdMessage(BaseModel):
    """Session initialization message from server."""

    type: Literal["session_id"]
    session_id: str
    server_port: int
    server_ip: str


# Discriminated union for incoming messages
IncomingMessage = Annotated[
    Union[AnswerMessage, IceCandidateMessage, SessionIdMessage],
    Field(discriminator="type"),
]

# Type adapter for parsing incoming messages
IncomingMessageAdapter = TypeAdapter(IncomingMessage)


# Outgoing Messages (to server)


class OfferMessage(BaseModel):
    """WebRTC offer to server."""

    type: Literal["offer"]
    sdp: str


class PromptMessage(BaseModel):
    """Update prompt message."""

    type: Literal["prompt"]
    prompt: str


class SwitchCameraMessage(BaseModel):
    """Switch camera/mirror message."""

    type: Literal["switch_camera"]
    rotateY: int


# Outgoing message union (no discriminator needed - we know what we're sending)
OutgoingMessage = Union[OfferMessage, IceCandidateMessage, PromptMessage, SwitchCameraMessage]


def parse_incoming_message(data: dict) -> IncomingMessage:
    """
    Parse incoming WebSocket message.

    Args:
        data: Message dictionary

    Returns:
        Parsed message instance

    Raises:
        ValidationError: If message format is invalid
    """
    return IncomingMessageAdapter.validate_python(data)


def message_to_json(message: OutgoingMessage) -> str:
    """
    Serialize outgoing message to JSON.

    Args:
        message: Message to serialize

    Returns:
        JSON string
    """
    return message.model_dump_json()
