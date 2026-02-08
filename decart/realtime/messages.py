from typing import Literal, Optional, Union, Annotated
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


class PromptAckMessage(BaseModel):
    """Acknowledgment for prompt update from server."""

    type: Literal["prompt_ack"]
    prompt: str
    success: bool
    error: Optional[str] = None


class SetImageAckMessage(BaseModel):
    """Acknowledgment for avatar image set from server."""

    type: Literal["set_image_ack"]
    success: bool
    error: Optional[str] = None


class ErrorMessage(BaseModel):
    """Error message from server."""

    type: Literal["error"]
    error: str


class ReadyMessage(BaseModel):
    """Server ready signal."""

    type: Literal["ready"]


class TurnConfig(BaseModel):
    """TURN server configuration."""

    username: str
    credential: str
    server_url: str


class IceRestartMessage(BaseModel):
    """ICE restart message with TURN config."""

    type: Literal["ice-restart"]
    turn_config: TurnConfig


class SetAckMessage(BaseModel):
    type: Literal["set_ack"]
    success: bool
    error: Optional[str] = None


# Discriminated union for incoming messages
IncomingMessage = Annotated[
    Union[
        AnswerMessage,
        IceCandidateMessage,
        SessionIdMessage,
        PromptAckMessage,
        SetImageAckMessage,
        SetAckMessage,
        ErrorMessage,
        ReadyMessage,
        IceRestartMessage,
    ],
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
    enhance_prompt: bool = True


class SetAvatarImageMessage(BaseModel):
    """Set avatar image message."""

    type: Literal["set_image"]
    image_data: str  # Base64-encoded image


class SetParamsMessage(BaseModel):
    type: Literal["set"] = "set"
    prompt: Optional[str] = None
    enhance_prompt: Optional[bool] = None
    image_data: Optional[str] = None


OutgoingMessage = Union[
    OfferMessage, IceCandidateMessage, PromptMessage, SetAvatarImageMessage, SetParamsMessage
]


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
    if isinstance(message, SetParamsMessage):
        return message.model_dump_json(exclude_unset=True)
    return message.model_dump_json()
