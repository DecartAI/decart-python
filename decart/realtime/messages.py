from typing import Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field, TypeAdapter


# Incoming Messages (from server)


class SessionIdMessage(BaseModel):
    """Legacy session initialization message from the pre-LiveKit protocol."""

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
    """Legacy server ready signal from the pre-LiveKit protocol."""

    type: Literal["ready"]


class LiveKitRoomInfoMessage(BaseModel):
    """LiveKit room credentials returned by the realtime control channel."""

    type: Literal["livekit_room_info"]
    livekit_url: str
    token: str
    room_name: str
    session_id: str


class QueuePositionMessage(BaseModel):
    """Queue position update while waiting for LiveKit room credentials."""

    type: Literal["queue_position"]
    position: int
    queue_size: int


class GenerationStartedMessage(BaseModel):
    """Server signals that generation has started."""

    type: Literal["generation_started"]


class GenerationTickMessage(BaseModel):
    """Periodic billing update during generation."""

    type: Literal["generation_tick"]
    seconds: int


class GenerationEndedMessage(BaseModel):
    """Server signals that generation has ended. Not exposed publicly."""

    type: Literal["generation_ended"]
    seconds: int
    reason: str


# Discriminated union for incoming messages
IncomingMessage = Annotated[
    Union[
        SessionIdMessage,
        PromptAckMessage,
        SetImageAckMessage,
        ErrorMessage,
        ReadyMessage,
        LiveKitRoomInfoMessage,
        QueuePositionMessage,
        GenerationStartedMessage,
        GenerationTickMessage,
        GenerationEndedMessage,
    ],
    Field(discriminator="type"),
]

# Type adapter for parsing incoming messages
IncomingMessageAdapter = TypeAdapter(IncomingMessage)


# Outgoing Messages (to server)


class LiveKitJoinMessage(BaseModel):
    """Ask the control channel for LiveKit room credentials."""

    type: Literal["livekit_join"]


class PromptMessage(BaseModel):
    """Update prompt message."""

    type: Literal["prompt"]
    prompt: str
    enhance_prompt: bool = True


class SetAvatarImageMessage(BaseModel):
    """Set avatar image message."""

    type: Literal["set_image"]
    image_data: Optional[str] = None
    prompt: Optional[str] = None
    enhance_prompt: Optional[bool] = None


# Outgoing message union (no discriminator needed - we know what we're sending)
OutgoingMessage = Union[LiveKitJoinMessage, PromptMessage, SetAvatarImageMessage]


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
    # SetAvatarImageMessage uses exclude_unset so explicitly-passed None values
    # (e.g. image_data=None, prompt=None for passthrough) are serialized as null,
    # while fields that were never set are omitted.
    if isinstance(message, SetAvatarImageMessage):
        return message.model_dump_json(exclude_unset=True)
    return message.model_dump_json(exclude_none=True)
