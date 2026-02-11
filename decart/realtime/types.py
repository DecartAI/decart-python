from typing import Literal, Callable, Optional
from dataclasses import dataclass
from ..models import ModelDefinition
from ..types import ModelState, FileInput

try:
    from aiortc import MediaStreamTrack
except ImportError:
    MediaStreamTrack = None  # type: ignore


ConnectionState = Literal["connecting", "connected", "generating", "disconnected", "reconnecting"]


@dataclass
class AvatarOptions:
    """Options for avatar-live model."""

    avatar_image: FileInput
    """The avatar image to use. Can be bytes, Path, URL string, or file-like object."""


@dataclass
class InitialPromptOptions:
    """Options for initial prompt sent before WebRTC handshake."""

    text: str
    """The prompt text to send."""

    enhance: bool = True
    """Whether to enhance the prompt. Defaults to True."""


@dataclass
class RealtimeConnectOptions:
    model: ModelDefinition
    on_remote_stream: Callable[[MediaStreamTrack], None]
    initial_state: Optional[ModelState] = None
    customize_offer: Optional[Callable] = None
    avatar: Optional[AvatarOptions] = None
    initial_prompt: Optional[InitialPromptOptions] = None
    """Initial prompt to send before WebRTC handshake (optional)."""
