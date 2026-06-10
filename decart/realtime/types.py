from typing import Literal, Callable, Optional, TYPE_CHECKING
from dataclasses import dataclass
from ..models import ModelDefinition
from ..types import ModelState

if TYPE_CHECKING:
    from livekit.rtc import RemoteVideoTrack


ConnectionState = Literal["connecting", "connected", "generating", "disconnected", "reconnecting"]
VideoCodec = Literal["h264", "vp9"]


@dataclass
class RealtimeConnectOptions:
    model: ModelDefinition
    on_remote_stream: Callable[["RemoteVideoTrack"], None]
    initial_state: Optional[ModelState] = None
    resolution: Optional[Literal["720p", "1080p"]] = None
    preferred_video_codec: VideoCodec = "h264"
