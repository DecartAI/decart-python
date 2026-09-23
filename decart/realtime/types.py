from typing import Literal, Callable, Optional, TYPE_CHECKING
from dataclasses import dataclass
from ..models import ModelDefinition, RealtimeSpeed
from ..types import ModelState

if TYPE_CHECKING:
    from livekit.rtc import RemoteVideoTrack


ConnectionState = Literal["connecting", "connected", "generating", "disconnected", "reconnecting"]
VideoCodec = Literal["h264", "vp9"]

__all__ = ["ConnectionState", "VideoCodec", "RealtimeSpeed", "RealtimeConnectOptions"]


@dataclass
class RealtimeConnectOptions:
    model: ModelDefinition
    on_remote_stream: Callable[["RemoteVideoTrack"], None]
    initial_state: Optional[ModelState] = None
    resolution: Optional[Literal["720p", "1080p"]] = None
    preferred_video_codec: VideoCodec = "h264"
    speed: Optional[RealtimeSpeed] = None
    """Realtime speed tier. ``"fast"`` serves the session from a higher-compute tier for
    lower latency and higher throughput; output quality is unchanged. Currently available
    for lucy-2.5 / lucy-latest and lucy-vton-3.5 / lucy-vton-latest, in the US region only,
    and billed at 2x the standard realtime rate for those models. Other models ignore the
    option. Omit it (the default) for standard mode."""
