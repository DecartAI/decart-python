import asyncio
import logging
from typing import Optional, Callable
from dataclasses import dataclass

try:
    from aiortc import MediaStreamTrack
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    MediaStreamTrack = None  # type: ignore

from .webrtc_connection import WebRTCConnection
from .messages import OutgoingMessage
from .types import ConnectionState
from ..types import ModelState

logger = logging.getLogger(__name__)


@dataclass
class WebRTCConfiguration:
    webrtc_url: str
    api_key: str
    session_id: str
    fps: int
    on_remote_stream: Callable[[MediaStreamTrack], None]
    on_connection_state_change: Optional[Callable[[ConnectionState], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    initial_state: Optional[ModelState] = None
    customize_offer: Optional[Callable] = None


class WebRTCManager:
    PERMANENT_ERRORS = [
        "permission denied",
        "not allowed",
        "invalid session",
    ]
    
    def __init__(self, configuration: WebRTCConfiguration):
        if not WEBRTC_AVAILABLE:
            raise ImportError(
                "aiortc is required for Realtime API. "
                "Install with: pip install decart-sdk[realtime]"
            )
        
        self._config = configuration
        self._connection = self._create_connection()
    
    async def connect(self, local_track: MediaStreamTrack) -> bool:
        retries = 0
        max_retries = 5
        delay = 1.0
        
        while retries < max_retries:
            try:
                logger.info(f"Connecting to WebRTC (attempt {retries + 1}/{max_retries})")
                
                await self._connection.connect(
                    url=self._config.webrtc_url,
                    local_track=local_track,
                )
                
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                is_permanent = any(err in error_msg for err in self.PERMANENT_ERRORS)
                
                if is_permanent or retries >= max_retries - 1:
                    logger.error(f"Connection failed permanently: {e}")
                    raise
                
                retries += 1
                logger.warning(f"Connection attempt {retries} failed: {e}. Retrying in {delay}s...")
                
                await self._connection.cleanup()
                self._connection = self._create_connection()
                await asyncio.sleep(delay)
                
                delay = min(delay * 2, 10.0)
        
        return False
    
    def _create_connection(self) -> WebRTCConnection:
        return WebRTCConnection(
            on_remote_stream=self._config.on_remote_stream,
            on_state_change=self._config.on_connection_state_change,
            on_error=self._config.on_error,
            customize_offer=self._config.customize_offer,
        )
    
    async def send_message(self, message: OutgoingMessage) -> None:
        await self._connection.send(message)
    
    async def cleanup(self) -> None:
        await self._connection.cleanup()
    
    def is_connected(self) -> bool:
        return self._connection.state == "connected"
    
    def get_connection_state(self) -> ConnectionState:
        return self._connection.state
