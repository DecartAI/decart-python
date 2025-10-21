import logging
from typing import Optional, Callable
from dataclasses import dataclass
from aiortc import MediaStreamTrack
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

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


def _is_retryable_error(exception: Exception) -> bool:
    """Check if an error is retryable (not a permanent error)."""
    permanent_errors = ["permission denied", "not allowed", "invalid session"]
    error_msg = str(exception).lower()
    return not any(err in error_msg for err in permanent_errors)


class WebRTCManager:
    def __init__(self, configuration: WebRTCConfiguration):
        self._config = configuration
        self._connection = self._create_connection()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def connect(self, local_track: MediaStreamTrack) -> bool:
        try:
            await self._connection.connect(
                url=self._config.webrtc_url,
                local_track=local_track,
            )
            return True
        except Exception as e:
            logger.error(f"Connection attempt failed: {e}")
            await self._connection.cleanup()
            self._connection = self._create_connection()
            raise

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
