import asyncio
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
    integration: Optional[str] = None
    is_avatar_live: bool = False


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
    async def connect(
        self,
        local_track: Optional[MediaStreamTrack],
        avatar_image_base64: Optional[str] = None,
        initial_prompt: Optional[dict] = None,
    ) -> bool:
        try:
            timeout = 60 * 5  # 5 minutes
            await self._connection.connect(
                url=self._config.webrtc_url,
                local_track=local_track,
                timeout=timeout,
                integration=self._config.integration,
                is_avatar_live=self._config.is_avatar_live,
                avatar_image_base64=avatar_image_base64,
                initial_prompt=initial_prompt,
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

    async def set_image(
        self,
        image_base64: Optional[str],
        options: Optional[dict] = None,
    ) -> None:
        from .messages import SetAvatarImageMessage

        opts = options or {}
        timeout = opts.get("timeout", 30.0)

        event, result = self._connection.register_image_set_wait()

        try:
            message = SetAvatarImageMessage(
                type="set_image",
                image_data=image_base64,
            )
            if opts.get("prompt") is not None:
                message.prompt = opts["prompt"]
            if opts.get("enhance") is not None:
                message.enhance_prompt = opts["enhance"]

            await self._connection.send(message)

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                from ..errors import DecartSDKError

                raise DecartSDKError("Image send timed out")

            if not result["success"]:
                from ..errors import DecartSDKError

                raise DecartSDKError(result.get("error") or "Failed to set image")
        finally:
            self._connection.unregister_image_set_wait()

    async def send_message(self, message: OutgoingMessage) -> None:
        await self._connection.send(message)

    async def cleanup(self) -> None:
        await self._connection.cleanup()

    def is_connected(self) -> bool:
        return self._connection.state == "connected"

    def get_connection_state(self) -> ConnectionState:
        return self._connection.state

    def register_prompt_wait(self, prompt: str) -> tuple[asyncio.Event, dict]:
        return self._connection.register_prompt_wait(prompt)

    def unregister_prompt_wait(self, prompt: str) -> None:
        self._connection.unregister_prompt_wait(prompt)

    def register_image_set_wait(self) -> tuple[asyncio.Event, dict]:
        return self._connection.register_image_set_wait()

    def unregister_image_set_wait(self) -> None:
        self._connection.unregister_image_set_wait()
