import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .livekit_connection import LiveKitConnection
from .messages import (
    GenerationTickMessage,
    LiveKitRoomInfoMessage,
    OutgoingMessage,
    QueuePositionMessage,
)
from .types import ConnectionState, VideoCodec
from ..types import ModelState

if TYPE_CHECKING:
    from livekit.rtc import LocalVideoTrack, RemoteVideoTrack

logger = logging.getLogger(__name__)

PERMANENT_ERRORS = [
    "permission denied",
    "not allowed",
    "invalid session",
    "401",
    "invalid api key",
    "unauthorized",
]

CONNECTION_TIMEOUT = 60 * 5

RETRY_MAX_ATTEMPTS = 5
RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 10


@dataclass
class LiveKitConfiguration:
    livekit_url: str
    api_key: str
    session_id: str
    fps: int
    on_remote_stream: Callable[["RemoteVideoTrack"], None]
    on_connection_state_change: Optional[Callable[[ConnectionState], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    on_session_started: Optional[Callable[[LiveKitRoomInfoMessage], None]] = None
    on_generation_tick: Optional[Callable[[GenerationTickMessage], None]] = None
    on_queue_position: Optional[Callable[[QueuePositionMessage], None]] = None
    room_info: Optional[LiveKitRoomInfoMessage] = None
    initial_state: Optional[ModelState] = None
    preferred_video_codec: VideoCodec = "h264"
    integration: Optional[str] = None


def _is_permanent_error(exception: BaseException) -> bool:
    error_msg = str(exception).lower()
    return any(err in error_msg for err in PERMANENT_ERRORS)


def _is_retryable_error(exception: BaseException) -> bool:
    if isinstance(exception, asyncio.CancelledError):
        return False
    return not _is_permanent_error(exception)


class LiveKitManager:
    def __init__(self, configuration: LiveKitConfiguration):
        self._config = configuration
        self._connection: Optional[LiveKitConnection] = None
        self._local_track: Optional["LocalVideoTrack"] = None
        self._subscribe_mode = False
        self._manager_state: ConnectionState = "disconnected"
        self._has_connected = False
        self._is_reconnecting = False
        self._intentional_disconnect = False
        self._reconnect_generation = 0
        self._reconnect_task: Optional[asyncio.Task] = None

    def _get_connection(self) -> LiveKitConnection:
        if self._connection is None:
            raise RuntimeError("LiveKitManager not connected")
        return self._connection

    def _emit_state(self, state: ConnectionState) -> None:
        if self._manager_state != state:
            self._manager_state = state
            if state in ("connected", "generating"):
                self._has_connected = True
            if self._config.on_connection_state_change:
                self._config.on_connection_state_change(state)

    def _handle_remote_stream(self, track, _publication, _participant) -> None:
        self._config.on_remote_stream(track)

    def _handle_connection_state_change(self, state: ConnectionState) -> None:
        if self._intentional_disconnect:
            self._emit_state("disconnected")
            return

        if self._is_reconnecting:
            if state in ("connected", "generating"):
                self._is_reconnecting = False
                self._emit_state(state)
            return

        if state == "disconnected" and not self._intentional_disconnect and self._has_connected:
            if not self._reconnect_task or self._reconnect_task.done():
                self._reconnect_task = asyncio.ensure_future(self._reconnect())
            return

        self._emit_state(state)

    async def _reconnect(self) -> None:
        if self._is_reconnecting or self._intentional_disconnect:
            return
        if not self._subscribe_mode and not self._local_track:
            return

        reconnect_generation = self._reconnect_generation + 1
        self._reconnect_generation = reconnect_generation
        self._is_reconnecting = True
        self._emit_state("reconnecting")

        try:
            await self._retry_reconnect(reconnect_generation)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            if self._intentional_disconnect or reconnect_generation != self._reconnect_generation:
                return
            self._emit_state("disconnected")
            if self._config.on_error:
                self._config.on_error(
                    error if isinstance(error, Exception) else Exception(str(error))
                )
        finally:
            self._is_reconnecting = False

    async def _retry_reconnect(self, reconnect_generation: int) -> None:
        @retry(
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            retry=retry_if_exception(_is_retryable_error),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _attempt():
            if self._intentional_disconnect or reconnect_generation != self._reconnect_generation:
                raise asyncio.CancelledError("Reconnect cancelled")

            if self._connection is not None:
                await self._connection.cleanup()
            conn = self._create_connection()
            self._connection = conn
            await conn.connect(
                url=self._config.livekit_url,
                local_track=self._local_track,
                timeout=CONNECTION_TIMEOUT,
                integration=self._config.integration,
                room_info=self._config.room_info,
                preferred_video_codec=self._config.preferred_video_codec,
            )

            if self._intentional_disconnect or reconnect_generation != self._reconnect_generation:
                await conn.cleanup()
                raise asyncio.CancelledError("Reconnect cancelled")

        await _attempt()

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def connect(
        self,
        local_track: Optional["LocalVideoTrack"],
        initial_image: Optional[str] = None,
        initial_prompt: Optional[dict] = None,
    ) -> bool:
        self._local_track = local_track
        self._subscribe_mode = local_track is None
        self._intentional_disconnect = False
        self._has_connected = False
        self._is_reconnecting = False
        self._reconnect_generation += 1
        self._connection = self._create_connection()
        self._emit_state("connecting")

        try:
            await self._connection.connect(
                url=self._config.livekit_url,
                local_track=local_track,
                timeout=CONNECTION_TIMEOUT,
                integration=self._config.integration,
                initial_image=initial_image,
                initial_prompt=initial_prompt,
                room_info=self._config.room_info,
                preferred_video_codec=self._config.preferred_video_codec,
            )
            return True
        except Exception as e:
            logger.error("Connection attempt failed: %s", e)
            await self._connection.cleanup()
            self._connection = None
            raise

    def _create_connection(self) -> LiveKitConnection:
        return LiveKitConnection(
            on_remote_stream=self._handle_remote_stream,
            on_state_change=self._handle_connection_state_change,
            on_error=self._config.on_error,
            on_session_started=self._config.on_session_started,
            on_generation_tick=self._config.on_generation_tick,
            on_queue_position=self._config.on_queue_position,
        )

    async def set_image(
        self,
        image_base64: Optional[str],
        options: Optional[dict] = None,
    ) -> None:
        from .messages import SetAvatarImageMessage

        opts = options or {}
        timeout = opts.get("timeout", 30.0)

        conn = self._get_connection()
        event, result = conn.register_image_set_wait()

        try:
            message = SetAvatarImageMessage(
                type="set_image",
                image_data=image_base64,
            )
            if opts.get("prompt") is not None:
                message.prompt = opts["prompt"]
            if opts.get("enhance") is not None:
                message.enhance_prompt = opts["enhance"]

            await conn.send(message)

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                from ..errors import DecartSDKError

                raise DecartSDKError("Image send timed out")

            if not result["success"]:
                from ..errors import DecartSDKError

                raise DecartSDKError(result.get("error") or "Failed to set image")
        finally:
            conn.unregister_image_set_wait()

    async def send_message(self, message: OutgoingMessage) -> None:
        await self._get_connection().send(message)

    async def cleanup(self) -> None:
        self._intentional_disconnect = True
        self._is_reconnecting = False
        self._reconnect_generation += 1
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        if self._connection:
            await self._connection.cleanup()
            self._connection = None
        self._local_track = None
        self._emit_state("disconnected")

    def is_connected(self) -> bool:
        return self._manager_state in ("connected", "generating")

    def get_connection_state(self) -> ConnectionState:
        return self._manager_state

    def register_prompt_wait(self, prompt: str) -> tuple[asyncio.Event, dict]:
        return self._get_connection().register_prompt_wait(prompt)

    def unregister_prompt_wait(self, prompt: str) -> None:
        self._get_connection().unregister_prompt_wait(prompt)

    def register_image_set_wait(self) -> tuple[asyncio.Event, dict]:
        return self._get_connection().register_image_set_wait()

    def unregister_image_set_wait(self) -> None:
        self._get_connection().unregister_image_set_wait()
