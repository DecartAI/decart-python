import asyncio
import json
import logging
from typing import Callable, Optional, TYPE_CHECKING
from urllib.parse import quote

import aiohttp
from livekit import rtc

from .._user_agent import build_user_agent
from ..errors import WebRTCError
from .messages import (
    ErrorMessage,
    GenerationTickMessage,
    LiveKitJoinMessage,
    LiveKitRoomInfoMessage,
    OutgoingMessage,
    PromptAckMessage,
    PromptMessage,
    QueuePositionMessage,
    SetAvatarImageMessage,
    SetImageAckMessage,
    message_to_json,
    parse_incoming_message,
)
from .types import ConnectionState, VideoCodec

if TYPE_CHECKING:
    from livekit.rtc import (
        LocalVideoTrack,
        RemoteParticipant,
        RemoteTrackPublication,
        RemoteVideoTrack,
    )

logger = logging.getLogger(__name__)

INFERENCE_SERVER_IDENTITY_PREFIX = "inference-server-"
LIVEKIT_HANDSHAKE_TIMEOUT = 15.0

VIDEO_CODEC_MAP = {
    "h264": rtc.VideoCodec.H264,
    "vp9": rtc.VideoCodec.VP9,
}


class LiveKitConnection:
    def __init__(
        self,
        on_remote_stream: Optional[
            Callable[["RemoteVideoTrack", "RemoteTrackPublication", "RemoteParticipant"], None]
        ] = None,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_session_started: Optional[Callable[[LiveKitRoomInfoMessage], None]] = None,
        on_generation_tick: Optional[Callable[[GenerationTickMessage], None]] = None,
        on_queue_position: Optional[Callable[[QueuePositionMessage], None]] = None,
    ):
        self._room: Optional[rtc.Room] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._state: ConnectionState = "disconnected"
        self._on_remote_stream = on_remote_stream
        self._on_state_change = on_state_change
        self._on_error = on_error
        self._on_session_started = on_session_started
        self._on_generation_tick = on_generation_tick
        self._on_queue_position = on_queue_position
        self._ws_task: Optional[asyncio.Task] = None
        self._pending_prompts: dict[str, tuple[asyncio.Event, dict]] = {}
        self._pending_image_set: Optional[tuple[asyncio.Event, dict]] = None
        self._pending_room_info: Optional[tuple[asyncio.Event, dict]] = None
        self._connection_error: Optional[str] = None
        self._intentional_disconnect = False
        self._on_error_fired = False

    async def connect(
        self,
        url: str,
        local_track: Optional["LocalVideoTrack"],
        timeout: float,
        integration: Optional[str] = None,
        initial_image: Optional[str] = None,
        initial_prompt: Optional[dict] = None,
        room_info: Optional[LiveKitRoomInfoMessage] = None,
        preferred_video_codec: VideoCodec = "h264",
    ) -> None:
        try:
            self._connection_error = None
            self._intentional_disconnect = False
            self._on_error_fired = False

            await self._set_state("connecting")
            if room_info is None:
                await self._connect_signaling(url, integration)
                room_info = await self._join_livekit_room(timeout=LIVEKIT_HANDSHAKE_TIMEOUT)

            has_initial_state = initial_image is not None or initial_prompt is not None

            if initial_image is not None:
                await self._send_initial_image_and_wait(
                    initial_image,
                    prompt=initial_prompt.get("text") if initial_prompt else None,
                    enhance=initial_prompt.get("enhance") if initial_prompt else None,
                )
            elif initial_prompt:
                await self._send_initial_prompt_and_wait(initial_prompt)

            await self._connect_room(room_info, local_track, preferred_video_codec)

            if not has_initial_state and local_track is not None:
                await self._send_passthrough_and_wait()

            if self._on_session_started:
                self._on_session_started(room_info)

            await self._wait_until_connected(timeout)
        except WebRTCError as e:
            logger.error("LiveKit connection failed: %s", e)
            await self._set_state("disconnected")
            self._fire_error_once(e)
            raise
        except Exception as e:
            logger.error("LiveKit connection failed: %s", e)
            await self._set_state("disconnected")
            self._fire_error_once(e)
            raise WebRTCError(str(e), cause=e)

    async def _connect_signaling(self, url: str, integration: Optional[str]) -> None:
        ws_url = url.replace("https://", "wss://").replace("http://", "ws://")
        user_agent = build_user_agent(integration)
        separator = "&" if "?" in ws_url else "?"
        ws_url = f"{ws_url}{separator}user_agent={quote(user_agent)}"

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(ws_url)
        self._ws_task = asyncio.create_task(self._receive_messages())

    async def _join_livekit_room(self, timeout: float) -> LiveKitRoomInfoMessage:
        event = asyncio.Event()
        result: dict = {"room_info": None, "error": None}
        self._pending_room_info = (event, result)

        try:
            await self._send_message(LiveKitJoinMessage(type="livekit_join"))
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise WebRTCError("LiveKit room info timed out") from e
        finally:
            self._pending_room_info = None

        if result["error"]:
            raise WebRTCError(str(result["error"]))
        room_info = result["room_info"]
        if not isinstance(room_info, LiveKitRoomInfoMessage):
            raise WebRTCError("Invalid LiveKit room info")
        return room_info

    async def _connect_room(
        self,
        room_info: LiveKitRoomInfoMessage,
        local_track: Optional["LocalVideoTrack"],
        preferred_video_codec: VideoCodec = "h264",
    ) -> None:
        room = rtc.Room()
        self._room = room

        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if not participant.identity.startswith(INFERENCE_SERVER_IDENTITY_PREFIX):
                return
            if getattr(track, "kind", None) != rtc.TrackKind.KIND_VIDEO:
                return
            logger.debug(
                "Received LiveKit remote video track: %s", getattr(track, "sid", "unknown")
            )
            if self._on_remote_stream:
                self._on_remote_stream(track, publication, participant)

        @room.on("connection_state_changed")
        def on_connection_state_changed(connection_state):
            mapped = self._map_room_state(connection_state)
            if mapped:
                asyncio.create_task(self._set_state(mapped))

        @room.on("disconnected")
        def on_disconnected(_reason=None):
            if not self._intentional_disconnect:
                logger.debug("LiveKit room disconnected")
            asyncio.create_task(self._set_state("disconnected"))

        await room.connect(room_info.livekit_url, room_info.token)

        if local_track is not None:
            await room.local_participant.publish_track(
                local_track,
                self._publish_options(preferred_video_codec),
            )

        await self._set_state("connected")

    def _publish_options(self, preferred_video_codec: VideoCodec) -> rtc.TrackPublishOptions:
        options = rtc.TrackPublishOptions()
        options.video_codec = VIDEO_CODEC_MAP[preferred_video_codec]
        return options

    def _map_room_state(self, connection_state) -> Optional[ConnectionState]:
        state_name = getattr(connection_state, "name", str(connection_state)).lower()
        if "reconnecting" in state_name:
            return "reconnecting"
        if "connecting" in state_name:
            return "connecting"
        if "connected" in state_name and "disconnected" not in state_name:
            return "connected"
        if "disconnected" in state_name:
            return "disconnected"
        return None

    async def _wait_until_connected(self, timeout: float) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self._state in ("connected", "generating"):
                return
            if self._connection_error:
                raise WebRTCError(self._connection_error)
            await asyncio.sleep(0.1)
        raise TimeoutError("Connection timeout")

    async def _send_initial_image_and_wait(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        enhance: Optional[bool] = None,
        timeout: float = 30.0,
    ) -> None:
        event, result = self.register_image_set_wait()

        try:
            message = SetAvatarImageMessage(type="set_image", image_data=image_base64)
            if prompt is not None:
                message.prompt = prompt
            if enhance is not None:
                message.enhance_prompt = enhance

            await self._send_message(message)
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise WebRTCError("Initial image acknowledgment timed out")

            if not result["success"]:
                raise WebRTCError(
                    f"Failed to set initial image: {result.get('error', 'unknown error')}"
                )
        finally:
            self.unregister_image_set_wait()

    async def _send_initial_prompt_and_wait(self, prompt: dict, timeout: float = 15.0) -> None:
        prompt_text = prompt.get("text", "")
        enhance = prompt.get("enhance", True)

        event, result = self.register_prompt_wait(prompt_text)

        try:
            await self._send_message(
                PromptMessage(type="prompt", prompt=prompt_text, enhance_prompt=enhance)
            )
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise WebRTCError("Initial prompt acknowledgment timed out")

            if not result["success"]:
                raise WebRTCError(
                    f"Failed to send initial prompt: {result.get('error', 'unknown error')}"
                )
        finally:
            self.unregister_prompt_wait(prompt_text)

    async def _send_passthrough_and_wait(self, timeout: float = 30.0) -> None:
        event, result = self.register_image_set_wait()

        try:
            await self._send_message(
                SetAvatarImageMessage(type="set_image", image_data=None, prompt=None)
            )
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise WebRTCError("Passthrough acknowledgment timed out")

            if not result["success"]:
                raise WebRTCError(
                    f"Failed to send passthrough: {result.get('error', 'unknown error')}"
                )
        finally:
            self.unregister_image_set_wait()

    async def _receive_messages(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.debug("Received %s message", data.get("type", "unknown"))
                        await self._handle_message(data)
                    except Exception as e:
                        logger.error("Error handling message: %s", e)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket error: %s", self._ws.exception())
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("WebSocket receive error: %s", e)
            if self._on_error:
                self._on_error(e)
        finally:
            final_error = self._connection_error or "Control channel disconnected"
            self._resolve_pending_waits(final_error)
            if not self._intentional_disconnect:
                await self._set_state("disconnected")

    async def _handle_message(self, data: dict) -> None:
        try:
            message = parse_incoming_message(data)
        except Exception as e:
            logger.warning("Failed to parse message: %s", e)
            return

        if message.type == "livekit_room_info":
            self._handle_room_info(message)
        elif message.type == "queue_position":
            if self._on_queue_position:
                self._on_queue_position(message)
        elif message.type == "session_id":
            logger.debug(
                "Ignoring legacy session_id message in LiveKit mode: %s", message.session_id
            )
        elif message.type == "prompt_ack":
            self._handle_prompt_ack(message)
        elif message.type == "set_image_ack":
            self._handle_set_image_ack(message)
        elif message.type == "generation_started":
            await self._set_state("generating")
        elif message.type == "generation_tick":
            if self._state == "connected":
                await self._set_state("generating")
            if self._on_generation_tick:
                self._on_generation_tick(message)
        elif message.type == "generation_ended":
            logger.debug("Generation ended: reason=%s, seconds=%s", message.reason, message.seconds)
        elif message.type == "error":
            self._handle_error(message)
        elif message.type == "ready":
            logger.debug("Ignoring legacy ready signal in LiveKit mode")

    def _handle_room_info(self, message: LiveKitRoomInfoMessage) -> None:
        if self._pending_room_info:
            event, result = self._pending_room_info
            result["room_info"] = message
            event.set()

    def _handle_prompt_ack(self, message: PromptAckMessage) -> None:
        if message.prompt in self._pending_prompts:
            event, result = self._pending_prompts[message.prompt]
            result["success"] = message.success
            result["error"] = message.error
            event.set()

    def _handle_set_image_ack(self, message: SetImageAckMessage) -> None:
        if self._pending_image_set:
            event, result = self._pending_image_set
            result["success"] = message.success
            result["error"] = message.error
            event.set()

    def _resolve_pending_waits(self, error_message: str) -> None:
        if self._pending_room_info:
            event, result = self._pending_room_info
            result["error"] = error_message
            event.set()

        if self._pending_image_set:
            event, result = self._pending_image_set
            result["success"] = False
            result["error"] = error_message
            event.set()

        for _prompt, (event, result) in list(self._pending_prompts.items()):
            result["success"] = False
            result["error"] = error_message
            event.set()

    def _handle_error(self, message: ErrorMessage) -> None:
        logger.error("Received error from server: %s", message.error)
        error = WebRTCError(message.error)

        if not self._connection_error:
            self._connection_error = message.error
        self._resolve_pending_waits(message.error)
        self._fire_error_once(error)

    def _fire_error_once(self, error: Exception) -> None:
        if self._on_error and not self._on_error_fired:
            self._on_error_fired = True
            self._on_error(error)

    def register_image_set_wait(self) -> tuple[asyncio.Event, dict]:
        event = asyncio.Event()
        result: dict = {"success": False, "error": None}
        self._pending_image_set = (event, result)
        return event, result

    def unregister_image_set_wait(self) -> None:
        self._pending_image_set = None

    def register_prompt_wait(self, prompt: str) -> tuple[asyncio.Event, dict]:
        event = asyncio.Event()
        result: dict = {"success": False, "error": None}
        self._pending_prompts[prompt] = (event, result)
        return event, result

    def unregister_prompt_wait(self, prompt: str) -> None:
        self._pending_prompts.pop(prompt, None)

    async def _send_message(self, message: OutgoingMessage) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("Control channel not connected")

        msg_json = message_to_json(message)
        logger.debug("Sending %s message", message.type)
        await self._ws.send_str(msg_json)

    async def _set_state(self, state: ConnectionState) -> None:
        if self._state == "generating" and state not in ("disconnected", "generating"):
            return
        if self._state != state:
            self._state = state
            logger.debug("Connection state changed to: %s", state)
            if self._on_state_change:
                self._on_state_change(state)

    async def send(self, message: OutgoingMessage) -> None:
        await self._send_message(message)

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def room(self) -> Optional[rtc.Room]:
        return self._room

    async def cleanup(self) -> None:
        self._intentional_disconnect = True

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        if self._room:
            await self._room.disconnect()
            self._room = None

        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        await self._set_state("disconnected")
