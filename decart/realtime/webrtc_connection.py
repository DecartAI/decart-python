import asyncio
import json
import logging
from typing import Optional, Callable
from urllib.parse import quote
import aiohttp
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
)

from ..errors import WebRTCError
from .._user_agent import build_user_agent
from .messages import (
    parse_incoming_message,
    message_to_json,
    OfferMessage,
    IceCandidateMessage,
    IceCandidatePayload,
    PromptMessage,
    PromptAckMessage,
    SetImageAckMessage,
    SetAvatarImageMessage,
    SetAckMessage,
    SetParamsMessage,
    ErrorMessage,
    IceRestartMessage,
    OutgoingMessage,
)
from .types import ConnectionState

logger = logging.getLogger(__name__)


class WebRTCConnection:
    def __init__(
        self,
        on_remote_stream: Optional[Callable[[MediaStreamTrack], None]] = None,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        customize_offer: Optional[Callable] = None,
    ):
        self._pc: Optional[RTCPeerConnection] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._state: ConnectionState = "disconnected"
        self._on_remote_stream = on_remote_stream
        self._on_state_change = on_state_change
        self._on_error = on_error
        self._customize_offer = customize_offer
        self._ws_task: Optional[asyncio.Task] = None
        self._ice_candidates_queue: list[RTCIceCandidate] = []
        self._pending_prompts: dict[str, tuple[asyncio.Event, dict]] = {}
        self._pending_image_set: Optional[tuple[asyncio.Event, dict]] = None
        self._pending_set: Optional[tuple[asyncio.Event, dict]] = None

    async def connect(
        self,
        url: str,
        local_track: Optional[MediaStreamTrack],
        timeout: float = 30,
        integration: Optional[str] = None,
        is_avatar_live: bool = False,
        avatar_image_base64: Optional[str] = None,
        initial_prompt: Optional[dict] = None,
    ) -> None:
        try:
            await self._set_state("connecting")

            ws_url = url.replace("https://", "wss://").replace("http://", "ws://")

            # Add user agent as query parameter (browsers don't support WS headers)
            user_agent = build_user_agent(integration)
            separator = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{separator}user_agent={quote(user_agent)}"

            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(ws_url)

            self._ws_task = asyncio.create_task(self._receive_messages())

            # For avatar-live, send avatar image before WebRTC handshake
            if is_avatar_live and avatar_image_base64:
                await self._send_avatar_image_and_wait(avatar_image_base64)

            # Send initial prompt before WebRTC handshake (if provided)
            if initial_prompt:
                await self._send_initial_prompt_and_wait(initial_prompt)

            await self._setup_peer_connection(local_track, is_avatar_live=is_avatar_live)

            await self._create_and_send_offer()

            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                if self._state == "connected":
                    return
                await asyncio.sleep(0.1)

            raise TimeoutError("Connection timeout")

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self._set_state("disconnected")
            if self._on_error:
                self._on_error(e)
            raise WebRTCError(str(e), cause=e)

    async def _send_avatar_image_and_wait(self, image_base64: str, timeout: float = 15.0) -> None:
        """Send avatar image and wait for acknowledgment."""
        event, result = self.register_image_set_wait()

        try:
            await self._send_message(
                SetAvatarImageMessage(type="set_image", image_data=image_base64)
            )

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise WebRTCError("Avatar image acknowledgment timed out")

            if not result["success"]:
                raise WebRTCError(
                    f"Failed to set avatar image: {result.get('error', 'unknown error')}"
                )
        finally:
            self.unregister_image_set_wait()

    async def _send_initial_prompt_and_wait(self, prompt: dict, timeout: float = 15.0) -> None:
        """Send initial prompt and wait for acknowledgment before WebRTC handshake."""
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

    async def _setup_peer_connection(
        self,
        local_track: Optional[MediaStreamTrack],
        is_avatar_live: bool = False,
    ) -> None:
        config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])

        self._pc = RTCPeerConnection(configuration=config)

        @self._pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.debug(f"Received remote track: {track.kind}")
            if self._on_remote_stream:
                self._on_remote_stream(track)

        @self._pc.on("icecandidate")
        async def on_ice_candidate(candidate: RTCIceCandidate):
            if candidate:
                logger.debug(f"Local ICE candidate: {candidate.candidate}")
                await self._send_message(
                    IceCandidateMessage(
                        type="ice-candidate",
                        candidate=IceCandidatePayload(
                            candidate=candidate.candidate,
                            sdpMLineIndex=candidate.sdpMLineIndex or 0,
                            sdpMid=candidate.sdpMid or "",
                        ),
                    )
                )

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.debug(f"Peer connection state: {self._pc.connectionState}")
            if self._pc.connectionState == "connected":
                await self._set_state("connected")
            elif self._pc.connectionState in ["failed", "closed"]:
                await self._set_state("disconnected")

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_connection_state_change():
            logger.debug(f"ICE connection state: {self._pc.iceConnectionState}")

        # For avatar-live, add recv-only video transceiver
        if is_avatar_live:
            self._pc.addTransceiver("video", direction="recvonly")
            logger.debug("Added video transceiver (recvonly) for avatar-live")

        # Add local audio track if provided
        if local_track:
            self._pc.addTrack(local_track)
            logger.debug("Added local track to peer connection")

    async def _create_and_send_offer(self) -> None:
        logger.debug("Creating offer...")

        offer = await self._pc.createOffer()
        logger.debug(f"Offer SDP:\n{offer.sdp}")

        if self._customize_offer:
            await self._customize_offer(offer)

        await self._pc.setLocalDescription(offer)
        logger.debug("Set local description (offer)")

        await self._send_message(OfferMessage(type="offer", sdp=self._pc.localDescription.sdp))

    async def _receive_messages(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.debug(f"Received {data.get('type', 'unknown')} message")
                        logger.debug(f"Message content: {msg.data}")
                        await self._handle_message(data)
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            if self._on_error:
                self._on_error(e)

    async def _handle_message(self, data: dict) -> None:
        try:
            message = parse_incoming_message(data)
        except Exception as e:
            logger.warning(f"Failed to parse message: {e}")
            return

        if message.type == "answer":
            await self._handle_answer(message.sdp)
        elif message.type == "ice-candidate":
            await self._handle_ice_candidate(message.candidate)
        elif message.type == "session_id":
            logger.debug(f"Session ID: {message.session_id}")
        elif message.type == "prompt_ack":
            self._handle_prompt_ack(message)
        elif message.type == "set_image_ack":
            self._handle_set_image_ack(message)
        elif message.type == "set_ack":
            self._handle_set_ack(message)
        elif message.type == "error":
            self._handle_error(message)
        elif message.type == "ready":
            logger.debug("Received ready signal from server")
        elif message.type == "ice-restart":
            await self._handle_ice_restart(message)

    async def _handle_answer(self, sdp: str) -> None:
        logger.debug("Received answer from server")
        logger.debug(f"Answer SDP:\n{sdp}")

        answer = RTCSessionDescription(sdp=sdp, type="answer")
        await self._pc.setRemoteDescription(answer)
        logger.debug("Set remote description (answer)")

        if self._ice_candidates_queue:
            logger.debug(f"Adding {len(self._ice_candidates_queue)} queued ICE candidates")
            for candidate in self._ice_candidates_queue:
                await self._pc.addIceCandidate(candidate)
            self._ice_candidates_queue.clear()

    async def _handle_ice_candidate(self, candidate_data: IceCandidatePayload) -> None:
        logger.debug(f"Remote ICE candidate: {candidate_data.candidate}")

        candidate = RTCIceCandidate(
            candidate=candidate_data.candidate,
            sdpMLineIndex=candidate_data.sdpMLineIndex,
            sdpMid=candidate_data.sdpMid,
        )

        if self._pc.remoteDescription:
            logger.debug("Adding ICE candidate to peer connection")
            await self._pc.addIceCandidate(candidate)
        else:
            logger.debug("Queuing ICE candidate (no remote description yet)")
            self._ice_candidates_queue.append(candidate)

    def _handle_prompt_ack(self, message: PromptAckMessage) -> None:
        logger.debug(f"Received prompt_ack for: {message.prompt}, success: {message.success}")
        if message.prompt in self._pending_prompts:
            event, result = self._pending_prompts[message.prompt]
            result["success"] = message.success
            result["error"] = message.error
            event.set()

    def _handle_set_image_ack(self, message: SetImageAckMessage) -> None:
        logger.debug(f"Received set_image_ack: success={message.success}, error={message.error}")
        if self._pending_image_set:
            event, result = self._pending_image_set
            result["success"] = message.success
            result["error"] = message.error
            event.set()

    def _handle_set_ack(self, message: SetAckMessage) -> None:
        logger.debug(f"Received set_ack: success={message.success}, error={message.error}")
        if self._pending_set:
            event, result = self._pending_set
            result["success"] = message.success
            result["error"] = message.error
            event.set()

    def _handle_error(self, message: ErrorMessage) -> None:
        logger.error(f"Received error from server: {message.error}")
        error = WebRTCError(message.error)
        if self._on_error:
            self._on_error(error)

    async def _handle_ice_restart(self, message: IceRestartMessage) -> None:
        logger.info("Received ICE restart request from server")
        turn_config = message.turn_config
        # Re-setup peer connection with TURN server
        await self._setup_peer_connection_with_turn(turn_config)

    async def _setup_peer_connection_with_turn(self, turn_config) -> None:
        """Re-setup peer connection with TURN server for ICE restart."""
        from aiortc import RTCConfiguration, RTCIceServer

        ice_servers = [
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(
                urls=[turn_config.server_url],
                username=turn_config.username,
                credential=turn_config.credential,
            ),
        ]
        config = RTCConfiguration(iceServers=ice_servers)

        # Close existing peer connection
        if self._pc:
            await self._pc.close()

        self._pc = RTCPeerConnection(configuration=config)
        logger.debug("Re-created peer connection with TURN server for ICE restart")

        # Re-register callbacks
        @self._pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.debug(f"Received remote track: {track.kind}")
            if self._on_remote_stream:
                self._on_remote_stream(track)

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.debug(f"Peer connection state: {self._pc.connectionState}")
            if self._pc.connectionState == "connected":
                await self._set_state("connected")
            elif self._pc.connectionState in ["failed", "closed"]:
                await self._set_state("disconnected")

        # Re-create and send offer
        await self._create_and_send_offer()

    def register_image_set_wait(self) -> tuple[asyncio.Event, dict]:
        event = asyncio.Event()
        result: dict = {"success": False, "error": None}
        self._pending_image_set = (event, result)
        return event, result

    def unregister_image_set_wait(self) -> None:
        self._pending_image_set = None

    def register_set_wait(self) -> tuple[asyncio.Event, dict]:
        event = asyncio.Event()
        result: dict = {"success": False, "error": None}
        self._pending_set = (event, result)
        return event, result

    def unregister_set_wait(self) -> None:
        self._pending_set = None

    def register_prompt_wait(self, prompt: str) -> tuple[asyncio.Event, dict]:
        event = asyncio.Event()
        result: dict = {"success": False, "error": None}
        self._pending_prompts[prompt] = (event, result)
        return event, result

    def unregister_prompt_wait(self, prompt: str) -> None:
        self._pending_prompts.pop(prompt, None)

    async def _send_message(self, message: OutgoingMessage) -> None:
        if not self._ws or self._ws.closed:
            raise RuntimeError("WebSocket not connected")

        msg_json = message_to_json(message)
        logger.debug(f"Sending {message.type} message")
        logger.debug(f"Message content: {msg_json}")
        await self._ws.send_str(msg_json)

    async def _set_state(self, state: ConnectionState) -> None:
        if self._state != state:
            self._state = state
            logger.debug(f"Connection state changed to: {state}")
            if self._on_state_change:
                self._on_state_change(state)

    async def send(self, message: OutgoingMessage) -> None:
        await self._send_message(message)

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def cleanup(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._pc:
            await self._pc.close()

        if self._ws and not self._ws.closed:
            await self._ws.close()

        if self._session and not self._session.closed:
            await self._session.close()

        await self._set_state("disconnected")
