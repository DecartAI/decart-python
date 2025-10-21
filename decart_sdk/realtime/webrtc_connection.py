import asyncio
import json
import logging
from typing import Optional, Callable
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
from .messages import (
    parse_incoming_message,
    message_to_json,
    OfferMessage,
    IceCandidateMessage,
    IceCandidatePayload,
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

    async def connect(self, url: str, local_track: MediaStreamTrack, timeout: float = 30) -> None:
        try:
            await self._set_state("connecting")

            ws_url = url.replace("https://", "wss://").replace("http://", "ws://")

            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(ws_url)

            self._ws_task = asyncio.create_task(self._receive_messages())

            await self._setup_peer_connection(local_track)

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

    async def _setup_peer_connection(self, local_track: MediaStreamTrack) -> None:
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
