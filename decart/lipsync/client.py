import asyncio
import websockets
from typing import Optional, Tuple
from .messages import (
    LipsyncClientMessage,
    LipsyncServerMessage,
    LipsyncServerMessageAdapter,
    LipsyncConfigMessage,
    LipsyncConfigAckMessage,
    LipsyncAudioInputMessage,
    LipsyncVideoInputMessage,
    LipsyncInterruptAudioMessage,
    LipsyncSyncedOutputMessage,
    LipsyncErrorMessage,
)
import fractions
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RealtimeLipsyncClient:
    DECART_LIPSYNC_ENDPOINT = "/router/lipsync/ws"
    VIDEO_FPS = 25

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.decart.ai",
        audio_sample_rate: int = 16000,
        video_fps: int = VIDEO_FPS,
        sync_latency: float = 0.0,
    ):
        """
        Args:
            api_key: The API key for the Decart Lipsync server
            url: The URL of the Decart Lipsync server
            audio_sample_rate: The sample rate of the audio
            video_fps: The FPS of the video
            sync_latency: Delay next frame up to this many seconds, to account for variable latency
        """
        self._url = f"{base_url}{self.DECART_LIPSYNC_ENDPOINT}".replace(
            "https://", "wss://"
        ).replace("http://", "ws://")
        self._api_key = api_key
        self._audio_sample_rate = audio_sample_rate
        self._video_fps = video_fps
        self._sync_latency = sync_latency

        self._websocket: Optional[websockets.ClientConnection] = None
        self._out_queue = asyncio.Queue()
        self._response_handling_task: Optional[asyncio.Task] = None

        self._video_frame_interval = fractions.Fraction(1, video_fps)
        self._video_out_frame_index = 0
        self._video_out_start_time = 0

    async def _recv(self) -> LipsyncServerMessage:
        response = await self._websocket.recv()
        return LipsyncServerMessageAdapter.validate_json(response)

    async def _send(self, message: LipsyncClientMessage):
        msg = message.model_dump_json()
        await self._websocket.send(msg)

    async def _handle_server_responses(self):
        try:
            while self._websocket is not None:
                response = await self._recv()
                if isinstance(response, LipsyncSyncedOutputMessage):
                    await self._out_queue.put(response)
                elif isinstance(response, LipsyncErrorMessage):
                    logger.error(f"Lipsync server error: {response.message}")
                    raise Exception(response.message)
                else:
                    logger.error(f"Unknown response from lipsync server: {response}")
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosedOK:
            logger.debug("Connection closed by server")

    async def _decode_video_frame(self, video_frame: bytes) -> bytes:
        def _decode_video_frame_sync(video_frame: bytes) -> bytes:
            nparr = np.frombuffer(video_frame, np.uint8)
            video_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return video_frame

        return await asyncio.to_thread(_decode_video_frame_sync, video_frame)

    async def _encode_video_frame(self, image: np.ndarray) -> bytes:
        def _encode_video_frame_sync(image: np.ndarray) -> bytes:
            success, encoded_image = cv2.imencode(".jpeg", image)
            if not success:
                raise Exception("Failed to encode video frame as JPEG")
            return encoded_image.tobytes()

        return await asyncio.to_thread(_encode_video_frame_sync, image)

    async def _decode_audio_frame(self, audio_frame: bytes) -> bytes:
        return audio_frame

    async def connect(self):
        logger.debug(f"Connecting to lipsync server at {self._url}")
        self._websocket = await websockets.connect(f"{self._url}?api_key={self._api_key}")
        logger.debug("WebSocket connected")
        # Initial handshake
        await self._send(
            LipsyncConfigMessage(
                video_fps=self._video_fps,
                audio_sample_rate=self._audio_sample_rate,
            )
        )
        logger.debug("Configuration sent")
        response = await self._recv()
        if not isinstance(response, LipsyncConfigAckMessage):
            raise Exception(f"Configuration not acknowledged by server: {response}")
        logger.debug("Configuration acknowledged")

        self._response_handling_task = asyncio.create_task(self._handle_server_responses())

        logger.debug("Connected to lipsync server")

    async def disconnect(self):
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

        if self._response_handling_task is not None:
            self._response_handling_task.cancel()
            try:
                await self._response_handling_task
            except asyncio.CancelledError:
                pass
            self._response_handling_task = None

    async def send_audio(self, audio_data: bytes):
        await self._send(LipsyncAudioInputMessage(audio_data=audio_data))

    async def send_video_frame_bytes(self, video_frame_bytes: bytes):
        await self._send(LipsyncVideoInputMessage(video_frame=video_frame_bytes))

    async def send_video_frame(self, image: np.ndarray):
        encoded_image = await self._encode_video_frame(image)
        await self.send_video_frame_bytes(encoded_image)

    async def interrupt_audio(self):
        await self._send(LipsyncInterruptAudioMessage())

    async def get_synced_output(self, timeout: Optional[float] = None) -> Tuple[bytes, bytes]:
        synced_output: LipsyncSyncedOutputMessage = await asyncio.wait_for(
            self._out_queue.get(), timeout=timeout
        )

        video_frame = await self._decode_video_frame(synced_output.video_frame)
        audio_frame = await self._decode_audio_frame(synced_output.audio_frame)

        if self._video_out_frame_index == 0:
            self._video_out_start_time = time.time() + self._sync_latency

        time_til_frame = (
            self._video_out_start_time
            + (self._video_out_frame_index * self._video_frame_interval)
            - time.time()
        )
        if time_til_frame > 0:
            await asyncio.sleep(time_til_frame)

        return video_frame, audio_frame
