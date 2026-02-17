from typing import Callable, Optional, Union
import asyncio
import base64
import logging
from pathlib import Path
from urllib.parse import urlparse, quote
import aiohttp
from aiortc import MediaStreamTrack
from pydantic import BaseModel

from .webrtc_manager import WebRTCManager, WebRTCConfiguration
from .messages import PromptMessage, SessionIdMessage, GenerationTickMessage
from .subscribe import (
    SubscribeClient,
    SubscribeOptions,
    encode_subscribe_token,
    decode_subscribe_token,
)
from .types import ConnectionState, RealtimeConnectOptions
from ..types import FileInput
from ..models import RealTimeModels
from ..errors import DecartSDKError, InvalidInputError, WebRTCError
from ..process.request import file_input_to_bytes

logger = logging.getLogger(__name__)

PROMPT_TIMEOUT_S = 15.0
UPDATE_TIMEOUT_S = 30.0


class SetInput(BaseModel):
    prompt: Optional[str] = None
    enhance: bool = True
    image: Optional[Union[bytes, str]] = None


async def _image_to_base64(
    image: Union[bytes, str, Path],
    http_session: aiohttp.ClientSession,
) -> str:
    if isinstance(image, Path):
        image_bytes, _ = await file_input_to_bytes(image, http_session)
        return base64.b64encode(image_bytes).decode("utf-8")

    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    if isinstance(image, str):
        parsed = urlparse(image)

        if parsed.scheme == "data":
            return image.split(",", 1)[1]

        if parsed.scheme in ("http", "https"):
            async with http_session.get(image) as resp:
                resp.raise_for_status()
                data = await resp.read()
                return base64.b64encode(data).decode("utf-8")

        if Path(image).exists():
            image_bytes, _ = await file_input_to_bytes(image, http_session)
            return base64.b64encode(image_bytes).decode("utf-8")

        # Non-URL, non-file string — treat as raw base64 (matches TS SDK behavior)
        return image


class RealtimeClient:
    def __init__(
        self,
        manager: WebRTCManager,
        http_session: Optional[aiohttp.ClientSession] = None,
        model_name: Optional[str] = None,
    ):
        self._manager = manager
        self._http_session = http_session
        self._model_name = model_name
        self._connection_callbacks: list[Callable[[ConnectionState], None]] = []
        self._error_callbacks: list[Callable[[DecartSDKError], None]] = []
        self._generation_tick_callbacks: list[Callable[[GenerationTickMessage], None]] = []
        self._session_id: Optional[str] = None
        self._subscribe_token: Optional[str] = None
        self._buffering = True
        self._buffer: list[tuple[str, object]] = []

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def subscribe_token(self) -> Optional[str]:
        return self._subscribe_token

    def _handle_session_id(self, msg: SessionIdMessage) -> None:
        self._session_id = msg.session_id
        self._subscribe_token = encode_subscribe_token(
            msg.session_id, msg.server_ip, msg.server_port
        )

    @classmethod
    async def connect(
        cls,
        base_url: str,
        api_key: str,
        local_track: Optional[MediaStreamTrack],
        options: RealtimeConnectOptions,
        integration: Optional[str] = None,
    ) -> "RealtimeClient":
        ws_url = f"{base_url}{options.model.url_path}"
        ws_url += f"?api_key={quote(api_key)}&model={quote(options.model.name)}"

        model_name: RealTimeModels = options.model.name  # type: ignore[assignment]

        config = WebRTCConfiguration(
            webrtc_url=ws_url,
            api_key=api_key,
            session_id="",
            fps=options.model.fps,
            on_remote_stream=options.on_remote_stream,
            on_connection_state_change=None,
            on_error=None,
            on_session_id=None,
            initial_state=options.initial_state,
            customize_offer=options.customize_offer,
            integration=integration,
            model_name=model_name,
        )

        # Create HTTP session for file conversions
        http_session = aiohttp.ClientSession()

        manager = WebRTCManager(config)
        client = cls(
            manager=manager,
            http_session=http_session,
            model_name=model_name,
        )

        config.on_connection_state_change = client._emit_connection_change
        config.on_error = lambda error: client._emit_error(WebRTCError(str(error), cause=error))
        config.on_session_id = client._handle_session_id
        config.on_generation_tick = client._emit_generation_tick

        try:
            initial_image: Optional[str] = None
            if options.initial_state and options.initial_state.image:
                initial_image = await _image_to_base64(options.initial_state.image, http_session)

            initial_prompt: Optional[dict] = None
            if options.initial_state and options.initial_state.prompt:
                initial_prompt = {
                    "text": options.initial_state.prompt.text,
                    "enhance": options.initial_state.prompt.enhance,
                }

            await manager.connect(
                local_track,
                initial_image=initial_image,
                initial_prompt=initial_prompt,
            )
        except Exception as e:
            await manager.cleanup()
            await http_session.close()
            raise WebRTCError(str(e), cause=e)

        client._flush()
        return client

    @classmethod
    async def subscribe(
        cls,
        base_url: str,
        api_key: str,
        options: SubscribeOptions,
        integration: Optional[str] = None,
    ) -> SubscribeClient:
        token_data = decode_subscribe_token(options.token)
        subscribe_url = (
            f"{base_url}/subscribe/{quote(token_data.sid)}"
            f"?IP={quote(token_data.ip)}"
            f"&port={quote(str(token_data.port))}"
            f"&api_key={quote(api_key)}"
        )

        config = WebRTCConfiguration(
            webrtc_url=subscribe_url,
            api_key=api_key,
            session_id=token_data.sid,
            fps=0,
            on_remote_stream=options.on_remote_stream,
            on_connection_state_change=None,
            on_error=None,
            integration=integration,
        )

        manager = WebRTCManager(config)
        sub_client = SubscribeClient(manager)

        config.on_connection_state_change = sub_client._emit_connection_change
        config.on_error = sub_client._emit_error

        try:
            await manager.connect(None)
        except Exception as e:
            await manager.cleanup()
            raise WebRTCError(str(e), cause=e)

        sub_client._flush()
        return sub_client

    def _flush(self) -> None:
        # Defer to next tick so caller can register handlers before buffered events fire
        asyncio.get_running_loop().call_soon(self._do_flush)

    def _do_flush(self) -> None:
        self._buffering = False
        for event, data in self._buffer:
            if event == "connection_change":
                self._dispatch_connection_change(data)  # type: ignore[arg-type]
            elif event == "error":
                self._dispatch_error(data)  # type: ignore[arg-type]
            elif event == "generation_tick":
                self._dispatch_generation_tick(data)  # type: ignore[arg-type]
        self._buffer.clear()

    def _dispatch_connection_change(self, state: ConnectionState) -> None:
        for callback in list(self._connection_callbacks):
            try:
                callback(state)
            except Exception as e:
                logger.exception(f"Error in connection_change callback: {e}")

    def _dispatch_error(self, error: DecartSDKError) -> None:
        for callback in list(self._error_callbacks):
            try:
                callback(error)
            except Exception as e:
                logger.exception(f"Error in error callback: {e}")

    def _emit_connection_change(self, state: ConnectionState) -> None:
        if self._buffering:
            self._buffer.append(("connection_change", state))
        else:
            self._dispatch_connection_change(state)

    def _emit_error(self, error: DecartSDKError) -> None:
        if self._buffering:
            self._buffer.append(("error", error))
        else:
            self._dispatch_error(error)

    def _dispatch_generation_tick(self, message: GenerationTickMessage) -> None:
        for callback in list(self._generation_tick_callbacks):
            try:
                callback(message)
            except Exception as e:
                logger.exception(f"Error in generation_tick callback: {e}")

    def _emit_generation_tick(self, message: GenerationTickMessage) -> None:
        if self._buffering:
            self._buffer.append(("generation_tick", message))
        else:
            self._dispatch_generation_tick(message)

    async def set(self, input: SetInput) -> None:
        if input.prompt is None and input.image is None:
            raise InvalidInputError("At least one of 'prompt' or 'image' must be provided")

        if input.prompt is not None and not input.prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        image_base64: Optional[str] = None
        if input.image is not None:
            if not self._http_session:
                raise InvalidInputError("HTTP session not available")
            image_base64 = await _image_to_base64(input.image, self._http_session)

        await self._manager.set_image(
            image_base64,
            {
                "prompt": input.prompt,
                "enhance": input.enhance,
                "timeout": UPDATE_TIMEOUT_S,
            },
        )

    async def set_prompt(
        self,
        prompt: str,
        enhance: bool = True,
        enrich: Optional[bool] = None,
    ) -> None:
        if enrich is not None:
            import warnings

            warnings.warn(
                "set_prompt(enrich=...) is deprecated, use set_prompt(enhance=...) instead",
                DeprecationWarning,
                stacklevel=2,
            )
            enhance = enrich
        if not prompt or not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        event, result = self._manager.register_prompt_wait(prompt)

        try:
            await self._manager.send_message(
                PromptMessage(type="prompt", prompt=prompt, enhance_prompt=enhance)
            )

            try:
                await asyncio.wait_for(event.wait(), timeout=PROMPT_TIMEOUT_S)
            except asyncio.TimeoutError:
                raise DecartSDKError("Prompt acknowledgment timed out")

            if not result["success"]:
                raise DecartSDKError(result["error"] or "Prompt failed")
        finally:
            self._manager.unregister_prompt_wait(prompt)

    async def set_image(
        self,
        image: Optional[FileInput],
        prompt: Optional[str] = None,
        enhance: bool = True,
        timeout: float = UPDATE_TIMEOUT_S,
    ) -> None:
        image_base64: Optional[str] = None
        if image is not None:
            if not self._http_session:
                raise InvalidInputError("HTTP session not available")
            image_bytes, _ = await file_input_to_bytes(image, self._http_session)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        opts: dict = {"timeout": timeout}
        if prompt is not None:
            opts["prompt"] = prompt
            opts["enhance"] = enhance

        await self._manager.set_image(image_base64, opts)

    def is_connected(self) -> bool:
        return self._manager.is_connected()

    def get_connection_state(self) -> ConnectionState:
        return self._manager.get_connection_state()

    async def disconnect(self) -> None:
        self._buffering = False
        self._buffer.clear()
        await self._manager.cleanup()
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    def on(self, event: str, callback: Callable) -> None:
        if event == "connection_change":
            self._connection_callbacks.append(callback)
        elif event == "error":
            self._error_callbacks.append(callback)
        elif event == "generation_tick":
            self._generation_tick_callbacks.append(callback)

    def off(self, event: str, callback: Callable) -> None:
        if event == "connection_change":
            try:
                self._connection_callbacks.remove(callback)
            except ValueError:
                pass
        elif event == "error":
            try:
                self._error_callbacks.remove(callback)
            except ValueError:
                pass
        elif event == "generation_tick":
            try:
                self._generation_tick_callbacks.remove(callback)
            except ValueError:
                pass
