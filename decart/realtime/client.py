from typing import Callable, Optional
import asyncio
import base64
import logging
import uuid
import aiohttp
from aiortc import MediaStreamTrack

from .webrtc_manager import WebRTCManager, WebRTCConfiguration
from .messages import PromptMessage, SetAvatarImageMessage
from .types import ConnectionState, RealtimeConnectOptions
from ..types import FileInput
from ..errors import DecartSDKError, InvalidInputError, WebRTCError
from ..process.request import file_input_to_bytes

logger = logging.getLogger(__name__)


class RealtimeClient:
    def __init__(
        self,
        manager: WebRTCManager,
        session_id: str,
        http_session: Optional[aiohttp.ClientSession] = None,
        is_avatar_live: bool = False,
    ):
        self._manager = manager
        self.session_id = session_id
        self._http_session = http_session
        self._is_avatar_live = is_avatar_live
        self._connection_callbacks: list[Callable[[ConnectionState], None]] = []
        self._error_callbacks: list[Callable[[DecartSDKError], None]] = []

    @classmethod
    async def connect(
        cls,
        base_url: str,
        api_key: str,
        local_track: Optional[MediaStreamTrack],
        options: RealtimeConnectOptions,
        integration: Optional[str] = None,
    ) -> "RealtimeClient":
        session_id = str(uuid.uuid4())
        ws_url = f"{base_url}{options.model.url_path}"
        ws_url += f"?api_key={api_key}&model={options.model.name}"

        is_avatar_live = options.model.name == "avatar-live"

        config = WebRTCConfiguration(
            webrtc_url=ws_url,
            api_key=api_key,
            session_id=session_id,
            fps=options.model.fps,
            on_remote_stream=options.on_remote_stream,
            on_connection_state_change=None,
            on_error=None,
            initial_state=options.initial_state,
            customize_offer=options.customize_offer,
            integration=integration,
            is_avatar_live=is_avatar_live,
        )

        # Create HTTP session for file conversions
        http_session = aiohttp.ClientSession()

        manager = WebRTCManager(config)
        client = cls(
            manager=manager,
            session_id=session_id,
            http_session=http_session,
            is_avatar_live=is_avatar_live,
        )

        config.on_connection_state_change = client._emit_connection_change
        config.on_error = lambda error: client._emit_error(WebRTCError(str(error), cause=error))

        try:
            # For avatar-live, convert and send avatar image before WebRTC connection
            avatar_image_base64: Optional[str] = None
            if is_avatar_live and options.avatar:
                image_bytes, _ = await file_input_to_bytes(
                    options.avatar.avatar_image, http_session
                )
                avatar_image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            await manager.connect(local_track, avatar_image_base64=avatar_image_base64)

            if options.initial_state:
                if options.initial_state.prompt:
                    await client.set_prompt(
                        options.initial_state.prompt.text,
                        enrich=options.initial_state.prompt.enrich,
                    )
        except Exception as e:
            await http_session.close()
            raise WebRTCError(str(e), cause=e)

        return client

    def _emit_connection_change(self, state: ConnectionState) -> None:
        for callback in self._connection_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.exception(f"Error in connection_change callback: {e}")

    def _emit_error(self, error: DecartSDKError) -> None:
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.exception(f"Error in error callback: {e}")

    async def set_prompt(self, prompt: str, enrich: bool = True) -> None:
        if not prompt or not prompt.strip():
            raise InvalidInputError("Prompt cannot be empty")

        event, result = self._manager.register_prompt_wait(prompt)

        try:
            await self._manager.send_message(
                PromptMessage(type="prompt", prompt=prompt, enhance_prompt=enrich)
            )

            try:
                await asyncio.wait_for(event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                raise DecartSDKError("Prompt acknowledgment timed out")

            if not result["success"]:
                raise DecartSDKError(result["error"] or "Prompt failed")
        finally:
            self._manager.unregister_prompt_wait(prompt)

    async def set_image(self, image: FileInput) -> None:
        """Set or update the avatar image.

        Only available for avatar-live model.

        Args:
            image: The image to set. Can be bytes, Path, URL string, or file-like object.

        Raises:
            InvalidInputError: If not using avatar-live model or image is invalid.
            DecartSDKError: If the server fails to acknowledge the image.
        """
        if not self._is_avatar_live:
            raise InvalidInputError("set_image() is only available for avatar-live model")

        if not self._http_session:
            raise InvalidInputError("HTTP session not available")

        # Convert image to base64
        image_bytes, _ = await file_input_to_bytes(image, self._http_session)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        event, result = self._manager.register_image_set_wait()

        try:
            await self._manager.send_message(
                SetAvatarImageMessage(type="set_image", image_data=image_base64)
            )

            try:
                await asyncio.wait_for(event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                raise DecartSDKError("Image set acknowledgment timed out")

            if not result["success"]:
                raise DecartSDKError(result.get("status") or "Failed to set avatar image")
        finally:
            self._manager.unregister_image_set_wait()

    def is_connected(self) -> bool:
        return self._manager.is_connected()

    def get_connection_state(self) -> ConnectionState:
        return self._manager.get_connection_state()

    async def disconnect(self) -> None:
        await self._manager.cleanup()
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    def on(self, event: str, callback: Callable) -> None:
        if event == "connection_change":
            self._connection_callbacks.append(callback)
        elif event == "error":
            self._error_callbacks.append(callback)

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
