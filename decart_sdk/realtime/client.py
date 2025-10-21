from typing import Callable
import logging
import uuid
from aiortc import MediaStreamTrack

from .webrtc_manager import WebRTCManager, WebRTCConfiguration
from .messages import PromptMessage, SwitchCameraMessage
from .types import ConnectionState, RealtimeConnectOptions
from ..errors import DecartSDKError, InvalidInputError, WebRTCError

logger = logging.getLogger(__name__)


class RealtimeClient:
    def __init__(self, manager: WebRTCManager, session_id: str):
        self._manager = manager
        self.session_id = session_id
        self._connection_callbacks: list[Callable[[ConnectionState], None]] = []
        self._error_callbacks: list[Callable[[DecartSDKError], None]] = []

    @classmethod
    async def connect(
        cls,
        base_url: str,
        api_key: str,
        local_track: MediaStreamTrack,
        options: RealtimeConnectOptions,
    ) -> "RealtimeClient":
        session_id = str(uuid.uuid4())
        ws_url = f"{base_url}{options.model.url_path}"
        ws_url += f"?api_key={api_key}&model={options.model.name}"

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
        )

        manager = WebRTCManager(config)
        client = cls(manager=manager, session_id=session_id)

        config.on_connection_state_change = client._emit_connection_change
        config.on_error = lambda error: client._emit_error(WebRTCError(str(error), cause=error))

        try:
            await manager.connect(local_track)

            if options.initial_state:
                if options.initial_state.prompt:
                    await client.set_prompt(
                        options.initial_state.prompt.text,
                        enrich=options.initial_state.prompt.enrich,
                    )
                if options.initial_state.mirror is not None:
                    await client.set_mirror(options.initial_state.mirror)
        except Exception as e:
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
        await self._manager.send_message(PromptMessage(type="prompt", prompt=prompt))

    async def set_mirror(self, enabled: bool) -> None:
        rotate_y = 2 if enabled else 0
        await self._manager.send_message(
            SwitchCameraMessage(type="switch_camera", rotateY=rotate_y)
        )

    def is_connected(self) -> bool:
        return self._manager.is_connected()

    def get_connection_state(self) -> ConnectionState:
        return self._manager.get_connection_state()

    async def disconnect(self) -> None:
        await self._manager.cleanup()

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
