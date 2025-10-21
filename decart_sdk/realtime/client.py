from typing import Callable, Optional
import logging
from .webrtc_manager import WebRTCManager
from .methods import RealtimeMethods
from .types import ConnectionState
from ..errors import DecartSDKError

logger = logging.getLogger(__name__)


class RealtimeClient:
    def __init__(self, webrtc_manager: Optional[WebRTCManager], session_id: str):
        self._manager = webrtc_manager
        self.session_id = session_id
        self._methods: Optional[RealtimeMethods] = None

        if webrtc_manager:
            self._methods = RealtimeMethods(webrtc_manager)

        self._connection_callbacks: list[Callable[[ConnectionState], None]] = []
        self._error_callbacks: list[Callable[[DecartSDKError], None]] = []

    def _set_manager(self, manager: WebRTCManager) -> None:
        self._manager = manager
        self._methods = RealtimeMethods(manager)

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
        if not self._methods:
            raise RuntimeError("Client not initialized")
        await self._methods.set_prompt(prompt, enrich)

    async def set_mirror(self, enabled: bool) -> None:
        if not self._methods:
            raise RuntimeError("Client not initialized")
        await self._methods.set_mirror(enabled)

    def is_connected(self) -> bool:
        if not self._manager:
            return False
        return self._manager.is_connected()

    def get_connection_state(self) -> ConnectionState:
        if not self._manager:
            return "disconnected"
        return self._manager.get_connection_state()

    async def disconnect(self) -> None:
        if self._manager:
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
