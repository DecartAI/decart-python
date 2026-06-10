from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass

from .types import ConnectionState

if TYPE_CHECKING:
    from livekit.rtc import RemoteVideoTrack
    from .livekit_manager import LiveKitManager

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    room_name: str


def encode_subscribe_token(room_name: str) -> str:
    payload = json.dumps({"room_name": room_name})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_subscribe_token(token: str) -> TokenPayload:
    try:
        raw = base64.urlsafe_b64decode(token).decode()
        data = json.loads(raw)
        if not data.get("room_name"):
            raise ValueError("Invalid subscribe token format")
        return TokenPayload(room_name=data["room_name"])
    except Exception:
        raise ValueError("Invalid subscribe token")


@dataclass
class SubscribeOptions:
    token: str
    on_remote_stream: Callable[["RemoteVideoTrack"], None]


class SubscribeClient:
    def __init__(self, manager: "LiveKitManager"):
        self._manager = manager
        self._connection_callbacks: list[Callable[[ConnectionState], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []
        self._buffering = True
        self._buffer: list[tuple[str, object]] = []

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
        self._buffer.clear()

    def _dispatch_connection_change(self, state: ConnectionState) -> None:
        for callback in list(self._connection_callbacks):
            try:
                callback(state)
            except Exception as e:
                logger.exception(f"Error in subscribe connection_change callback: {e}")

    def _dispatch_error(self, error: Exception) -> None:
        for callback in list(self._error_callbacks):
            try:
                callback(error)
            except Exception as e:
                logger.exception(f"Error in subscribe error callback: {e}")

    def _emit_connection_change(self, state: ConnectionState) -> None:
        if self._buffering:
            self._buffer.append(("connection_change", state))
        else:
            self._dispatch_connection_change(state)

    def _emit_error(self, error: Exception) -> None:
        if self._buffering:
            self._buffer.append(("error", error))
        else:
            self._dispatch_error(error)

    def is_connected(self) -> bool:
        return self._manager.is_connected()

    def get_connection_state(self) -> ConnectionState:
        return self._manager.get_connection_state()

    async def disconnect(self) -> None:
        self._buffering = False
        self._buffer.clear()
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
