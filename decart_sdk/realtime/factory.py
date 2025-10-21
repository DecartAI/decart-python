import uuid

try:
    from aiortc import MediaStreamTrack

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    MediaStreamTrack = None  # type: ignore

from .client import RealtimeClient
from .webrtc_manager import WebRTCManager, WebRTCConfiguration
from .types import RealtimeConnectOptions
from ..errors import create_webrtc_error


class RealtimeClientFactory:
    def __init__(self, base_url: str, api_key: str):
        if not WEBRTC_AVAILABLE:
            raise ImportError(
                "aiortc is required for Realtime API. "
                "Install with: pip install decart-sdk[realtime]"
            )

        self._base_url = base_url
        self._api_key = api_key

    async def connect(
        self, local_track: MediaStreamTrack, options: RealtimeConnectOptions
    ) -> RealtimeClient:
        session_id = str(uuid.uuid4())

        ws_url = f"{self._base_url}{options.model.url_path}"
        ws_url += f"?api_key={self._api_key}&model={options.model.name}"

        client = RealtimeClient(webrtc_manager=None, session_id=session_id)

        config = WebRTCConfiguration(
            webrtc_url=ws_url,
            api_key=self._api_key,
            session_id=session_id,
            fps=options.model.fps,
            on_remote_stream=options.on_remote_stream,
            on_connection_state_change=lambda state: client._emit_connection_change(state),
            on_error=lambda error: client._emit_error(create_webrtc_error(error)),
            initial_state=options.initial_state,
            customize_offer=options.customize_offer,
        )

        manager = WebRTCManager(config)
        client._set_manager(manager)

        try:
            await manager.connect(local_track)

            # Send initial state if provided
            if options.initial_state:
                if options.initial_state.prompt:
                    await client.set_prompt(
                        options.initial_state.prompt.text,
                        enrich=options.initial_state.prompt.enrich,
                    )
                if options.initial_state.mirror is not None:
                    await client.set_mirror(options.initial_state.mirror)
        except Exception as e:
            raise create_webrtc_error(e)

        return client
