import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decart import DecartClient, ModelDefinition, models

try:
    from decart.realtime.client import RealtimeClient

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REALTIME_AVAILABLE,
    reason="Realtime API not available - install with: pip install decart[realtime]",
)


def _mock_manager(connected: bool = True) -> AsyncMock:
    manager = AsyncMock()
    manager.connect = AsyncMock(return_value=True)
    manager.is_connected = MagicMock(return_value=connected)
    manager.get_connection_state = MagicMock(
        return_value="connected" if connected else "disconnected"
    )
    manager.send_message = AsyncMock()
    manager.cleanup = AsyncMock()
    return manager


def test_realtime_client_available():
    assert REALTIME_AVAILABLE
    assert RealtimeClient is not None


def test_realtime_models_available():
    model = models.realtime("lucy-restyle-2")
    assert model.name == "lucy-restyle-2"
    assert model.fps == 30
    assert model.width == 1280
    assert model.height == 704
    assert model.url_path == "/v1/stream"


@pytest.mark.asyncio
async def test_realtime_connect_wires_livekit_manager_and_session_started_callback():
    from decart.realtime.messages import LiveKitRoomInfoMessage
    from decart.realtime.types import RealtimeConnectOptions
    from decart.types import ModelState, Prompt

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.LiveKitManager") as mock_manager_class:
        mock_manager = _mock_manager()
        mock_manager_class.return_value = mock_manager

        realtime_client = await RealtimeClient.connect(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            local_track=MagicMock(),
            options=RealtimeConnectOptions(
                model=models.realtime("lucy-restyle-2"),
                on_remote_stream=lambda t: None,
                initial_state=ModelState(prompt=Prompt(text="Test", enhance=True)),
            ),
        )

        assert realtime_client.is_connected()
        config = mock_manager_class.call_args.args[0]
        assert config.on_session_started is not None
        assert "livekit_early_room_info=true" in config.livekit_url
        assert config.preferred_video_codec == "h264"

        config.on_session_started(
            LiveKitRoomInfoMessage(
                type="livekit_room_info",
                livekit_url="wss://livekit.example",
                token="lk-token",
                room_name="room-123",
                session_id="session-123",
            )
        )

        assert realtime_client.session_id == "session-123"
        assert realtime_client.subscribe_token is not None
        await realtime_client.disconnect()


@pytest.mark.asyncio
async def test_realtime_connect_accepts_custom_model_definition():
    from decart.realtime.types import RealtimeConnectOptions

    client = DecartClient(api_key="test-key")
    custom_model = ModelDefinition(
        name="lucy_2_rt_preview",
        url_path="/v1/stream",
        fps=20,
        width=1280,
        height=720,
    )

    with (
        patch("decart.realtime.client.LiveKitManager") as mock_manager_class,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = _mock_manager()
        mock_manager_class.return_value = mock_manager

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        realtime_client = await RealtimeClient.connect(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            local_track=MagicMock(),
            options=RealtimeConnectOptions(
                model=custom_model,
                on_remote_stream=lambda t: None,
            ),
        )

        config = mock_manager_class.call_args.args[0]
        assert "model=lucy_2_rt_preview" in config.livekit_url
        assert config.fps == 20

        await realtime_client.disconnect()


async def _connect_and_capture_url(resolution=None) -> str:
    from decart.realtime.types import RealtimeConnectOptions

    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.LiveKitManager") as mock_manager_class,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = _mock_manager()
        mock_manager_class.return_value = mock_manager

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        kwargs = {"resolution": resolution} if resolution is not None else {}
        realtime_client = await RealtimeClient.connect(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            local_track=MagicMock(),
            options=RealtimeConnectOptions(
                model=models.realtime("lucy-2.1"),
                on_remote_stream=lambda t: None,
                **kwargs,
            ),
        )

        config = mock_manager_class.call_args.args[0]
        await realtime_client.disconnect()
        return config.livekit_url


@pytest.mark.asyncio
async def test_realtime_connect_omits_resolution_when_unset():
    url = await _connect_and_capture_url()
    assert "resolution=" not in url
    assert "livekit_early_room_info=true" in url


@pytest.mark.asyncio
async def test_realtime_connect_appends_resolution_720p():
    url = await _connect_and_capture_url("720p")
    assert "&resolution=720p" in url


@pytest.mark.asyncio
async def test_realtime_connect_allows_preferred_video_codec_override():
    from decart.realtime.types import RealtimeConnectOptions

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.LiveKitManager") as mock_manager_class:
        mock_manager = _mock_manager()
        mock_manager_class.return_value = mock_manager

        realtime_client = await RealtimeClient.connect(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            local_track=MagicMock(),
            options=RealtimeConnectOptions(
                model=models.realtime("lucy-restyle-2"),
                on_remote_stream=lambda t: None,
                preferred_video_codec="vp9",
            ),
        )

        config = mock_manager_class.call_args.args[0]
        assert config.preferred_video_codec == "vp9"
        await realtime_client.disconnect()


@pytest.mark.asyncio
async def test_realtime_set_prompt_with_mock():
    from decart.realtime.types import RealtimeConnectOptions

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.LiveKitManager") as mock_manager_class:
        mock_manager = _mock_manager()
        prompt_event = asyncio.Event()
        prompt_result = {"success": True, "error": None}
        mock_manager.register_prompt_wait = MagicMock(return_value=(prompt_event, prompt_result))
        mock_manager.unregister_prompt_wait = MagicMock()
        mock_manager_class.return_value = mock_manager

        realtime_client = await RealtimeClient.connect(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            local_track=MagicMock(),
            options=RealtimeConnectOptions(
                model=models.realtime("lucy-restyle-2"),
                on_remote_stream=lambda t: None,
            ),
        )

        async def set_event():
            await asyncio.sleep(0.01)
            prompt_event.set()

        asyncio.create_task(set_event())
        await realtime_client.set_prompt("New prompt")

        call_args = mock_manager.send_message.call_args.args[0]
        assert call_args.type == "prompt"
        assert call_args.prompt == "New prompt"
        assert call_args.enhance_prompt is True
        mock_manager.unregister_prompt_wait.assert_called_with("New prompt")
        await realtime_client.disconnect()


@pytest.mark.asyncio
async def test_subscribe_fetches_watch_stream_credentials_and_connects_room_directly():
    from decart.realtime.messages import LiveKitRoomInfoMessage
    from decart.realtime.subscribe import SubscribeOptions, encode_subscribe_token

    client = DecartClient(api_key="test-key")
    room_info = LiveKitRoomInfoMessage(
        type="livekit_room_info",
        livekit_url="wss://livekit.example",
        token="lk-token",
        room_name="room-123",
        session_id="room-123",
    )

    with (
        patch(
            "decart.realtime.client._fetch_watch_stream_credentials",
            AsyncMock(return_value=room_info),
        ),
        patch("decart.realtime.client.LiveKitManager") as mock_manager_class,
    ):
        mock_manager = _mock_manager()
        mock_manager_class.return_value = mock_manager

        sub_client = await RealtimeClient.subscribe(
            base_url=client.realtime_base_url,
            api_key=client.api_key,
            options=SubscribeOptions(
                token=encode_subscribe_token("room-123"),
                on_remote_stream=lambda t: None,
            ),
        )

        assert sub_client.is_connected()
        config = mock_manager_class.call_args.args[0]
        assert config.room_info == room_info
        mock_manager.connect.assert_awaited_once_with(None)


def test_subscribe_token_round_trip_uses_room_name():
    from decart.realtime.subscribe import decode_subscribe_token, encode_subscribe_token

    token = encode_subscribe_token("room-123")
    assert decode_subscribe_token(token).room_name == "room-123"


def test_legacy_subscribe_token_is_rejected():
    from decart.realtime.subscribe import decode_subscribe_token

    legacy = json.dumps({"sid": "sid", "ip": "1.2.3.4", "port": 8080}).encode()
    import base64

    token = base64.urlsafe_b64encode(legacy).decode()
    with pytest.raises(ValueError, match="Invalid subscribe token"):
        decode_subscribe_token(token)


def test_livekit_messages_serialize_join_and_passthrough():
    from decart.realtime.messages import LiveKitJoinMessage, SetAvatarImageMessage, message_to_json

    assert message_to_json(LiveKitJoinMessage(type="livekit_join")) == '{"type":"livekit_join"}'

    passthrough = json.loads(
        message_to_json(SetAvatarImageMessage(type="set_image", image_data=None, prompt=None))
    )
    assert passthrough == {"type": "set_image", "image_data": None, "prompt": None}


@pytest.mark.asyncio
async def test_livekit_connection_join_sends_join_and_resolves_room_info():
    from decart.realtime.livekit_connection import LiveKitConnection

    class FakeWebSocket:
        closed = False

        def __init__(self):
            self.sent: list[str] = []

        async def send_str(self, message: str):
            self.sent.append(message)

    ws = FakeWebSocket()
    connection = LiveKitConnection()
    connection._ws = ws  # type: ignore[attr-defined]

    async def resolve_room_info():
        await asyncio.sleep(0.01)
        await connection._handle_message(
            {
                "type": "livekit_room_info",
                "livekit_url": "wss://livekit.example",
                "token": "lk-token",
                "room_name": "room-123",
                "session_id": "session-123",
            }
        )

    asyncio.create_task(resolve_room_info())
    room_info = await connection._join_livekit_room(timeout=1)

    assert json.loads(ws.sent[0]) == {"type": "livekit_join"}
    assert room_info.room_name == "room-123"
    assert room_info.session_id == "session-123"


@pytest.mark.asyncio
async def test_livekit_connection_can_connect_directly_with_room_info():
    from decart.realtime.livekit_connection import LiveKitConnection
    from decart.realtime.messages import LiveKitRoomInfoMessage

    connection = LiveKitConnection()
    connection._connect_signaling = AsyncMock()  # type: ignore[method-assign]
    connection._join_livekit_room = AsyncMock()  # type: ignore[method-assign]
    connection._connect_room = AsyncMock()  # type: ignore[method-assign]
    connection._wait_until_connected = AsyncMock()  # type: ignore[method-assign]
    room_info = LiveKitRoomInfoMessage(
        type="livekit_room_info",
        livekit_url="wss://livekit.example",
        token="lk-token",
        room_name="room-123",
        session_id="session-123",
    )

    await connection.connect(url="", local_track=None, timeout=1, room_info=room_info)

    connection._connect_signaling.assert_not_called()
    connection._join_livekit_room.assert_not_called()
    connection._connect_room.assert_awaited_once_with(room_info, None, "h264")


@pytest.mark.asyncio
async def test_livekit_connection_filters_remote_tracks_to_inference_video():
    from livekit import rtc

    from decart.realtime.livekit_connection import LiveKitConnection
    from decart.realtime.messages import LiveKitRoomInfoMessage

    handlers = {}

    class FakeParticipant:
        def __init__(self, identity):
            self.identity = identity

    class FakeLocalParticipant:
        def __init__(self):
            self.publish_track = AsyncMock()

    class FakeRoom:
        def __init__(self):
            self.local_participant = FakeLocalParticipant()

        def on(self, event):
            def decorator(callback):
                handlers[event] = callback
                return callback

            return decorator

        async def connect(self, _url, _token):
            return None

    remote_tracks = []
    connection = LiveKitConnection(
        on_remote_stream=lambda track, _pub, _p: remote_tracks.append(track)
    )

    with patch("decart.realtime.livekit_connection.rtc.Room", FakeRoom):
        await connection._connect_room(
            LiveKitRoomInfoMessage(
                type="livekit_room_info",
                livekit_url="wss://livekit.example",
                token="lk-token",
                room_name="room-123",
                session_id="session-123",
            ),
            local_track=MagicMock(),
        )

    video_track = MagicMock()
    video_track.kind = rtc.TrackKind.KIND_VIDEO
    audio_track = MagicMock()
    audio_track.kind = rtc.TrackKind.KIND_AUDIO

    handlers["track_subscribed"](video_track, MagicMock(), FakeParticipant("viewer"))
    handlers["track_subscribed"](audio_track, MagicMock(), FakeParticipant("inference-server-1"))
    handlers["track_subscribed"](video_track, MagicMock(), FakeParticipant("inference-server-1"))

    assert remote_tracks == [video_track]


@pytest.mark.asyncio
async def test_livekit_connection_publishes_local_track_with_preferred_codec():
    from livekit import rtc

    from decart.realtime.livekit_connection import LiveKitConnection
    from decart.realtime.messages import LiveKitRoomInfoMessage

    handlers = {}

    class FakeLocalParticipant:
        def __init__(self):
            self.publish_track = AsyncMock()

    class FakeRoom:
        def __init__(self):
            self.local_participant = FakeLocalParticipant()

        def on(self, event):
            def decorator(callback):
                handlers[event] = callback
                return callback

            return decorator

        async def connect(self, _url, _token):
            return None

    local_track = MagicMock()
    connection = LiveKitConnection()

    with patch("decart.realtime.livekit_connection.rtc.Room", FakeRoom):
        await connection._connect_room(
            LiveKitRoomInfoMessage(
                type="livekit_room_info",
                livekit_url="wss://livekit.example",
                token="lk-token",
                room_name="room-123",
                session_id="session-123",
            ),
            local_track=local_track,
            preferred_video_codec="vp9",
        )

    room = connection.room
    publish_options = room.local_participant.publish_track.call_args.args[1]
    assert publish_options.video_codec == rtc.VideoCodec.VP9


@pytest.mark.asyncio
async def test_fetch_watch_stream_credentials_uses_http_base_and_api_key():
    from decart.realtime.client import _fetch_watch_stream_credentials

    class FakeResponse:
        status = 200
        reason = "OK"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def text(self):
            return json.dumps(
                {
                    "livekit_url": "wss://livekit.example",
                    "token": "lk-token",
                    "room_name": "room-123",
                }
            )

    class FakeSession:
        def __init__(self):
            self.post_calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def post(self, url, headers):
            self.post_calls.append((url, headers))
            return FakeResponse()

    fake_session = FakeSession()

    with patch("decart.realtime.client.aiohttp.ClientSession", return_value=fake_session):
        room_info = await _fetch_watch_stream_credentials(
            base_url="wss://api3.decart.ai",
            api_key="test-key",
            room_name="room-123",
        )

    assert fake_session.post_calls == [
        ("https://api3.decart.ai/watch-stream/room-123", {"x-api-key": "test-key"})
    ]
    assert room_info.livekit_url == "wss://livekit.example"
    assert room_info.session_id == "room-123"
