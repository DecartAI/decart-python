import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decart import DecartClient, models

try:
    from decart.realtime.client import RealtimeClient

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REALTIME_AVAILABLE,
    reason="Realtime API not available - install with: pip install decart[realtime]",
)


def test_realtime_client_available():
    """Test that realtime client is available when aiortc is installed"""
    assert REALTIME_AVAILABLE
    assert RealtimeClient is not None


def test_realtime_models_available():
    """Test that realtime models are available"""
    model = models.realtime("mirage")
    assert model.name == "mirage"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 704

    model2 = models.realtime("lucy_v2v_720p_rt")
    assert model2.name == "lucy_v2v_720p_rt"


@pytest.mark.asyncio
async def test_realtime_client_creation_with_mock():
    """Test client creation with mocked WebRTC"""
    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.is_connected = MagicMock(return_value=True)
        mock_manager.get_connection_state = MagicMock(return_value="connected")
        mock_manager_class.return_value = mock_manager

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions
        from decart.types import ModelState, Prompt

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("mirage"),
                on_remote_stream=lambda t: None,
                initial_state=ModelState(prompt=Prompt(text="Test", enrich=True), mirror=False),
            ),
        )

        assert realtime_client is not None
        assert realtime_client.session_id
        assert realtime_client.is_connected()


@pytest.mark.asyncio
async def test_realtime_set_prompt_with_mock():
    """Test set_prompt with mocked WebRTC"""
    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()
        mock_manager_class.return_value = mock_manager

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("mirage"),
                on_remote_stream=lambda t: None,
            ),
        )

        await realtime_client.set_prompt("New prompt")

        mock_manager.send_message.assert_called_once()
        call_args = mock_manager.send_message.call_args[0][0]
        assert call_args.type == "prompt"
        assert call_args.prompt == "New prompt"


@pytest.mark.asyncio
async def test_realtime_set_mirror_with_mock():
    """Test set_mirror with mocked WebRTC"""
    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()
        mock_manager_class.return_value = mock_manager

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("mirage"),
                on_remote_stream=lambda t: None,
            ),
        )

        await realtime_client.set_mirror(True)

        mock_manager.send_message.assert_called_once()
        call_args = mock_manager.send_message.call_args[0][0]
        assert call_args.type == "switch_camera"
        assert call_args.rotateY == 2


@pytest.mark.asyncio
async def test_realtime_events():
    """Test event handling"""
    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager_class.return_value = mock_manager

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("mirage"),
                on_remote_stream=lambda t: None,
            ),
        )

        connection_states = []
        errors = []

        def on_connection_change(state):
            connection_states.append(state)

        def on_error(error):
            errors.append(error)

        realtime_client.on("connection_change", on_connection_change)
        realtime_client.on("error", on_error)

        realtime_client._emit_connection_change("connected")
        assert connection_states == ["connected"]

        from decart.errors import DecartSDKError

        test_error = DecartSDKError("Test error")
        realtime_client._emit_error(test_error)
        assert len(errors) == 1
        assert errors[0].message == "Test error"
