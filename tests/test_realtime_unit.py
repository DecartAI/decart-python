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
    assert model.url_path == "/v1/stream"

    model2 = models.realtime("mirage_v2")
    assert model2.name == "mirage_v2"
    assert model2.fps == 22
    assert model2.width == 1280
    assert model2.height == 704
    assert model2.url_path == "/v1/stream"

    model2 = models.realtime("lucy_v2v_720p_rt")
    assert model2.name == "lucy_v2v_720p_rt"


@pytest.mark.asyncio
async def test_realtime_client_creation_with_mock():
    """Test client creation with mocked WebRTC"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.is_connected = MagicMock(return_value=True)
        mock_manager.get_connection_state = MagicMock(return_value="connected")
        mock_manager.send_message = AsyncMock()

        prompt_event = asyncio.Event()
        prompt_result = {"success": True, "error": None}
        prompt_event.set()

        mock_manager.register_prompt_wait = MagicMock(return_value=(prompt_event, prompt_result))
        mock_manager.unregister_prompt_wait = MagicMock()
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
                initial_state=ModelState(prompt=Prompt(text="Test", enrich=True)),
            ),
        )

        assert realtime_client is not None
        assert realtime_client.session_id
        assert realtime_client.is_connected()


@pytest.mark.asyncio
async def test_realtime_set_prompt_with_mock():
    """Test set_prompt with mocked WebRTC and prompt_ack"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()

        prompt_event = asyncio.Event()
        prompt_result = {"success": True, "error": None}

        def register_prompt_wait(prompt):
            return prompt_event, prompt_result

        mock_manager.register_prompt_wait = MagicMock(side_effect=register_prompt_wait)
        mock_manager.unregister_prompt_wait = MagicMock()
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

        async def set_event():
            await asyncio.sleep(0.01)
            prompt_event.set()

        asyncio.create_task(set_event())
        await realtime_client.set_prompt("New prompt")

        mock_manager.send_message.assert_called()
        call_args = mock_manager.send_message.call_args[0][0]
        assert call_args.type == "prompt"
        assert call_args.prompt == "New prompt"
        assert call_args.enhance_prompt is True
        mock_manager.unregister_prompt_wait.assert_called_with("New prompt")


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


@pytest.mark.asyncio
async def test_realtime_set_prompt_timeout():
    """Test set_prompt raises on timeout"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()

        prompt_event = asyncio.Event()
        prompt_result = {"success": False, "error": None}

        def register_prompt_wait(prompt):
            return prompt_event, prompt_result

        mock_manager.register_prompt_wait = MagicMock(side_effect=register_prompt_wait)
        mock_manager.unregister_prompt_wait = MagicMock()
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

        from decart.errors import DecartSDKError

        with pytest.raises(DecartSDKError) as exc_info:
            await realtime_client.set_prompt("New prompt", max_timeout=0.01)

        assert "timed out" in str(exc_info.value)
        mock_manager.unregister_prompt_wait.assert_called_with("New prompt")


@pytest.mark.asyncio
async def test_realtime_set_prompt_server_error():
    """Test set_prompt raises on server error"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with patch("decart.realtime.client.WebRTCManager") as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()

        prompt_event = asyncio.Event()
        prompt_result = {"success": False, "error": "Server rejected prompt"}

        def register_prompt_wait(prompt):
            return prompt_event, prompt_result

        mock_manager.register_prompt_wait = MagicMock(side_effect=register_prompt_wait)
        mock_manager.unregister_prompt_wait = MagicMock()
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

        async def set_event():
            await asyncio.sleep(0.01)
            prompt_event.set()

        asyncio.create_task(set_event())

        from decart.errors import DecartSDKError

        with pytest.raises(DecartSDKError) as exc_info:
            await realtime_client.set_prompt("New prompt")

        assert "Server rejected prompt" in str(exc_info.value)
        mock_manager.unregister_prompt_wait.assert_called_with("New prompt")
