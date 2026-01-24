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
    assert model2.fps == 25
    assert model2.width == 1280
    assert model2.height == 704
    assert model2.url_path == "/v1/stream"

    model2 = models.realtime("lucy_v2v_14b_rt")
    assert model2.name == "lucy_v2v_14b_rt"
    assert model2.fps == 20
    assert model2.width == 1280
    assert model2.height == 720
    assert model2.url_path == "/v1/stream"


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

        # Mock asyncio.wait_for to immediately raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(DecartSDKError) as exc_info:
                await realtime_client.set_prompt("New prompt")

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


# Tests for avatar-live model


def test_avatar_live_model_available():
    """Test that avatar-live model is available"""
    model = models.realtime("avatar-live")
    assert model.name == "avatar-live"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 720
    assert model.url_path == "/v1/avatar-live/stream"


@pytest.mark.asyncio
async def test_avatar_live_connect_with_avatar_image():
    """Test avatar-live connection with avatar image option"""

    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.WebRTCManager") as mock_manager_class,
        patch("decart.realtime.client.file_input_to_bytes") as mock_file_input,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.is_connected = MagicMock(return_value=True)
        mock_manager_class.return_value = mock_manager

        mock_file_input.return_value = (b"fake image data", "image/png")

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions, AvatarOptions

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("avatar-live"),
                on_remote_stream=lambda t: None,
                avatar=AvatarOptions(avatar_image=b"fake image bytes"),
            ),
        )

        assert realtime_client is not None
        assert realtime_client._is_avatar_live is True
        mock_file_input.assert_called_once()
        # Verify avatar_image_base64 was passed to connect
        mock_manager.connect.assert_called_once()
        call_kwargs = mock_manager.connect.call_args[1]
        assert "avatar_image_base64" in call_kwargs
        assert call_kwargs["avatar_image_base64"] is not None


@pytest.mark.asyncio
async def test_avatar_live_set_image():
    """Test set_image method for avatar-live"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.WebRTCManager") as mock_manager_class,
        patch("decart.realtime.client.file_input_to_bytes") as mock_file_input,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()

        image_set_event = asyncio.Event()
        image_set_result = {"success": True, "error": None}

        mock_manager.register_image_set_wait = MagicMock(
            return_value=(image_set_event, image_set_result)
        )
        mock_manager.unregister_image_set_wait = MagicMock()
        mock_manager_class.return_value = mock_manager

        mock_file_input.return_value = (b"new image data", "image/png")

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("avatar-live"),
                on_remote_stream=lambda t: None,
            ),
        )

        async def set_event():
            await asyncio.sleep(0.01)
            image_set_event.set()

        asyncio.create_task(set_event())
        await realtime_client.set_image(b"new avatar image")

        mock_manager.send_message.assert_called()
        call_args = mock_manager.send_message.call_args[0][0]
        assert call_args.type == "set_image"
        assert call_args.image_data is not None
        mock_manager.unregister_image_set_wait.assert_called_once()


@pytest.mark.asyncio
async def test_set_image_only_for_avatar_live():
    """Test that set_image raises error for non-avatar-live models"""
    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.WebRTCManager") as mock_manager_class,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager_class.return_value = mock_manager

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session_cls.return_value = mock_session

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions
        from decart.errors import InvalidInputError

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("mirage"),  # Not avatar-live
                on_remote_stream=lambda t: None,
            ),
        )

        with pytest.raises(InvalidInputError) as exc_info:
            await realtime_client.set_image(b"test image")

        assert "avatar-live" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_avatar_live_set_image_timeout():
    """Test set_image raises on timeout"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.WebRTCManager") as mock_manager_class,
        patch("decart.realtime.client.file_input_to_bytes") as mock_file_input,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()

        image_set_event = asyncio.Event()
        image_set_result = {"success": False, "error": None}

        mock_manager.register_image_set_wait = MagicMock(
            return_value=(image_set_event, image_set_result)
        )
        mock_manager.unregister_image_set_wait = MagicMock()
        mock_manager_class.return_value = mock_manager

        mock_file_input.return_value = (b"image data", "image/png")

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions
        from decart.errors import DecartSDKError

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("avatar-live"),
                on_remote_stream=lambda t: None,
            ),
        )

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(DecartSDKError) as exc_info:
                await realtime_client.set_image(b"test image")

        assert "timed out" in str(exc_info.value).lower()
        mock_manager.unregister_image_set_wait.assert_called_once()


@pytest.mark.asyncio
async def test_avatar_live_set_image_server_error():
    """Test set_image raises on server error"""
    import asyncio

    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.WebRTCManager") as mock_manager_class,
        patch("decart.realtime.client.file_input_to_bytes") as mock_file_input,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.send_message = AsyncMock()

        image_set_event = asyncio.Event()
        image_set_result = {"success": False, "error": "Invalid image format"}

        mock_manager.register_image_set_wait = MagicMock(
            return_value=(image_set_event, image_set_result)
        )
        mock_manager.unregister_image_set_wait = MagicMock()
        mock_manager_class.return_value = mock_manager

        mock_file_input.return_value = (b"image data", "image/png")

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions
        from decart.errors import DecartSDKError

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("avatar-live"),
                on_remote_stream=lambda t: None,
            ),
        )

        async def set_event():
            await asyncio.sleep(0.01)
            image_set_event.set()

        asyncio.create_task(set_event())

        with pytest.raises(DecartSDKError) as exc_info:
            await realtime_client.set_image(b"test image")

        assert "Invalid image format" in str(exc_info.value)
        mock_manager.unregister_image_set_wait.assert_called_once()


@pytest.mark.asyncio
async def test_connect_with_initial_prompt():
    """Test connection with initial_prompt option"""

    client = DecartClient(api_key="test-key")

    with (
        patch("decart.realtime.client.WebRTCManager") as mock_manager_class,
        patch("decart.realtime.client.aiohttp.ClientSession") as mock_session_cls,
    ):
        mock_manager = AsyncMock()
        mock_manager.connect = AsyncMock(return_value=True)
        mock_manager.is_connected = MagicMock(return_value=True)
        mock_manager_class.return_value = mock_manager

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        mock_session_cls.return_value = mock_session

        mock_track = MagicMock()

        from decart.realtime.types import RealtimeConnectOptions, InitialPromptOptions

        realtime_client = await RealtimeClient.connect(
            base_url=client.base_url,
            api_key=client.api_key,
            local_track=mock_track,
            options=RealtimeConnectOptions(
                model=models.realtime("mirage"),
                on_remote_stream=lambda t: None,
                initial_prompt=InitialPromptOptions(text="Test prompt", enhance=False),
            ),
        )

        assert realtime_client is not None
        mock_manager.connect.assert_called_once()
        call_kwargs = mock_manager.connect.call_args[1]
        assert "initial_prompt" in call_kwargs
        assert call_kwargs["initial_prompt"] == {"text": "Test prompt", "enhance": False}
