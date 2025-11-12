import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from decart import DecartClient, models, DecartSDKError
from decart.types import MotionTrajectoryInput


@pytest.mark.asyncio
async def test_process_text_to_video() -> None:
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake video data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await client.process(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A cat walking",
            }
        )

        assert result == b"fake video data"


@pytest.mark.asyncio
async def test_process_missing_model() -> None:
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.process(
            {
                "prompt": "A cat walking",
            }
        )


@pytest.mark.asyncio
async def test_process_missing_required_field() -> None:
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.process(
            {
                "model": models.video("lucy-pro-i2v"),
            }
        )


@pytest.mark.asyncio
async def test_process_video_to_video() -> None:
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake video data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await client.process(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "Anime style",
                "data": b"fake input video",
                "enhance_prompt": True,
            }
        )

        assert result == b"fake video data"


@pytest.mark.asyncio
async def test_process_image_to_motion_video() -> None:
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake video data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await client.process(
            {
                "model": models.video("lucy-motion"),
                "data": b"fake input image",
                "trajectory": [
                    MotionTrajectoryInput(frame=0, x=0, y=0),
                    MotionTrajectoryInput(frame=1, x=0.5, y=0.5),
                    MotionTrajectoryInput(frame=2, x=1, y=1),
                    MotionTrajectoryInput(frame=3, x=1.5, y=1.5),
                    MotionTrajectoryInput(frame=4, x=2, y=2),
                ],
            }
        )

        assert result == b"fake video data"


@pytest.mark.asyncio
async def test_process_image_to_motion_video_invalid_trajectory() -> None:
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError) as exception:
        await client.process(
            {
                "model": models.video("lucy-motion"),
                "data": b"fake input image",
                "trajectory": [
                    MotionTrajectoryInput(frame=0, x=0, y=0),
                ],
            }
        )
    assert "Invalid inputs for lucy-motion: 1 validation error for ImageToMotionVideoInput" in str(
        exception
    )


@pytest.mark.asyncio
async def test_process_with_cancellation() -> None:
    client = DecartClient(api_key="test-key")
    cancel_token = asyncio.Event()

    cancel_token.set()

    with pytest.raises(asyncio.CancelledError):
        await client.process(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A video that will be cancelled",
                "cancel_token": cancel_token,
            }
        )


@pytest.mark.asyncio
async def test_process_includes_user_agent_header() -> None:
    """Test that User-Agent header is included in requests."""
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake video data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        await client.process(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "Test prompt",
            }
        )

        # Verify post was called with User-Agent header
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args[1]
        headers = call_kwargs.get("headers", {})

        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("decart-python-sdk/")
        assert "lang/py" in headers["User-Agent"]


@pytest.mark.asyncio
async def test_process_includes_integration_in_user_agent() -> None:
    """Test that integration parameter is included in User-Agent header."""
    client = DecartClient(api_key="test-key", integration="langchain/0.1.0")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake video data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        await client.process(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "Test prompt",
            }
        )

        # Verify post was called with User-Agent header including integration
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args[1]
        headers = call_kwargs.get("headers", {})

        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("decart-python-sdk/")
        assert "lang/py" in headers["User-Agent"]
        assert "langchain/0.1.0" in headers["User-Agent"]
        assert headers["User-Agent"].endswith(" langchain/0.1.0")
