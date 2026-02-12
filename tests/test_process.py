"""
Tests for the process API.
Note: process() only supports image models (t2i, i2i).
Video models must use the queue API.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from decart import DecartClient, models, DecartSDKError


@pytest.mark.asyncio
async def test_process_text_to_image() -> None:
    """Test text-to-image generation with process API."""
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake image data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await client.process(
            {
                "model": models.image("lucy-pro-t2i"),
                "prompt": "A cat walking",
            }
        )

        assert result == b"fake image data"


@pytest.mark.asyncio
async def test_process_image_to_image() -> None:
    """Test image-to-image transformation with process API."""
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake image data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await client.process(
            {
                "model": models.image("lucy-pro-i2i"),
                "prompt": "Oil painting style",
                "data": b"fake input image",
                "enhance_prompt": True,
            }
        )

        assert result == b"fake image data"


@pytest.mark.asyncio
async def test_process_rejects_video_models() -> None:
    """Test that process() rejects video models with helpful error message."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError) as exc_info:
        await client.process(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A cat walking",
            }
        )

    assert "not supported by process()" in str(exc_info.value)
    assert "queue" in str(exc_info.value).lower()


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
    """Test that missing required fields raise an error."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.process(
            {
                "model": models.image("lucy-pro-i2i"),
                # Missing 'data' field which is required for i2i
            }
        )


@pytest.mark.asyncio
async def test_process_max_prompt_length() -> None:
    client = DecartClient(api_key="test-key")
    prompt = "a" * 1001
    with pytest.raises(DecartSDKError) as exception:
        await client.process(
            {
                "model": models.image("lucy-pro-t2i"),
                "prompt": prompt,
            }
        )
    assert "Invalid inputs for lucy-pro-t2i: 1 validation error for TextToImageInput" in str(
        exception
    )


@pytest.mark.asyncio
async def test_process_with_cancellation() -> None:
    """Test that process() respects cancellation token."""
    client = DecartClient(api_key="test-key")
    cancel_token = asyncio.Event()

    cancel_token.set()

    with pytest.raises(asyncio.CancelledError):
        await client.process(
            {
                "model": models.image("lucy-pro-t2i"),
                "prompt": "An image that will be cancelled",
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
        mock_response.read = AsyncMock(return_value=b"fake image data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        await client.process(
            {
                "model": models.image("lucy-pro-t2i"),
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
        mock_response.read = AsyncMock(return_value=b"fake image data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        await client.process(
            {
                "model": models.image("lucy-pro-t2i"),
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


@pytest.mark.asyncio
async def test_process_rejects_file_exceeding_20mb() -> None:
    """Test that files exceeding 20MB are rejected before upload."""
    client = DecartClient(api_key="test-key")

    large_data = b"x" * (21 * 1024 * 1024)

    with pytest.raises(DecartSDKError, match="exceeds the maximum allowed size of 20MB"):
        await client.process(
            {
                "model": models.image("lucy-pro-i2i"),
                "prompt": "test",
                "data": large_data,
            }
        )


@pytest.mark.asyncio
async def test_process_accepts_file_at_20mb_limit() -> None:
    """Test that files at exactly 20MB are accepted."""
    client = DecartClient(api_key="test-key")

    exact_data = b"x" * (20 * 1024 * 1024)

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.read = AsyncMock(return_value=b"fake image data")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        result = await client.process(
            {
                "model": models.image("lucy-pro-i2i"),
                "prompt": "test",
                "data": exact_data,
            }
        )

        assert result == b"fake image data"
