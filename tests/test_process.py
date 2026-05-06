"""
Tests for the process API.
Note: process() accepts any model definition and lets the backend validate support.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from decart import DecartClient, ModelDefinition, models, DecartSDKError


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
                "model": models.image("lucy-image-2"),
                "prompt": "Apply an oil-painting treatment while preserving the composition",
                "data": b"fake input image",
                "enhance_prompt": True,
            }
        )

        assert result == b"fake image data"


@pytest.mark.asyncio
async def test_process_image_to_image_with_reference_image() -> None:
    """Test image-to-image with optional reference_image."""
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
                "model": models.image("lucy-image-2"),
                "prompt": "Add the object from the reference image",
                "data": b"fake input image",
                "reference_image": b"fake reference image",
                "enhance_prompt": False,
            }
        )

        assert result == b"fake image data"


@pytest.mark.asyncio
async def test_process_accepts_custom_model_definition_without_schema() -> None:
    client = DecartClient(api_key="test-key")
    custom_model = ModelDefinition(
        name="lucy_image_preview",
        url_path="/v1/generate/lucy_image_preview",
        fps=25,
        width=1280,
        height=704,
    )

    with patch("decart.client.send_request", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = b"fake image data"

        result = await client.process(
            {
                "model": custom_model,
                "prompt": "Apply a preview model treatment",
                "data": b"fake image data",
                "custom_strength": 0.7,
                "optional": None,
            }
        )

    assert result == b"fake image data"
    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["model"] is custom_model
    assert call_kwargs["inputs"] == {
        "prompt": "Apply a preview model treatment",
        "data": b"fake image data",
        "custom_strength": 0.7,
    }


@pytest.mark.asyncio
async def test_process_allows_custom_model_definition_for_realtime_url_path() -> None:
    client = DecartClient(api_key="test-key")
    custom_model = ModelDefinition(
        name="lucy_image_preview",
        url_path="/v1/stream",
        fps=25,
        width=1280,
        height=704,
    )

    with patch("decart.client.send_request", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = b"fake image data"

        result = await client.process(
            {
                "model": custom_model,
                "prompt": "Apply a preview model treatment",
                "data": b"fake image data",
            }
        )

    assert result == b"fake image data"
    assert mock_send.call_args.kwargs["model"] is custom_model


@pytest.mark.asyncio
async def test_process_missing_model() -> None:
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.process(
            {
                "prompt": "Apply an editorial color grade",
            }
        )


@pytest.mark.asyncio
async def test_process_missing_required_field() -> None:
    """Test that missing required fields raise an error."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.process(
            {
                "model": models.image("lucy-image-2"),
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
                "model": models.image("lucy-image-2"),
                "prompt": prompt,
                "data": b"fake image data",
            }
        )
    assert "Invalid inputs for lucy-image-2: 1 validation error for ImageToImageInput" in str(
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
                "model": models.image("lucy-image-2"),
                "prompt": "Apply a high-contrast editorial treatment",
                "data": b"fake image data",
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
                "model": models.image("lucy-image-2"),
                "prompt": "Apply a soft watercolor treatment",
                "data": b"fake image data",
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
                "model": models.image("lucy-image-2"),
                "prompt": "Apply a soft watercolor treatment",
                "data": b"fake image data",
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
