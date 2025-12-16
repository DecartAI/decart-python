"""Tests for the tokens API."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from decart import DecartClient, TokenCreateError


@pytest.mark.asyncio
async def test_create_token() -> None:
    """Creates a client token successfully."""
    client = DecartClient(api_key="test-api-key")

    mock_response = AsyncMock()
    mock_response.ok = True
    mock_response.json = AsyncMock(
        return_value={"apiKey": "ek_test123", "expiresAt": "2024-12-15T12:10:00Z"}
    )

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch.object(client, "_get_session", AsyncMock(return_value=mock_session)):
        result = await client.tokens.create()

    assert result.api_key == "ek_test123"
    assert result.expires_at == "2024-12-15T12:10:00Z"


@pytest.mark.asyncio
async def test_create_token_401_error() -> None:
    """Handles 401 error."""
    client = DecartClient(api_key="test-api-key")

    mock_response = AsyncMock()
    mock_response.ok = False
    mock_response.status = 401
    mock_response.text = AsyncMock(return_value="Invalid API key")

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch.object(client, "_get_session", AsyncMock(return_value=mock_session)):
        with pytest.raises(TokenCreateError, match="Failed to create token"):
            await client.tokens.create()


@pytest.mark.asyncio
async def test_create_token_403_error() -> None:
    """Handles 403 error."""
    client = DecartClient(api_key="test-api-key")

    mock_response = AsyncMock()
    mock_response.ok = False
    mock_response.status = 403
    mock_response.text = AsyncMock(return_value="Cannot create token from client token")

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch.object(client, "_get_session", AsyncMock(return_value=mock_session)):
        with pytest.raises(TokenCreateError, match="Failed to create token"):
            await client.tokens.create()
