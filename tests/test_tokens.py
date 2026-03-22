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
    assert result.permissions is None
    assert result.constraints is None


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


@pytest.mark.asyncio
async def test_create_token_with_metadata() -> None:
    """Sends metadata as JSON body when provided."""
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
        result = await client.tokens.create(metadata={"role": "viewer"})

    assert result.api_key == "ek_test123"
    assert result.expires_at == "2024-12-15T12:10:00Z"
    call_kwargs = mock_session.post.call_args
    assert call_kwargs.kwargs["json"] == {"metadata": {"role": "viewer"}}


@pytest.mark.asyncio
async def test_create_token_without_metadata_sends_null() -> None:
    """Sends JSON body with null metadata when none provided."""
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
        await client.tokens.create()

    call_kwargs = mock_session.post.call_args
    assert call_kwargs.kwargs["json"] == {}


@pytest.mark.asyncio
async def test_create_token_with_expires_in() -> None:
    """Sends expiresIn in request body."""
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
        await client.tokens.create(expires_in=120)

    call_kwargs = mock_session.post.call_args
    assert call_kwargs.kwargs["json"] == {"expiresIn": 120}


@pytest.mark.asyncio
async def test_create_token_with_allowed_models() -> None:
    """Sends allowedModels in request body."""
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
        await client.tokens.create(allowed_models=["lucy_2_rt"])

    call_kwargs = mock_session.post.call_args
    assert call_kwargs.kwargs["json"] == {"allowedModels": ["lucy_2_rt"]}


@pytest.mark.asyncio
async def test_create_token_with_constraints() -> None:
    """Sends constraints in request body."""
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

    constraints = {"realtime": {"maxSessionDuration": 120}}
    with patch.object(client, "_get_session", AsyncMock(return_value=mock_session)):
        await client.tokens.create(constraints=constraints)

    call_kwargs = mock_session.post.call_args
    assert call_kwargs.kwargs["json"] == {
        "constraints": {"realtime": {"maxSessionDuration": 120}}
    }


@pytest.mark.asyncio
async def test_create_token_with_all_v2_fields() -> None:
    """Sends all v2 fields and parses permissions/constraints from response."""
    client = DecartClient(api_key="test-api-key")

    mock_response = AsyncMock()
    mock_response.ok = True
    mock_response.json = AsyncMock(
        return_value={
            "apiKey": "ek_test123",
            "expiresAt": "2024-12-15T12:10:00Z",
            "permissions": {"models": ["lucy_2_rt"]},
            "constraints": {"realtime": {"maxSessionDuration": 120}},
        }
    )

    mock_session = MagicMock()
    mock_session.post = MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
    )

    with patch.object(client, "_get_session", AsyncMock(return_value=mock_session)):
        result = await client.tokens.create(
            metadata={"role": "viewer"},
            expires_in=120,
            allowed_models=["lucy_2_rt"],
            constraints={"realtime": {"maxSessionDuration": 120}},
        )

    assert result.api_key == "ek_test123"
    assert result.expires_at == "2024-12-15T12:10:00Z"
    assert result.permissions == {"models": ["lucy_2_rt"]}
    assert result.constraints == {"realtime": {"maxSessionDuration": 120}}

    call_kwargs = mock_session.post.call_args
    assert call_kwargs.kwargs["json"] == {
        "metadata": {"role": "viewer"},
        "expiresIn": 120,
        "allowedModels": ["lucy_2_rt"],
        "constraints": {"realtime": {"maxSessionDuration": 120}},
    }
