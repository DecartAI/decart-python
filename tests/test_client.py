import pytest
from decart import DecartClient, InvalidAPIKeyError, InvalidBaseURLError


class TestDecartClient:
    """Tests for DecartClient initialization."""

    def test_create_client_with_explicit_api_key(self) -> None:
        """Creates a client with explicit api_key."""
        client = DecartClient(api_key="test-key")
        assert client is not None
        assert client.process is not None
        assert client.api_key == "test-key"

    def test_create_client_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creates a client using DECART_API_KEY env var."""
        monkeypatch.setenv("DECART_API_KEY", "env-api-key")
        client = DecartClient()
        assert client is not None
        assert client.api_key == "env-api-key"

    def test_explicit_api_key_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit api_key takes precedence over env var."""
        monkeypatch.setenv("DECART_API_KEY", "env-api-key")
        client = DecartClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_create_client_no_api_key_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Throws an error if api key is not provided and env var is not set."""
        monkeypatch.delenv("DECART_API_KEY", raising=False)
        with pytest.raises(InvalidAPIKeyError, match="Missing API key"):
            DecartClient()

    def test_create_client_empty_api_key(self) -> None:
        """Throws an error if api key is empty string."""
        with pytest.raises(InvalidAPIKeyError, match="Missing API key"):
            DecartClient(api_key="")

    def test_create_client_empty_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Throws an error if env var is empty string."""
        monkeypatch.setenv("DECART_API_KEY", "")
        with pytest.raises(InvalidAPIKeyError, match="Missing API key"):
            DecartClient()

    def test_create_client_whitespace_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Throws an error if env var is only whitespace."""
        monkeypatch.setenv("DECART_API_KEY", "   ")
        with pytest.raises(InvalidAPIKeyError, match="Missing API key"):
            DecartClient()

    def test_create_client_invalid_base_url(self) -> None:
        """Throws an error if invalid base url is provided."""
        with pytest.raises(InvalidBaseURLError):
            DecartClient(api_key="test-key", base_url="invalid-url")

    def test_create_client_custom_base_url(self) -> None:
        """Creates a client with custom base url."""
        client = DecartClient(api_key="test-key", base_url="https://custom.decart.ai")
        assert client is not None
        assert client.base_url == "https://custom.decart.ai"
