import pytest
from decart import DecartClient, InvalidAPIKeyError, InvalidBaseURLError


def test_create_client_success() -> None:
    client = DecartClient(api_key="test-key")
    assert client is not None
    assert client.process is not None


def test_create_client_invalid_api_key() -> None:
    with pytest.raises(InvalidAPIKeyError):
        DecartClient(api_key="")


def test_create_client_invalid_base_url() -> None:
    with pytest.raises(InvalidBaseURLError):
        DecartClient(api_key="test-key", base_url="invalid-url")


def test_create_client_custom_base_url() -> None:
    client = DecartClient(api_key="test-key", base_url="https://custom.decart.ai")
    assert client is not None
    assert client.base_url == "https://custom.decart.ai"
