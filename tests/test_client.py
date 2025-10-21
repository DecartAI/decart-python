import pytest
from decart_sdk import create_decart_client, DecartSDKError, ErrorCodes


def test_create_client_success() -> None:
    client = create_decart_client(api_key="test-key")
    assert client is not None
    assert client.process is not None


def test_create_client_invalid_api_key() -> None:
    with pytest.raises(DecartSDKError) as exc_info:
        create_decart_client(api_key="")

    assert exc_info.value.code == ErrorCodes.INVALID_API_KEY


def test_create_client_invalid_base_url() -> None:
    with pytest.raises(DecartSDKError) as exc_info:
        create_decart_client(api_key="test-key", base_url="invalid-url")

    assert exc_info.value.code == ErrorCodes.INVALID_BASE_URL


def test_create_client_custom_base_url() -> None:
    client = create_decart_client(api_key="test-key", base_url="https://custom.decart.ai")
    assert client is not None
    assert client._process_client.base_url == "https://custom.decart.ai"
