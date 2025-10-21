from typing import Any
from pydantic import BaseModel, Field, field_validator
from .errors import create_invalid_api_key_error, create_invalid_base_url_error
from .process.client import ProcessClient

try:
    from .realtime.factory import RealtimeClientFactory

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeClientFactory = None  # type: ignore


class DecartConfiguration(BaseModel):
    api_key: str = Field(..., min_length=1)
    base_url: str = Field(default="https://api.decart.ai")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v


class DecartClient:
    def __init__(self, configuration: DecartConfiguration) -> None:
        self._process_client = ProcessClient(
            api_key=configuration.api_key,
            base_url=configuration.base_url,
        )

        if REALTIME_AVAILABLE:
            self.realtime = RealtimeClientFactory(
                base_url=configuration.base_url, api_key=configuration.api_key
            )
        else:
            self.realtime = None  # type: ignore

    async def process(self, options: dict[str, Any]) -> bytes:
        return await self._process_client.process(options)


def create_decart_client(
    api_key: str,
    base_url: str = "https://api.decart.ai",
) -> DecartClient:
    try:
        config = DecartConfiguration(api_key=api_key, base_url=base_url)
    except ValueError as e:
        if "api_key" in str(e):
            raise create_invalid_api_key_error() from e
        if "base_url" in str(e):
            raise create_invalid_base_url_error(base_url) from e
        raise

    return DecartClient(configuration=config)
