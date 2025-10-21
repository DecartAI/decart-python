from typing import Any, Optional
import aiohttp
from pydantic import ValidationError
from .errors import InvalidAPIKeyError, InvalidBaseURLError, InvalidInputError
from .models import ModelDefinition
from .process.request import send_request

try:
    from .realtime.client import RealtimeClient

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeClient = None  # type: ignore


class DecartClient:
    """
    Decart API client for video and image generation/transformation.

    Args:
        api_key: Your Decart API key
        base_url: API base URL (defaults to production)

    Example:
        ```python
        client = DecartClient(api_key="your-key")
        result = await client.process({
            "model": models.video("lucy-pro-t2v"),
            "prompt": "A serene lake at sunset",
        })
        ```
    """

    def __init__(self, api_key: str, base_url: str = "https://api.decart.ai") -> None:
        if not api_key or not api_key.strip():
            raise InvalidAPIKeyError()

        if not base_url.startswith(("http://", "https://")):
            raise InvalidBaseURLError(base_url)

        self.api_key = api_key
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def process(self, options: dict[str, Any]) -> bytes:
        """
        Process video or image generation/transformation.

        Args:
            options: Processing options including model and inputs

        Returns:
            Generated/transformed media as bytes

        Raises:
            InvalidInputError: If inputs are invalid
            ProcessingError: If processing fails
        """
        if "model" not in options:
            raise InvalidInputError("model is required")

        model: ModelDefinition = options["model"]
        cancel_token = options.get("cancel_token")

        inputs = {k: v for k, v in options.items() if k not in ("model", "cancel_token")}

        # File fields that need special handling (not validated by Pydantic)
        FILE_FIELDS = {"data", "start", "end"}

        # Separate file inputs from regular inputs
        file_inputs = {k: v for k, v in inputs.items() if k in FILE_FIELDS}
        non_file_inputs = {k: v for k, v in inputs.items() if k not in FILE_FIELDS}

        # Validate non-file inputs and create placeholder for file fields
        validation_inputs = {
            **non_file_inputs,
            **{k: b"" for k in file_inputs.keys()},  # Placeholder bytes for validation
        }

        try:
            validated_inputs = model.input_schema(**validation_inputs)
        except ValidationError as e:
            raise InvalidInputError(f"Invalid inputs for {model.name}: {str(e)}") from e

        # Build final inputs: validated non-file inputs + original file inputs
        processed_inputs = {
            **validated_inputs.model_dump(exclude_none=True),
            **file_inputs,  # Override placeholders with actual file data
        }

        session = await self._get_session()
        response = await send_request(
            session=session,
            base_url=self.base_url,
            api_key=self.api_key,
            model=model,
            inputs=processed_inputs,
            cancel_token=cancel_token,
        )

        return response
