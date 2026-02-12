import os
from typing import Any, Optional
import aiohttp
from pydantic import ValidationError
from .errors import InvalidAPIKeyError, InvalidBaseURLError, InvalidInputError
from .models import ImageModelDefinition, _MODELS
from .process.request import send_request
from .queue.client import QueueClient
from .tokens.client import TokensClient

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
        api_key: Your Decart API key. Defaults to the DECART_API_KEY environment variable.
        base_url: API base URL (defaults to production)
        integration: Optional integration identifier (e.g., "langchain/0.1.0")

    Example:
        ```python
        # Option 1: Explicit API key
        client = DecartClient(api_key="your-key")

        # Option 2: Using DECART_API_KEY environment variable
        client = DecartClient()

        # Image generation (sync) - use process()
        image = await client.process({
            "model": models.image("lucy-pro-t2i"),
            "prompt": "A serene lake at sunset",
        })

        # Video generation (async) - use queue
        result = await client.queue.submit_and_poll({
            "model": models.video("lucy-pro-t2v"),
            "prompt": "A serene lake at sunset",
        })
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.decart.ai",
        integration: Optional[str] = None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("DECART_API_KEY", "").strip() or None

        if not resolved_api_key:
            raise InvalidAPIKeyError()

        if not base_url.startswith(("http://", "https://")):
            raise InvalidBaseURLError(base_url)

        self.api_key = resolved_api_key
        self.base_url = base_url
        self.integration = integration
        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: Optional[QueueClient] = None
        self._tokens: Optional[TokensClient] = None

    @property
    def queue(self) -> QueueClient:
        """
        Queue client for async job-based video generation.
        Only video models support the queue API.

        Example:
            ```python
            # Submit and poll automatically
            result = await client.queue.submit_and_poll({
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A cat playing piano",
            })

            # Or submit and poll manually
            job = await client.queue.submit({...})
            status = await client.queue.status(job.job_id)
            data = await client.queue.result(job.job_id)
            ```
        """
        if self._queue is None:
            self._queue = QueueClient(self)
        return self._queue

    @property
    def tokens(self) -> TokensClient:
        """
        Client for creating client tokens.
        Client tokens are short-lived API keys safe for client-side use.

        Example:
            ```python
            client = DecartClient(api_key=os.getenv("DECART_API_KEY"))
            token = await client.tokens.create()
            # Returns: CreateTokenResponse(api_key="ek_...", expires_at="...")
            ```
        """
        if self._tokens is None:
            self._tokens = TokensClient(self)
        return self._tokens

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
        Process image generation/transformation synchronously.
        Only image models support the process API.

        For video generation, use the queue API instead:
            result = await client.queue.submit_and_poll({...})

        Args:
            options: Processing options including model and inputs
                - model: ImageModelDefinition from models.image()
                - prompt: Text prompt for generation
                - Additional model-specific inputs

        Returns:
            Generated/transformed image as bytes

        Raises:
            InvalidInputError: If inputs are invalid or model is not an image model
            ProcessingError: If processing fails
        """
        if "model" not in options:
            raise InvalidInputError("model is required")

        model: ImageModelDefinition = options["model"]

        # Validate that this is an image model (check against registry)
        if model.name not in _MODELS["image"]:
            raise InvalidInputError(
                f"Model '{model.name}' is not supported by process(). "
                f"Only image models support sync processing. "
                f"For video models, use client.queue.submit_and_poll() instead."
            )

        cancel_token = options.get("cancel_token")

        inputs = {k: v for k, v in options.items() if k not in ("model", "cancel_token")}

        # File fields that need special handling (not validated by Pydantic)
        FILE_FIELDS = {"data", "start", "end", "reference_image"}

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
            integration=self.integration,
        )

        return response
