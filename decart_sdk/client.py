from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError
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

        # Separate file inputs from other inputs
        file_fields = {"data", "start", "end"}
        file_inputs = {k: v for k, v in inputs.items() if k in file_fields}
        non_file_inputs = {k: v for k, v in inputs.items() if k not in file_fields}

        # Validate inputs using model's schema
        validation_inputs = non_file_inputs.copy()
        for field in file_fields:
            if field in file_inputs:
                validation_inputs[field] = b""  # Placeholder for validation

        try:
            validated_inputs = model.input_schema(**validation_inputs)
        except ValidationError as e:
            raise InvalidInputError(f"Invalid inputs for {model.name}: {str(e)}") from e

        # Merge validated inputs with file inputs
        processed_inputs = validated_inputs.model_dump(exclude_none=True)
        for field in file_fields:
            if field in file_inputs:
                processed_inputs[field] = file_inputs[field]

        response = await send_request(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model,
            inputs=processed_inputs,
            cancel_token=cancel_token,
        )

        return response
