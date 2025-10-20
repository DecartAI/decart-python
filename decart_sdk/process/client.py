from typing import Any
from pydantic import ValidationError
from ..models import ModelDefinition
from ..errors import create_invalid_input_error
from .request import send_request


class ProcessClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def process(self, options: dict[str, Any]) -> bytes:
        if "model" not in options:
            raise create_invalid_input_error("model is required")

        model: ModelDefinition = options["model"]
        cancel_token = options.get("cancel_token")

        inputs = {k: v for k, v in options.items() if k not in ("model", "cancel_token")}

        file_fields = {"data", "start", "end"}
        file_inputs = {k: v for k, v in inputs.items() if k in file_fields}
        non_file_inputs = {k: v for k, v in inputs.items() if k not in file_fields}

        validation_inputs = non_file_inputs.copy()
        for field in file_fields:
            if field in file_inputs:
                validation_inputs[field] = b""

        try:
            validated_inputs = model.input_schema(**validation_inputs)
        except ValidationError as e:
            raise create_invalid_input_error(
                f"Invalid inputs for {model.name}: {str(e)}"
            ) from e

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
