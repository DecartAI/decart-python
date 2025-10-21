import aiohttp
import asyncio
from typing import Any, Optional
from ..types import FileInput
from ..models import ModelDefinition
from ..errors import InvalidInputError, ProcessingError


async def file_input_to_bytes(input_data: FileInput) -> tuple[bytes, str]:
    if isinstance(input_data, bytes):
        return input_data, "application/octet-stream"

    if hasattr(input_data, "read"):
        content = await asyncio.to_thread(input_data.read)
        if isinstance(content, str):
            content = content.encode()
        return content, "application/octet-stream"

    if isinstance(input_data, str):
        if not input_data.startswith(("http://", "https://")):
            raise InvalidInputError("URL must start with http:// or https://")

        async with aiohttp.ClientSession() as session:
            async with session.get(input_data) as response:
                if not response.ok:
                    raise InvalidInputError(
                        f"Failed to fetch file from URL: {response.status}"
                    )
                content = await response.read()
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                return content, content_type

    raise InvalidInputError("Invalid file input type")


async def send_request(
    base_url: str,
    api_key: str,
    model: ModelDefinition,
    inputs: dict[str, Any],
    cancel_token: Optional[asyncio.Event] = None,
) -> bytes:
    form_data = aiohttp.FormData()

    for key, value in inputs.items():
        if value is not None:
            if key in ("data", "start", "end"):
                content, content_type = await file_input_to_bytes(value)
                form_data.add_field(key, content, content_type=content_type)
            else:
                form_data.add_field(key, str(value))

    endpoint = f"{base_url}{model.url_path}"

    timeout = aiohttp.ClientTimeout(total=300)

    async def make_request() -> bytes:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                endpoint,
                headers={"X-API-KEY": api_key},
                data=form_data,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ProcessingError(
                        f"Processing failed: {response.status} - {error_text}"
                    )
                return await response.read()

    if cancel_token:
        request_task = asyncio.create_task(make_request())
        cancel_task = asyncio.create_task(cancel_token.wait())

        done, pending = await asyncio.wait(
            [request_task, cancel_task], return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        if cancel_task in done:
            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass
            raise asyncio.CancelledError("Request cancelled by user")

        return await request_task

    return await make_request()
