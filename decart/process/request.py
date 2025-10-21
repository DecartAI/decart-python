import aiohttp
import aiofiles
import asyncio
from pathlib import Path
from typing import Any, Optional
from ..types import FileInput
from ..models import ModelDefinition
from ..errors import InvalidInputError, ProcessingError


async def file_input_to_bytes(
    input_data: FileInput, session: aiohttp.ClientSession
) -> tuple[bytes, str]:
    """Convert various file input types to bytes asynchronously.

    Args:
        input_data: The file input (bytes, Path, str, or file-like object)
        session: Reusable aiohttp session for URL fetching

    Returns:
        Tuple of (content bytes, content type)

    Raises:
        InvalidInputError: If input is invalid or processing fails
    """

    if isinstance(input_data, bytes):
        return input_data, "application/octet-stream"

    if isinstance(input_data, Path):
        # Async file reading with aiofiles
        try:
            async with aiofiles.open(input_data, mode="rb") as f:
                content = await f.read()
            return content, "application/octet-stream"
        except FileNotFoundError:
            raise InvalidInputError(f"File not found: {input_data}")
        except Exception as e:
            raise InvalidInputError(f"Failed to read file {input_data}: {str(e)}")

    if isinstance(input_data, str):
        # Check if it's a file path
        path = Path(input_data)
        if path.exists():
            try:
                async with aiofiles.open(path, mode="rb") as f:
                    content = await f.read()
                return content, "application/octet-stream"
            except Exception as e:
                raise InvalidInputError(f"Failed to read file {input_data}: {str(e)}")

        # Otherwise treat as URL
        if not input_data.startswith(("http://", "https://")):
            raise InvalidInputError(
                f"Input must be a URL (http:// or https://) or existing file path: {input_data}"
            )

        # Use the provided session instead of creating a new one
        async with session.get(input_data) as response:
            if not response.ok:
                raise InvalidInputError(f"Failed to fetch file from URL: {response.status}")
            content = await response.read()
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            return content, content_type

    from ..types import HasRead

    if isinstance(input_data, HasRead):
        # Sync file-like objects (for backwards compatibility)
        content = await asyncio.to_thread(input_data.read)
        if isinstance(content, str):
            content = content.encode()
        return content, "application/octet-stream"

    raise InvalidInputError(f"Invalid file input type: {type(input_data)}")


async def send_request(
    session: aiohttp.ClientSession,
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
                content, content_type = await file_input_to_bytes(value, session)
                form_data.add_field(key, content, content_type=content_type)
            else:
                form_data.add_field(key, str(value))

    endpoint = f"{base_url}{model.url_path}"

    async def make_request() -> bytes:
        async with session.post(
            endpoint,
            headers={"X-API-KEY": api_key},
            data=form_data,
        ) as response:
            if not response.ok:
                error_text = await response.text()
                raise ProcessingError(f"Processing failed: {response.status} - {error_text}")
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
            try:
                await request_task
            except asyncio.CancelledError:
                pass
            raise asyncio.CancelledError("Request cancelled by user")

        return await request_task

    return await make_request()
