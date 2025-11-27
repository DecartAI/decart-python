import aiohttp
from typing import Any, Optional

from ..models import ModelDefinition
from ..errors import QueueSubmitError, QueueStatusError, QueueResultError
from .._user_agent import build_user_agent
from ..process.request import file_input_to_bytes
from .types import JobSubmitResponse, JobStatusResponse


async def submit_job(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: ModelDefinition,
    inputs: dict[str, Any],
    integration: Optional[str] = None,
) -> JobSubmitResponse:
    """Submit a job to the queue.

    POST /v1/jobs/{model}
    """
    form_data = aiohttp.FormData()

    for key, value in inputs.items():
        if value is not None:
            if key in ("data", "start", "end"):
                content, content_type = await file_input_to_bytes(value, session)
                form_data.add_field(key, content, content_type=content_type)
            else:
                form_data.add_field(key, str(value))

    endpoint = f"{base_url}/v1/jobs/{model.name}"

    async with session.post(
        endpoint,
        headers={
            "X-API-KEY": api_key,
            "User-Agent": build_user_agent(integration),
        },
        data=form_data,
    ) as response:
        if not response.ok:
            error_text = await response.text()
            raise QueueSubmitError(
                f"Failed to submit job: {response.status} - {error_text}",
                data={"status": response.status},
            )
        data = await response.json()
        return JobSubmitResponse(**data)


async def get_job_status(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    job_id: str,
    integration: Optional[str] = None,
) -> JobStatusResponse:
    """Get the status of a job.

    GET /v1/jobs/{job_id}
    """
    endpoint = f"{base_url}/v1/jobs/{job_id}"

    async with session.get(
        endpoint,
        headers={
            "X-API-KEY": api_key,
            "User-Agent": build_user_agent(integration),
        },
    ) as response:
        if not response.ok:
            error_text = await response.text()
            raise QueueStatusError(
                f"Failed to get job status: {response.status} - {error_text}",
                data={"status": response.status},
            )
        data = await response.json()
        return JobStatusResponse(**data)


async def get_job_content(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    job_id: str,
    integration: Optional[str] = None,
) -> bytes:
    """Get the content/result of a completed job.

    GET /v1/jobs/{job_id}/content
    """
    endpoint = f"{base_url}/v1/jobs/{job_id}/content"

    async with session.get(
        endpoint,
        headers={
            "X-API-KEY": api_key,
            "User-Agent": build_user_agent(integration),
        },
    ) as response:
        if not response.ok:
            error_text = await response.text()
            raise QueueResultError(
                f"Failed to get job content: {response.status} - {error_text}",
                data={"status": response.status},
            )
        return await response.read()
