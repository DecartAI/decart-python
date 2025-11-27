import asyncio
from typing import Any, Optional, TYPE_CHECKING

import aiohttp
from pydantic import ValidationError

from ..models import ModelDefinition
from ..errors import InvalidInputError
from .request import submit_job, get_job_status, get_job_content
from .types import (
    JobSubmitResponse,
    JobStatusResponse,
    QueueJobResult,
    QueueJobResultCompleted,
    QueueJobResultFailed,
    OnStatusChangeCallback,
)

if TYPE_CHECKING:
    from ..client import DecartClient

POLLING_INTERVAL = 1.5  # seconds
INITIAL_DELAY = 0.5  # seconds


class QueueClient:
    """
    Queue client for async job-based video and image generation.

    Jobs are submitted and processed asynchronously, allowing you to
    poll for status and retrieve results when ready.

    Example:
        ```python
        client = DecartClient(api_key="your-key")

        # Option 1: Submit and poll automatically
        result = await client.queue.submit_and_poll({
            "model": models.video("lucy-pro-t2v"),
            "prompt": "A cat playing piano",
            "on_status_change": lambda job: print(f"Status: {job.status}"),
        })

        # Option 2: Submit and poll manually
        job = await client.queue.submit({
            "model": models.video("lucy-pro-t2v"),
            "prompt": "A cat playing piano",
        })
        status = await client.queue.status(job.job_id)
        result = await client.queue.result(job.job_id)
        ```
    """

    def __init__(self, parent: "DecartClient") -> None:
        self._parent = parent

    async def _get_session(self) -> aiohttp.ClientSession:
        return await self._parent._get_session()

    async def submit(self, options: dict[str, Any]) -> JobSubmitResponse:
        """
        Submit a job to the queue for async processing.
        Returns immediately with job_id and initial status.

        Args:
            options: Submit options including model and inputs

        Returns:
            JobSubmitResponse with job_id and status

        Raises:
            InvalidInputError: If inputs are invalid
            QueueSubmitError: If submission fails
        """
        if "model" not in options:
            raise InvalidInputError("model is required")

        model: ModelDefinition = options["model"]

        inputs = {k: v for k, v in options.items() if k not in ("model", "cancel_token")}

        # File fields that need special handling
        FILE_FIELDS = {"data", "start", "end"}

        # Separate file inputs from regular inputs
        file_inputs = {k: v for k, v in inputs.items() if k in FILE_FIELDS}
        non_file_inputs = {k: v for k, v in inputs.items() if k not in FILE_FIELDS}

        # Validate non-file inputs
        validation_inputs = {
            **non_file_inputs,
            **{k: b"" for k in file_inputs.keys()},
        }

        try:
            validated_inputs = model.input_schema(**validation_inputs)
        except ValidationError as e:
            raise InvalidInputError(f"Invalid inputs for {model.name}: {str(e)}") from e

        # Build final inputs
        processed_inputs = {
            **validated_inputs.model_dump(exclude_none=True),
            **file_inputs,
        }

        session = await self._get_session()
        return await submit_job(
            session=session,
            base_url=self._parent.base_url,
            api_key=self._parent.api_key,
            model=model,
            inputs=processed_inputs,
            integration=self._parent.integration,
        )

    async def status(self, job_id: str) -> JobStatusResponse:
        """
        Get the current status of a job.

        Args:
            job_id: The job ID returned from submit()

        Returns:
            JobStatusResponse with job_id and status

        Raises:
            QueueStatusError: If status check fails
        """
        session = await self._get_session()
        return await get_job_status(
            session=session,
            base_url=self._parent.base_url,
            api_key=self._parent.api_key,
            job_id=job_id,
            integration=self._parent.integration,
        )

    async def result(self, job_id: str) -> bytes:
        """
        Get the result of a completed job.
        Should only be called when job status is "completed".

        Args:
            job_id: The job ID returned from submit()

        Returns:
            Generated media as bytes

        Raises:
            QueueResultError: If result retrieval fails
        """
        session = await self._get_session()
        return await get_job_content(
            session=session,
            base_url=self._parent.base_url,
            api_key=self._parent.api_key,
            job_id=job_id,
            integration=self._parent.integration,
        )

    async def submit_and_poll(
        self,
        options: dict[str, Any],
    ) -> QueueJobResult:
        """
        Submit a job and automatically poll until completion.
        Returns a result object with status (does not throw on job failure).

        Args:
            options: Submit options including model, inputs, and optional on_status_change callback

        Returns:
            QueueJobResult - either completed with data or failed with error

        Raises:
            InvalidInputError: If inputs are invalid
            QueueSubmitError: If submission fails
            QueueStatusError: If status check fails
            QueueResultError: If result retrieval fails
        """
        on_status_change: Optional[OnStatusChangeCallback] = options.pop(
            "on_status_change", None
        )

        # Submit the job
        job = await self.submit(options)

        # Notify of initial status
        if on_status_change:
            on_status_change(JobStatusResponse(job_id=job.job_id, status=job.status))

        # Initial delay before polling
        await asyncio.sleep(INITIAL_DELAY)

        # Poll until complete
        while True:
            status = await self.status(job.job_id)

            if on_status_change:
                on_status_change(status)

            if status.status == "completed":
                data = await self.result(job.job_id)
                return QueueJobResultCompleted(status="completed", data=data)

            if status.status == "failed":
                return QueueJobResultFailed(status="failed", error="Job failed")

            # Still pending or processing
            await asyncio.sleep(POLLING_INTERVAL)
