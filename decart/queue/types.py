from typing import Literal, Callable, Union
from pydantic import BaseModel


class JobSubmitResponse(BaseModel):
    """Response from POST /v1/jobs/{model} - job submission."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]


class JobStatusResponse(BaseModel):
    """Response from GET /v1/jobs/{job_id} - job status check."""

    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]


class QueueJobResultCompleted(BaseModel):
    """Result when job completes successfully."""

    status: Literal["completed"]
    data: bytes


class QueueJobResultFailed(BaseModel):
    """Result when job fails."""

    status: Literal["failed"]
    error: str


QueueJobResult = Union[QueueJobResultCompleted, QueueJobResultFailed]

JobStatus = Literal["pending", "processing", "completed", "failed"]

OnStatusChangeCallback = Callable[[JobStatusResponse], None]
