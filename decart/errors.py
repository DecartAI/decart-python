from typing import Any, Optional

# Maximum file size allowed for uploads (20MB)
MAX_FILE_SIZE = 20 * 1024 * 1024


class DecartSDKError(Exception):
    """Base exception for all Decart SDK errors."""

    def __init__(
        self,
        message: str,
        data: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.message = message
        self.data = data or {}
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class InvalidAPIKeyError(DecartSDKError):
    """Raised when API key is invalid or missing."""

    def __init__(self) -> None:
        super().__init__(
            "Missing API key. Pass `api_key` to DecartClient() or set the DECART_API_KEY environment variable."
        )


class InvalidBaseURLError(DecartSDKError):
    """Raised when base URL is invalid."""

    def __init__(self, url: Optional[str] = None) -> None:
        message = f"Invalid base URL: {url}" if url else "Invalid base URL"
        super().__init__(message)


class WebRTCError(DecartSDKError):
    """Raised when WebRTC connection fails."""

    def __init__(self, message: str = "WebRTC error", cause: Optional[Exception] = None) -> None:
        super().__init__(message, cause=cause)


class InvalidInputError(DecartSDKError):
    """Raised when input validation fails."""

    pass


class ModelNotFoundError(DecartSDKError):
    """Raised when model is not found."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Model {model} not found")
        self.model = model


class ProcessingError(DecartSDKError):
    """Raised when processing fails."""

    pass


class QueueSubmitError(DecartSDKError):
    """Raised when queue job submission fails."""

    pass


class QueueStatusError(DecartSDKError):
    """Raised when getting queue job status fails."""

    pass


class QueueResultError(DecartSDKError):
    """Raised when getting queue job result fails."""

    pass


class TokenCreateError(DecartSDKError):
    """Raised when token creation fails."""

    pass


class FileTooLargeError(DecartSDKError):
    """Raised when a file exceeds the maximum allowed size."""

    def __init__(self, file_size: int, max_size: int, field_name: str | None = None) -> None:
        file_size_mb = f"{file_size / (1024 * 1024):.1f}"
        max_size_mb = f"{max_size / (1024 * 1024):.0f}"
        field = f" for field '{field_name}'" if field_name else ""
        super().__init__(
            f"File size{field} ({file_size_mb}MB) exceeds the maximum allowed size of {max_size_mb}MB. "
            f"Please reduce the file size or resolution before uploading.",
            data={"file_size": file_size, "max_size": max_size, "field_name": field_name},
        )
        self.file_size = file_size
        self.max_size = max_size
        self.field_name = field_name
