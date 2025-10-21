from typing import Any, Optional


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
        super().__init__("API key is required and must be a non-empty string")


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
