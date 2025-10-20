from typing import Any, Optional


class DecartSDKError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        data: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = data or {}
        self.cause = cause
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return f"DecartSDKError(code={self.code!r}, message={self.message!r})"


class ErrorCodes:
    INVALID_API_KEY = "INVALID_API_KEY"
    INVALID_BASE_URL = "INVALID_BASE_URL"
    WEB_RTC_ERROR = "WEB_RTC_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_OPTIONS = "INVALID_OPTIONS"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"


def create_invalid_api_key_error() -> DecartSDKError:
    return DecartSDKError(
        ErrorCodes.INVALID_API_KEY,
        "API key is required and must be a non-empty string",
    )


def create_invalid_base_url_error(url: Optional[str] = None) -> DecartSDKError:
    message = f"Invalid base URL: {url}" if url else "Invalid base URL"
    return DecartSDKError(ErrorCodes.INVALID_BASE_URL, message)


def create_webrtc_error(error: Exception) -> DecartSDKError:
    return DecartSDKError(ErrorCodes.WEB_RTC_ERROR, "WebRTC error", cause=error)


def create_invalid_input_error(message: str) -> DecartSDKError:
    return DecartSDKError(ErrorCodes.INVALID_INPUT, message)


def create_model_not_found_error(model: str) -> DecartSDKError:
    return DecartSDKError(ErrorCodes.MODEL_NOT_FOUND, f"Model {model} not found")


def create_processing_error(message: str) -> DecartSDKError:
    return DecartSDKError(ErrorCodes.PROCESSING_ERROR, message)
