from .client import DecartClient
from .errors import (
    DecartSDKError,
    InvalidAPIKeyError,
    InvalidBaseURLError,
    InvalidInputError,
    ModelNotFoundError,
    ProcessingError,
    WebRTCError,
)
from .models import models, ModelDefinition
from .types import FileInput, ModelState, Prompt

try:
    from .realtime import (
        RealtimeClient,
        RealtimeConnectOptions,
        ConnectionState,
    )

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeClient = None  # type: ignore
    RealtimeConnectOptions = None  # type: ignore
    ConnectionState = None  # type: ignore

__version__ = "0.0.1"

__all__ = [
    "DecartClient",
    "DecartSDKError",
    "InvalidAPIKeyError",
    "InvalidBaseURLError",
    "InvalidInputError",
    "ModelNotFoundError",
    "ProcessingError",
    "WebRTCError",
    "models",
    "ModelDefinition",
    "FileInput",
    "ModelState",
    "Prompt",
]

if REALTIME_AVAILABLE:
    __all__.extend(
        [
            "RealtimeClient",
            "RealtimeConnectOptions",
            "ConnectionState",
        ]
    )
