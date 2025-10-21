from .client import create_decart_client, DecartClient, DecartConfiguration
from .errors import DecartSDKError, ErrorCodes
from .models import models, ModelDefinition
from .types import FileInput, ModelState, Prompt
from .process import ProcessClient

try:
    from .realtime import (
        RealtimeClient,
        RealtimeClientFactory,
        RealtimeConnectOptions,
        ConnectionState,
    )

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeClient = None  # type: ignore
    RealtimeClientFactory = None  # type: ignore
    RealtimeConnectOptions = None  # type: ignore
    ConnectionState = None  # type: ignore

__version__ = "0.0.1"

__all__ = [
    "create_decart_client",
    "DecartClient",
    "DecartConfiguration",
    "DecartSDKError",
    "ErrorCodes",
    "models",
    "ModelDefinition",
    "FileInput",
    "ModelState",
    "Prompt",
    "ProcessClient",
]

if REALTIME_AVAILABLE:
    __all__.extend(
        [
            "RealtimeClient",
            "RealtimeClientFactory",
            "RealtimeConnectOptions",
            "ConnectionState",
        ]
    )
