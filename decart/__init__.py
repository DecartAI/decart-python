from .client import DecartClient
from .errors import (
    DecartSDKError,
    InvalidAPIKeyError,
    InvalidBaseURLError,
    InvalidInputError,
    ModelNotFoundError,
    ProcessingError,
    WebRTCError,
    QueueSubmitError,
    QueueStatusError,
    QueueResultError,
    TokenCreateError,
    TokenDecodeError,
    TokenVerifyError,
)
from .models import (
    models,
    ModelDefinition,
    CustomModelDefinition,
    VideoRestyleInput,
    RealtimeSpeed,
)
from .types import FileInput, ModelState, Prompt
from .queue import (
    QueueClient,
    JobStatus,
    JobSubmitResponse,
    JobStatusResponse,
    QueueJobResult,
)
from .tokens import (
    TokensClient,
    ClientTokenClaims,
    CreateTokenResponse,
    RealtimeConstraints,
    TokenConstraints,
    TokenPermissions,
    VerifiedClientToken,
    decode_client_token,
    verify_client_token,
)

try:
    from .realtime import (
        RealtimeClient,
        SetInput,
        SubscribeClient,
        SubscribeOptions,
        encode_subscribe_token,
        decode_subscribe_token,
        RealtimeConnectOptions,
        ConnectionState,
    )

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    RealtimeClient = None  # type: ignore
    SetInput = None  # type: ignore
    SubscribeClient = None  # type: ignore
    SubscribeOptions = None  # type: ignore
    encode_subscribe_token = None  # type: ignore
    decode_subscribe_token = None  # type: ignore
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
    "QueueSubmitError",
    "QueueStatusError",
    "QueueResultError",
    "models",
    "ModelDefinition",
    "CustomModelDefinition",
    "VideoRestyleInput",
    "RealtimeSpeed",
    "FileInput",
    "ModelState",
    "Prompt",
    "QueueClient",
    "JobStatus",
    "JobSubmitResponse",
    "JobStatusResponse",
    "QueueJobResult",
    "TokensClient",
    "ClientTokenClaims",
    "CreateTokenResponse",
    "RealtimeConstraints",
    "TokenConstraints",
    "TokenPermissions",
    "VerifiedClientToken",
    "decode_client_token",
    "verify_client_token",
    "TokenCreateError",
    "TokenDecodeError",
    "TokenVerifyError",
]

if REALTIME_AVAILABLE:
    __all__.extend(
        [
            "RealtimeClient",
            "SetInput",
            "SubscribeClient",
            "SubscribeOptions",
            "encode_subscribe_token",
            "decode_subscribe_token",
            "RealtimeConnectOptions",
            "ConnectionState",
        ]
    )
