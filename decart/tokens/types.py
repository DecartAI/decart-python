from typing_extensions import TypedDict

from pydantic import BaseModel


class RealtimeConstraints(TypedDict, total=False):
    maxSessionDuration: int


class TokenConstraints(TypedDict, total=False):
    realtime: RealtimeConstraints


class TokenPermissions(TypedDict, total=False):
    models: list[str]
    origins: list[str]


class CreateTokenResponse(BaseModel):
    """Response from creating a client token."""

    api_key: str
    """Opaque ``ek_...`` form of the credential, verified online. Use it for calls
    other than realtime and file uploads."""
    token: str
    """Signed JWT carrying the same scope and expiry as ``api_key``. The gateway
    verifies it offline against the public JWKS; hand this to your frontend for
    realtime connections and file uploads."""
    expires_at: str
    permissions: TokenPermissions | None = None
    constraints: TokenConstraints | None = None
