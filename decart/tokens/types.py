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
    token: str | None = None
    """Signed JWT mirroring ``api_key``, verifiable offline via the public JWKS."""
    expires_at: str
    permissions: TokenPermissions | None = None
    constraints: TokenConstraints | None = None
