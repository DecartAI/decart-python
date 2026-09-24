from datetime import datetime
from typing import Any, Literal

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


class ClientTokenClaims(BaseModel):
    """
    Claims of a client token, under the names the gateway derives from them.

    ``decode_client_token`` returns this **without verification**: treat it as
    untrusted input. ``verify_client_token`` returns the ``VerifiedClientToken``
    subclass once the signature, issuer, audience and expiry have been checked.
    """

    user_id: str  # sub
    organization_id: str | None = None  # organizationId
    api_key_id: str | None = None  # parent_api_key_id when minted with an API key, else jti
    api_key_name: str | None = None
    service_tier: int | None = None  # 0 free, 1 user, 2 pro, 3 priority; None when unset
    allowed_models: list[str] | None = None  # models; None = unrestricted
    allowed_origins: list[str] | None = None  # origins; None = any
    constraints: TokenConstraints | None = None
    realtime_concurrent_session_limit: int | None = None
    zero_data_retention: bool = False
    attribution: dict[str, str] | None = None  # metadata.attribution usage labels
    expires_at: datetime  # exp, UTC
    claims: dict[str, Any]  # raw payload, for anything not mapped above

    @property
    def pool(self) -> Literal["free", "paid"]:
        """``"free"`` when ``service_tier`` is 0, otherwise ``"paid"`` (including no tier)."""
        return "free" if self.service_tier == 0 else "paid"


class VerifiedClientToken(ClientTokenClaims):
    """Claims whose signature, issuer, audience and expiry were verified against the platform JWKS."""
