from typing import TYPE_CHECKING, Any, Union

import aiohttp

from ..errors import TokenCreateError
from ..models import Model
from .._user_agent import build_user_agent
from .types import ClientTokenClaims, CreateTokenResponse, TokenConstraints, VerifiedClientToken
from .verify import (
    DEFAULT_AUDIENCE,
    DEFAULT_ISSUER,
    DEFAULT_JWKS_URL,
    DEFAULT_LEEWAY,
    decode_client_token,
    verify_client_token,
)

if TYPE_CHECKING:
    from ..client import DecartClient


class TokensClient:
    """
    Client for creating and verifying client tokens.
    Client tokens are short-lived API keys safe for client-side use.

    Example:
        ```python
        client = DecartClient(api_key=os.getenv("DECART_API_KEY"))
        token = await client.tokens.create()
        # Returns: CreateTokenResponse(api_key="ek_...", token="eyJhbGciOiJFZERTQS...", expires_at="...")

        # With metadata:
        token = await client.tokens.create(metadata={"role": "viewer"})

        # With expiry, model restrictions, origin restrictions, and constraints:
        token = await client.tokens.create(
            expires_in=120,
            allowed_models=["lucy-2.1"],
            allowed_origins=["https://example.com"],
            constraints={"realtime": {"maxSessionDuration": 300}},
        )

        # Verify a token offline against the platform JWKS (needs `decart[verify]`):
        verified = await client.tokens.verify(token.token)
        verified.service_tier, verified.pool, verified.user_id
        ```
    """

    def __init__(self, parent: "DecartClient") -> None:
        self._parent = parent

    async def _get_session(self) -> aiohttp.ClientSession:
        return await self._parent._get_session()

    async def create(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        expires_in: int | None = None,
        allowed_models: list[Union[Model, str]] | None = None,
        allowed_origins: list[str] | None = None,
        constraints: TokenConstraints | None = None,
    ) -> CreateTokenResponse:
        """
        Create a client token.

        Args:
            metadata: Optional custom key-value pairs to attach to the token.
            expires_in: Seconds until the token expires (1-3600, default 60).
            allowed_models: Restrict which models this token can access (max 20).
            allowed_origins: Restrict which web origins this token can be used
                from (max 20). Each entry must be a full origin including
                scheme, e.g. ``https://example.com``. Enforced on realtime
                sessions by matching the WebSocket ``Origin`` header verbatim.
                Defense-in-depth — only effective for browser-based clients.
            constraints: Operational limits, e.g.
                ``{"realtime": {"maxSessionDuration": 120}}``.

        Returns:
            A short-lived client token: the signed ``token`` your frontend uses for
            realtime connections and file uploads, plus its opaque ``api_key`` twin.

        Example:
            ```python
            token = await client.tokens.create()
            # Returns: CreateTokenResponse(api_key="ek_...", token="eyJhbGciOiJFZERTQS...", expires_at="...")

            # With all options:
            token = await client.tokens.create(
                metadata={"role": "viewer"},
                expires_in=120,
                allowed_models=["lucy-2.1"],
                allowed_origins=["https://example.com"],
                constraints={"realtime": {"maxSessionDuration": 300}},
            )
            ```

        Raises:
            TokenCreateError: If token creation fails (401, 403, etc.)
        """
        session = await self._get_session()
        endpoint = f"{self._parent.base_url}/v1/client/tokens"

        headers = {
            "X-API-KEY": self._parent.api_key,
            "User-Agent": build_user_agent(self._parent.integration),
        }

        body: dict[str, Any] = {}
        if metadata is not None:
            body["metadata"] = metadata
        if expires_in is not None:
            body["expiresIn"] = expires_in
        if allowed_models is not None:
            body["allowedModels"] = list(allowed_models)
        if allowed_origins is not None:
            body["allowedOrigins"] = list(allowed_origins)
        if constraints is not None:
            body["constraints"] = constraints

        async with session.post(
            endpoint,
            headers=headers,
            json=body,
        ) as response:
            if not response.ok:
                error_text = await response.text()
                raise TokenCreateError(
                    f"Failed to create token: {response.status} - {error_text}",
                    data={"status": response.status},
                )
            data = await response.json()
            if "token" not in data:
                # The platform guarantees the signed token; a response without it
                # is a contract violation, not a value to hand back as None.
                raise TokenCreateError(
                    "Failed to create token: response is missing the signed token",
                    data={"status": response.status},
                )
            return CreateTokenResponse(
                api_key=data["apiKey"],
                token=data["token"],
                expires_at=data["expiresAt"],
                permissions=data.get("permissions"),
                constraints=data.get("constraints"),
            )

    async def verify(
        self,
        token: str,
        *,
        jwks_url: str = DEFAULT_JWKS_URL,
        issuer: str = DEFAULT_ISSUER,
        audience: str = DEFAULT_AUDIENCE,
        leeway: float = DEFAULT_LEEWAY,
    ) -> VerifiedClientToken:
        """
        Verify a client token offline against the platform's public JWKS.

        Same as the module-level ``verify_client_token``: checks the EdDSA
        signature, ``exp`` (with ``leeway``), ``iss`` and ``aud`` and returns the
        claims signed into the token. The JWKS comes from the platform host (not
        this client's ``base_url``) and is cached in-process. Requires the
        ``verify`` extra: ``pip install 'decart[verify]'``.

        Example:
            ```python
            verified = await client.tokens.verify(token.token)
            verified.service_tier, verified.pool, verified.user_id
            ```

        Raises:
            TokenVerifyError: If the token is malformed, tampered with, expired,
                from the wrong issuer or audience, or signed by an unknown key.
            ImportError: If the ``verify`` extra is not installed.
        """
        return await verify_client_token(
            token, jwks_url=jwks_url, issuer=issuer, audience=audience, leeway=leeway
        )

    def decode(self, token: str) -> ClientTokenClaims:
        """
        Decode a client token's claims **without verifying it** (no network, no extra).
        Same as the module-level ``decode_client_token``. The result is untrusted:
        use ``verify()`` before acting on a token you received from elsewhere.

        Raises:
            TokenDecodeError: If the string is not a well-formed client-token JWT.
        """
        return decode_client_token(token)
