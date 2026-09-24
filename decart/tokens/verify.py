"""
Offline verification of client tokens.

Client tokens from ``client.tokens.create`` are EdDSA (Ed25519) JWTs signed by the
Decart platform. ``verify_client_token`` checks one against the platform's public
JWKS locally, so a backend can trust the claims signed into it (``service_tier``,
allowed models, expiry, ...) without a round trip. This is offline JWKS
verification, unrelated to the gateway's online ``POST /v1/verify``.

The JWKS is served by the platform host, not the API host the SDK talks to, so
the defaults are pinned to ``https://platform.decart.ai`` regardless of
``DecartClient.base_url``. Verification needs the ``verify`` extra
(``pip install 'decart[verify]'``); decoding does not.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from typing import Any, TypeVar

from ..errors import TokenDecodeError, TokenVerifyError
from .types import ClientTokenClaims, VerifiedClientToken

PLATFORM_URL = "https://platform.decart.ai"
DEFAULT_JWKS_URL = f"{PLATFORM_URL}/api/auth/jwks"
DEFAULT_ISSUER = PLATFORM_URL
DEFAULT_AUDIENCE = PLATFORM_URL
DEFAULT_LEEWAY = 30.0  # seconds of clock tolerance on exp

_ClaimsT = TypeVar("_ClaimsT", bound=ClientTokenClaims)


def _service_tier(payload: dict[str, Any]) -> int | None:
    """Same rules as the gateway: an int (not bool) or digit string naming a tier
    0-3, else no tier; the legacy ``priority: true`` alias means tier 3."""
    raw = payload.get("service_tier")
    tier: int | None = None
    if isinstance(raw, int) and not isinstance(raw, bool):
        tier = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        tier = int(raw)
    if tier not in (0, 1, 2, 3):
        tier = None
    if tier is None and payload.get("priority"):
        tier = 3
    return tier


def _claims(payload: dict[str, Any], model: type[_ClaimsT]) -> _ClaimsT:
    """Map a JWT payload onto ``model``; pydantic rejects payloads that are not client tokens."""
    exp = payload.get("exp")
    # Convert explicitly so expires_at is UTC-aware on every supported pydantic version.
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        exp = datetime.fromtimestamp(exp, tz=timezone.utc)
    return model.model_validate(
        {
            "user_id": payload.get("sub"),
            "organization_id": payload.get("organizationId"),
            "api_key_id": payload.get("parent_api_key_id") or payload.get("jti"),
            "api_key_name": payload.get("api_key_name"),
            "service_tier": _service_tier(payload),
            "allowed_models": payload.get("models"),
            "allowed_origins": payload.get("origins"),
            "constraints": payload.get("constraints"),
            "realtime_concurrent_session_limit": payload.get("realtimeConcurrentSessionLimit"),
            "zero_data_retention": bool(payload.get("zeroDataRetention")),
            "attribution": payload.get("attribution"),
            "expires_at": exp,
            "claims": payload,
        }
    )


def decode_client_token(token: str) -> ClientTokenClaims:
    """
    Decode a client token's claims **without verifying it**. No network, no extra.

    The result is untrusted: the signature is not checked and expiry is not
    enforced, so a forged token decodes like a real one. Never make authorization
    decisions on it; use ``verify_client_token`` for that.

    Raises:
        TokenDecodeError: If ``token`` is not a well-formed client-token JWT.
    """
    try:
        _header, payload, _signature = token.split(".")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if not isinstance(claims, dict):
            raise ValueError("payload is not a JSON object")
        return _claims(claims, ClientTokenClaims)
    except ValueError as e:  # also covers binascii, JSON and pydantic errors
        raise TokenDecodeError(f"Not a valid client token: {e}") from e


def _import_jwt() -> Any:
    try:
        import jwt
        import jwt.algorithms
    except ImportError as e:
        raise ImportError(
            "Offline client-token verification requires the 'verify' extra: "
            "pip install 'decart[verify]'"
        ) from e
    if not jwt.algorithms.has_crypto:
        raise ImportError(
            "Offline client-token verification requires the cryptography backend of "
            "PyJWT: pip install 'decart[verify]'"
        )
    return jwt


_jwks_clients: dict[str, Any] = {}  # jwks_url -> jwt.PyJWKClient, shared process-wide


def _jwks_client(jwt: Any, jwks_url: str) -> Any:
    client = _jwks_clients.get(jwks_url)
    if client is None:
        client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600, timeout=10)
        _jwks_clients[jwks_url] = client
    return client


async def verify_client_token(
    token: str,
    *,
    jwks_url: str = DEFAULT_JWKS_URL,
    issuer: str = DEFAULT_ISSUER,
    audience: str = DEFAULT_AUDIENCE,
    leeway: float = DEFAULT_LEEWAY,
) -> VerifiedClientToken:
    """
    Verify a client token offline against the platform's public JWKS and return its claims.

    Checks the EdDSA signature with the key named by the token's ``kid``, then
    ``exp`` (with ``leeway`` seconds of tolerance), ``iss`` and ``aud``. Signing
    keys are fetched from ``jwks_url`` on first use and cached in-process, so
    steady-state calls make no network requests and the token never leaves your
    process. Like the gateway, this does not consult the platform for early
    revocation: a token verifies until it expires (at most one hour).

    Requires the ``verify`` extra: ``pip install 'decart[verify]'``.

    Args:
        token: The signed ``token`` from ``client.tokens.create``.
        jwks_url: JWKS to verify against; defaults to the production platform's.
        issuer: Expected ``iss`` claim.
        audience: Expected ``aud`` claim.
        leeway: Clock tolerance in seconds for the expiry check.

    Raises:
        TokenVerifyError: If the token is malformed, tampered with, expired, from
            the wrong issuer or audience, signed by an unknown key, or the JWKS
            cannot be fetched. The message says which.
        ImportError: If the ``verify`` extra is not installed.

    Example:
        ```python
        verified = await verify_client_token(token)
        verified.service_tier  # e.g. 0
        verified.pool          # "free" for tier 0, else "paid"
        ```
    """
    jwt = _import_jwt()
    try:
        # PyJWKClient fetches over urllib, so keep it off the event loop.
        signing_key = await asyncio.to_thread(
            _jwks_client(jwt, jwks_url).get_signing_key_from_jwt, token
        )
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["EdDSA"],
            issuer=issuer,
            audience=audience,
            leeway=leeway,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
        return _claims(payload, VerifiedClientToken)
    except (jwt.PyJWTError, ValueError) as e:
        raise TokenVerifyError(f"Client token verification failed: {e}", cause=e) from e
