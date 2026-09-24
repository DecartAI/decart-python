"""Tests for offline client-token verification and decoding."""

from __future__ import annotations

import base64
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from decart import (
    ClientTokenClaims,
    DecartClient,
    TokenDecodeError,
    TokenVerifyError,
    VerifiedClientToken,
    decode_client_token,
    verify_client_token,
)
from decart.tokens import verify as verify_module

PLATFORM = "https://platform.decart.ai"
KID = "test-kid-1"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_json(obj: Any) -> str:
    return _b64url(json.dumps(obj).encode())


def _jwk(public_key: ed25519.Ed25519PublicKey, kid: str = KID) -> dict[str, str]:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {"kty": "OKP", "crv": "Ed25519", "alg": "EdDSA", "kid": kid, "x": _b64url(raw)}


def _claims(**overrides: Any) -> dict[str, Any]:
    """The claim set the platform signs into a client token. ``None`` removes a claim."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": PLATFORM,
        "aud": PLATFORM,
        "iat": now,
        "exp": now + 60,
        "sub": "user_123",
        "organizationId": "org_456",
        "jti": "token_789",
        "parent_api_key_id": "key_abc",
        "api_key_name": "backend",
        "constraints": {"realtime": {"maxSessionDuration": 120}},
        "models": ["lucy-2.1"],
        "origins": ["https://example.com"],
        "zeroDataRetention": False,
        "realtimeConcurrentSessionLimit": 5,
        "service_tier": 2,
        "attribution": {"campaign": "launch"},
    }
    for key, value in overrides.items():
        if value is None:
            claims.pop(key, None)
        else:
            claims[key] = value
    return claims


def _unsigned(claims: dict[str, Any]) -> str:
    return f"{_b64url_json({'alg': 'EdDSA', 'kid': KID})}.{_b64url_json(claims)}.sig"


PRIVATE_KEY = ed25519.Ed25519PrivateKey.generate()
JWKS = {"keys": [_jwk(PRIVATE_KEY.public_key())]}


def _sign(claims: dict[str, Any], key: Any = PRIVATE_KEY, kid: str = KID) -> str:
    return jwt.encode(claims, key, algorithm="EdDSA", headers={"kid": kid})


@pytest.fixture(autouse=True)
def jwks_fetch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Serve JWKS from memory instead of the network, and reset the process-wide cache."""
    verify_module._jwks_clients.clear()
    fetch = MagicMock(return_value=JWKS)
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch)
    return fetch


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_verify_valid_token_exposes_claims() -> None:
    client = DecartClient(api_key="test-api-key")
    claims = _claims()

    verified = await client.tokens.verify(_sign(claims))

    assert isinstance(verified, VerifiedClientToken)
    assert verified.service_tier == 2
    assert verified.pool == "paid"
    assert verified.user_id == "user_123"
    assert verified.organization_id == "org_456"
    assert verified.api_key_name == "backend"
    assert verified.api_key_id == "key_abc"
    assert verified.allowed_models == ["lucy-2.1"]
    assert verified.allowed_origins == ["https://example.com"]
    assert verified.constraints == {"realtime": {"maxSessionDuration": 120}}
    assert verified.realtime_concurrent_session_limit == 5
    assert verified.zero_data_retention is False
    assert verified.attribution == {"campaign": "launch"}
    assert verified.expires_at == datetime.fromtimestamp(claims["exp"], tz=timezone.utc)
    assert verified.claims == claims
    # Keys come from the platform host, not the SDK's API base URL.
    assert list(verify_module._jwks_clients) == ["https://platform.decart.ai/api/auth/jwks"]


@pytest.mark.asyncio
async def test_verify_free_tier_and_fallbacks() -> None:
    free = await verify_client_token(_sign(_claims(service_tier=0)))
    assert free.service_tier == 0
    assert free.pool == "free"

    minimal = await verify_client_token(
        _sign({"iss": PLATFORM, "aud": PLATFORM, "exp": int(time.time()) + 60, "sub": "u"})
    )
    assert minimal.service_tier is None
    assert minimal.pool == "paid"
    assert minimal.api_key_id is None
    assert minimal.allowed_models is None

    no_parent = await verify_client_token(_sign(_claims(parent_api_key_id=None)))
    assert no_parent.api_key_id == "token_789"  # falls back to the token's own id


@pytest.mark.asyncio
async def test_verify_rejects_tampered_signature() -> None:
    header, payload, signature = _sign(_claims()).split(".")
    flipped = ("A" if signature[-1] != "A" else "B") + signature[1:]

    with pytest.raises(TokenVerifyError, match="Signature verification failed"):
        await verify_client_token(f"{header}.{payload}.{flipped}")


@pytest.mark.asyncio
async def test_verify_rejects_tampered_payload() -> None:
    header, _payload, signature = _sign(_claims(service_tier=0)).split(".")
    upgraded = f"{header}.{_b64url_json(_claims(service_tier=3))}.{signature}"

    with pytest.raises(TokenVerifyError, match="Signature verification failed"):
        await verify_client_token(upgraded)
    # decode happily reports the forged value: that is why it is untrusted.
    assert decode_client_token(upgraded).service_tier == 3


@pytest.mark.asyncio
async def test_verify_rejects_token_signed_by_another_key() -> None:
    token = _sign(_claims(), key=ed25519.Ed25519PrivateKey.generate())  # same kid, other key

    with pytest.raises(TokenVerifyError, match="Signature verification failed"):
        await verify_client_token(token)


@pytest.mark.asyncio
async def test_verify_rejects_expired_token_within_leeway_rules() -> None:
    with pytest.raises(TokenVerifyError, match="expired"):
        await verify_client_token(_sign(_claims(exp=int(time.time()) - 120)))

    slightly_stale = _sign(_claims(exp=int(time.time()) - 10))
    assert (await verify_client_token(slightly_stale)).user_id == "user_123"  # 30s leeway
    with pytest.raises(TokenVerifyError, match="expired"):
        await verify_client_token(slightly_stale, leeway=0)


@pytest.mark.asyncio
async def test_verify_rejects_wrong_issuer_or_audience() -> None:
    with pytest.raises(TokenVerifyError, match="[Ii]ssuer"):
        await verify_client_token(_sign(_claims(iss="https://evil.example.com")))
    with pytest.raises(TokenVerifyError, match="[Aa]udience"):
        await verify_client_token(_sign(_claims(aud="https://api.decart.ai")))
    with pytest.raises(TokenVerifyError, match="iss"):
        await verify_client_token(_sign(_claims(iss=None)))


@pytest.mark.asyncio
async def test_verify_rejects_other_algorithms() -> None:
    public_raw = PRIVATE_KEY.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    hmac_token = jwt.encode(_claims(), public_raw, algorithm="HS256", headers={"kid": KID})
    with pytest.raises(TokenVerifyError, match="alg"):
        await verify_client_token(hmac_token)

    none_token = f"{_b64url_json({'alg': 'none', 'kid': KID})}.{_b64url_json(_claims())}."
    with pytest.raises(TokenVerifyError, match="alg"):
        await verify_client_token(none_token)


@pytest.mark.asyncio
async def test_verify_rejects_unknown_or_missing_kid(jwks_fetch: MagicMock) -> None:
    with pytest.raises(TokenVerifyError, match="signing key"):
        await verify_client_token(_sign(_claims(), kid="not-in-jwks"))
    with pytest.raises(TokenVerifyError, match="signing key"):
        await verify_client_token(jwt.encode(_claims(), PRIVATE_KEY, algorithm="EdDSA"))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "not-a-jwt", "a.b", "eyJ.eyJ.sig"])
async def test_verify_rejects_malformed_tokens(bad: str) -> None:
    with pytest.raises(TokenVerifyError):
        await verify_client_token(bad)


@pytest.mark.asyncio
async def test_jwks_is_fetched_once_and_shared(jwks_fetch: MagicMock) -> None:
    client = DecartClient(api_key="test-api-key")
    token = _sign(_claims())

    await client.tokens.verify(token)
    await verify_client_token(token)
    await verify_client_token(_sign(_claims(sub="other")))

    assert jwks_fetch.call_count == 1


@pytest.mark.asyncio
async def test_rotated_key_is_fetched_on_unknown_kid(jwks_fetch: MagicMock) -> None:
    new_key = ed25519.Ed25519PrivateKey.generate()
    jwks_fetch.side_effect = [JWKS, {"keys": [_jwk(new_key.public_key(), kid="kid-2")]}]

    await verify_client_token(_sign(_claims()))
    verified = await verify_client_token(_sign(_claims(), key=new_key, kid="kid-2"))

    assert verified.user_id == "user_123"
    assert jwks_fetch.call_count == 2


@pytest.mark.asyncio
async def test_jwks_unavailable_fails_clearly(jwks_fetch: MagicMock) -> None:
    # What PyJWKClient.fetch_data raises when the platform is unreachable.
    jwks_fetch.side_effect = jwt.PyJWKClientError("Fail to fetch data from the url")

    with pytest.raises(TokenVerifyError, match="fetch"):
        await verify_client_token(_sign(_claims()))


@pytest.mark.asyncio
async def test_verify_without_extra_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _sign(_claims())
    monkeypatch.setitem(sys.modules, "jwt", None)
    monkeypatch.setitem(sys.modules, "jwt.algorithms", None)

    with pytest.raises(ImportError, match=r"decart\[verify\]"):
        await verify_client_token(token)
    assert decode_client_token(token).user_id == "user_123"  # decode never needed it


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


def test_decode_is_offline_and_unverified(jwks_fetch: MagicMock) -> None:
    claims = _claims(exp=int(time.time()) - 3600)
    forged_and_expired = _sign(claims, key=ed25519.Ed25519PrivateKey.generate())

    decoded = DecartClient(api_key="test-api-key").tokens.decode(forged_and_expired)

    jwks_fetch.assert_not_called()
    assert isinstance(decoded, ClientTokenClaims)
    assert not isinstance(decoded, VerifiedClientToken)
    assert decoded.service_tier == 2
    assert decoded.user_id == "user_123"
    assert decoded.api_key_id == "key_abc"
    assert decoded.expires_at == datetime.fromtimestamp(claims["exp"], tz=timezone.utc)
    assert decoded.claims == claims


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not-a-jwt",
        "a.b.c.d",
        "!!!.!!!.sig",
        f"h.{_b64url(b'not json')}.sig",
        f"h.{_b64url_json(['array'])}.sig",
        _unsigned({"exp": 1_700_000_000}),  # no sub
        _unsigned({"sub": "u"}),  # no exp
        _unsigned({"sub": "u", "exp": "soon"}),
    ],
)
def test_decode_rejects_malformed_tokens(bad: str) -> None:
    with pytest.raises(TokenDecodeError, match="Not a valid client token"):
        decode_client_token(bad)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"service_tier": 0}, 0),
        ({"service_tier": 3}, 3),
        ({"service_tier": "2"}, 2),  # digit strings count
        ({"service_tier": True}, None),  # booleans do not
        ({"service_tier": "pro"}, None),
        ({"service_tier": 7}, None),  # unknown tier
        ({"service_tier": None}, None),
        ({"service_tier": None, "priority": True}, 3),  # legacy alias
        ({"service_tier": 0, "priority": True}, 0),  # explicit tier wins
    ],
)
def test_service_tier_parsing(overrides: dict[str, Any], expected: int | None) -> None:
    decoded = decode_client_token(_unsigned(_claims(**overrides)))
    assert decoded.service_tier == expected
    assert decoded.pool == ("free" if expected == 0 else "paid")
