from typing import TYPE_CHECKING, Any, Union

import aiohttp

from ..errors import TokenCreateError
from ..models import Model
from .._user_agent import build_user_agent
from .types import CreateTokenResponse, TokenConstraints

if TYPE_CHECKING:
    from ..client import DecartClient


class TokensClient:
    """
    Client for creating client tokens.
    Client tokens are short-lived API keys safe for client-side use.

    Example:
        ```python
        client = DecartClient(api_key=os.getenv("DECART_API_KEY"))
        token = await client.tokens.create()
        # Returns: CreateTokenResponse(api_key="ek_...", expires_at="...")

        # With metadata:
        token = await client.tokens.create(metadata={"role": "viewer"})

        # With expiry, model restrictions, and constraints:
        token = await client.tokens.create(
            expires_in=120,
            allowed_models=["lucy_2_rt"],
            constraints={"realtime": {"maxSessionDuration": 300}},
        )
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
        constraints: TokenConstraints | None = None,
    ) -> CreateTokenResponse:
        """
        Create a client token.

        Args:
            metadata: Optional custom key-value pairs to attach to the token.
            expires_in: Seconds until the token expires (1-3600, default 60).
            allowed_models: Restrict which models this token can access (max 20).
            constraints: Operational limits, e.g.
                ``{"realtime": {"maxSessionDuration": 120}}``.

        Returns:
            A short-lived API key safe for client-side use.

        Example:
            ```python
            token = await client.tokens.create()
            # Returns: CreateTokenResponse(api_key="ek_...", expires_at="...")

            # With all options:
            token = await client.tokens.create(
                metadata={"role": "viewer"},
                expires_in=120,
                allowed_models=["lucy_2_rt"],
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
            return CreateTokenResponse(
                api_key=data["apiKey"],
                expires_at=data["expiresAt"],
                permissions=data.get("permissions"),
                constraints=data.get("constraints"),
            )
