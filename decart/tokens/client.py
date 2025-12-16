from typing import TYPE_CHECKING

import aiohttp

from ..errors import TokenCreateError
from .._user_agent import build_user_agent
from .types import CreateTokenResponse

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
        ```
    """

    def __init__(self, parent: "DecartClient") -> None:
        self._parent = parent

    async def _get_session(self) -> aiohttp.ClientSession:
        return await self._parent._get_session()

    async def create(self) -> CreateTokenResponse:
        """
        Create a client token.

        Returns:
            A short-lived API key safe for client-side use.

        Example:
            ```python
            token = await client.tokens.create()
            # Returns: CreateTokenResponse(api_key="ek_...", expires_at="...")
            ```

        Raises:
            TokenCreateError: If token creation fails (401, 403, etc.)
        """
        session = await self._get_session()
        endpoint = f"{self._parent.base_url}/v1/client/tokens"

        async with session.post(
            endpoint,
            headers={
                "X-API-KEY": self._parent.api_key,
                "User-Agent": build_user_agent(self._parent.integration),
            },
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
            )
