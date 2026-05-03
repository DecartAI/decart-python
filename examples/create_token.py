import asyncio
import os
from decart import DecartClient


async def main() -> None:
    # Server-side: Create client token using API key
    async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as server_client:
        print("Creating client token...")

        token = await server_client.tokens.create(
            allowed_origins=["https://example.com"],
        )

        print("Token created successfully:")
        print(f"  API Key: {token.api_key[:10]}...")
        print(f"  Expires At: {token.expires_at}")
        origins = (token.permissions or {}).get("origins")
        print(f"  Allowed Origins: {', '.join(origins) if origins else '(any)'}")

        # Client-side: Use the client token
        # In a real app, you would send token.api_key to the frontend
        _client = DecartClient(api_key=token.api_key)

        print("Client created with client token.")
        print("This token can now be used for realtime connections.")


if __name__ == "__main__":
    asyncio.run(main())
