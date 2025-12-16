from pydantic import BaseModel


class CreateTokenResponse(BaseModel):
    """Response from creating a client token."""

    api_key: str
    expires_at: str
