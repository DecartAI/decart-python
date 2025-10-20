from typing import BinaryIO, Union
from pydantic import BaseModel, Field


FileInput = Union[BinaryIO, bytes, str]


class Prompt(BaseModel):
    text: str = Field(..., min_length=1)
    enrich: bool = Field(default=True)


class ModelState(BaseModel):
    prompt: Prompt | None = None
    mirror: bool = Field(default=False)
