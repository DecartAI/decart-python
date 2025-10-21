from typing import Union, Optional, Protocol, runtime_checkable
from pathlib import Path
from pydantic import BaseModel, Field


@runtime_checkable
class HasRead(Protocol):
    """Protocol for file-like objects with a read method."""

    def read(self) -> Union[bytes, str]: ...


FileInput = Union[HasRead, bytes, str, Path]


class Prompt(BaseModel):
    text: str = Field(..., min_length=1)
    enrich: bool = Field(default=True)


class ModelState(BaseModel):
    prompt: Optional[Prompt] = None
    mirror: bool = Field(default=False)
