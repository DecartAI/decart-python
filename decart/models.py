import warnings
from typing import Literal, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .errors import ModelNotFoundError
from .types import FileInput

RealTimeModels = Literal[
    # Canonical names
    "lucy-2.1",
    "lucy-2.5",
    "lucy-vton-2",
    "lucy-vton-3",
    "lucy-restyle-2",
    # Latest aliases (server-side resolution)
    "lucy-latest",
    "lucy-vton-latest",
    "lucy-restyle-latest",
    # Deprecated names
    "lucy-2.1-vton-2",
    "mirage_v2",
]
VideoModels = Literal[
    # Canonical names
    "lucy-clip",
    "lucy-2.1",
    "lucy-2.5",
    "lucy-vton-2",
    "lucy-vton-3",
    "lucy-restyle-2",
    # Latest aliases (server-side resolution)
    "lucy-latest",
    "lucy-vton-latest",
    "lucy-restyle-latest",
    "lucy-clip-latest",
    # Deprecated / alias names
    "lucy-2.1-vton-2",
    "lucy-pro-v2v",
    "lucy-restyle-v2v",
]
ImageModels = Literal[
    # Canonical names
    "lucy-image-2",
    # Latest alias (server-side resolution)
    "lucy-image-latest",
    # Deprecated names
    "lucy-pro-i2i",
]
Model = Literal[RealTimeModels, VideoModels, ImageModels]

MODEL_ALIASES: dict[str, str] = {
    # Realtime aliases
    "mirage_v2": "lucy-restyle-2",
    # Video aliases
    "lucy-pro-v2v": "lucy-clip",
    "lucy-restyle-v2v": "lucy-restyle-2",
    # VTON aliases
    "lucy-2.1-vton-2": "lucy-vton-2",
    # Image aliases
    "lucy-pro-i2i": "lucy-image-2",
}

_warned_aliases: set[str] = set()


def _warn_deprecated(model: str) -> None:
    canonical = MODEL_ALIASES.get(model)
    if canonical and model not in _warned_aliases:
        _warned_aliases.add(model)
        warnings.warn(
            f'Model "{model}" is deprecated. Use "{canonical}" instead. '
            f"See https://docs.platform.decart.ai/models for details.",
            DeprecationWarning,
            stacklevel=3,
        )


# Type variable for model name
ModelT = TypeVar("ModelT", bound=str)


class DecartBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelDefinition(DecartBaseModel, Generic[ModelT]):
    name: ModelT
    url_path: str
    fps: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    input_schema: Optional[type[BaseModel]] = None


# Type aliases for model definitions that support specific APIs
ImageModelDefinition = ModelDefinition[ImageModels]
"""Type alias for model definitions that support synchronous processing (process API)."""

VideoModelDefinition = ModelDefinition[VideoModels]
"""Type alias for model definitions that support queue processing (queue API)."""

RealTimeModelDefinition = ModelDefinition[RealTimeModels]
"""Type alias for model definitions that support realtime streaming."""

CustomModelDefinition = ModelDefinition[str]
"""Type alias for model definitions with arbitrary (non-registry) model names."""


class VideoToVideoInput(DecartBaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    data: FileInput
    reference_image: Optional[FileInput] = None
    seed: Optional[int] = None
    resolution: Optional[str] = None
    enhance_prompt: Optional[bool] = None


class VideoRestyleInput(DecartBaseModel):
    """Input for lucy-restyle-v2v model.

    Must provide either `prompt` OR `reference_image`, but not both.
    `enhance_prompt` is only valid when using `prompt`, not `reference_image`.
    """

    prompt: Optional[str] = Field(default=None, min_length=1, max_length=1000)
    reference_image: Optional[FileInput] = None
    data: FileInput
    seed: Optional[int] = None
    resolution: Optional[str] = None
    enhance_prompt: Optional[bool] = None

    @model_validator(mode="after")
    def validate_prompt_or_reference_image(self) -> "VideoRestyleInput":
        has_prompt = self.prompt is not None
        has_reference_image = self.reference_image is not None

        if has_prompt == has_reference_image:
            raise ValueError("Must provide either 'prompt' or 'reference_image', but not both")

        if has_reference_image and self.enhance_prompt is not None:
            raise ValueError(
                "'enhance_prompt' is only valid when using 'prompt', not 'reference_image'"
            )

        return self


class VideoEdit2Input(DecartBaseModel):
    """Input for Lucy 2.1 video editing models.

    Prompt is required but can be an empty string.
    Optional reference_image can also be provided.
    """

    prompt: str = Field(..., max_length=1000)
    reference_image: Optional[FileInput] = None
    data: FileInput
    seed: Optional[int] = None
    resolution: Optional[str] = None
    enhance_prompt: Optional[bool] = None


class ImageToImageInput(DecartBaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    data: FileInput
    reference_image: Optional[FileInput] = None
    seed: Optional[int] = None
    resolution: Optional[str] = None
    enhance_prompt: Optional[bool] = None


_MODELS = {
    "realtime": {
        # Canonical names
        "lucy-2.1": ModelDefinition(
            name="lucy-2.1",
            url_path="/v1/stream",
            fps=30,
            width=1088,
            height=624,
        ),
        "lucy-2.5": ModelDefinition(
            name="lucy-2.5",
            url_path="/v1/stream",
            fps=30,
            width=1280,
            height=720,
        ),
        "lucy-restyle-2": ModelDefinition(
            name="lucy-restyle-2",
            url_path="/v1/stream",
            fps=30,
            width=1280,
            height=704,
        ),
        # Latest aliases (server-side resolution)
        "lucy-latest": ModelDefinition(
            name="lucy-latest",
            url_path="/v1/stream",
            fps=30,
            width=1088,
            height=624,
        ),
        "lucy-vton-latest": ModelDefinition(
            name="lucy-vton-latest",
            url_path="/v1/stream",
            fps=30,
            width=1088,
            height=624,
        ),
        "lucy-vton-2": ModelDefinition(
            name="lucy-vton-2",
            url_path="/v1/stream",
            fps=30,
            width=1088,
            height=624,
        ),
        "lucy-vton-3": ModelDefinition(
            name="lucy-vton-3",
            url_path="/v1/stream",
            fps=30,
            width=1088,
            height=624,
        ),
        "lucy-2.1-vton-2": ModelDefinition(
            name="lucy-2.1-vton-2",
            url_path="/v1/stream",
            fps=30,
            width=1088,
            height=624,
        ),
        "lucy-restyle-latest": ModelDefinition(
            name="lucy-restyle-latest",
            url_path="/v1/stream",
            fps=30,
            width=1280,
            height=704,
        ),
        # Deprecated names
        "mirage_v2": ModelDefinition(
            name="mirage_v2",
            url_path="/v1/stream",
            fps=30,
            width=1280,
            height=704,
        ),
    },
    "video": {
        # Canonical names
        "lucy-clip": ModelDefinition(
            name="lucy-clip",
            url_path="/v1/jobs/lucy-clip",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoToVideoInput,
        ),
        "lucy-2.1": ModelDefinition(
            name="lucy-2.1",
            url_path="/v1/jobs/lucy-2.1",
            fps=20,
            width=1088,
            height=624,
            input_schema=VideoEdit2Input,
        ),
        "lucy-2.5": ModelDefinition(
            name="lucy-2.5",
            url_path="/v1/jobs/lucy-2.5",
            fps=20,
            width=1280,
            height=720,
            input_schema=VideoEdit2Input,
        ),
        "lucy-vton-2": ModelDefinition(
            name="lucy-vton-2",
            url_path="/v1/jobs/lucy-vton-2",
            fps=20,
            width=1088,
            height=624,
            input_schema=VideoEdit2Input,
        ),
        "lucy-vton-3": ModelDefinition(
            name="lucy-vton-3",
            url_path="/v1/jobs/lucy-vton-3",
            fps=20,
            width=1088,
            height=624,
            input_schema=VideoEdit2Input,
        ),
        "lucy-2.1-vton-2": ModelDefinition(
            name="lucy-2.1-vton-2",
            url_path="/v1/jobs/lucy-2.1-vton-2",
            fps=20,
            width=1088,
            height=624,
            input_schema=VideoEdit2Input,
        ),
        "lucy-restyle-2": ModelDefinition(
            name="lucy-restyle-2",
            url_path="/v1/jobs/lucy-restyle-2",
            fps=22,
            width=1280,
            height=704,
            input_schema=VideoRestyleInput,
        ),
        # Latest aliases (server-side resolution)
        "lucy-latest": ModelDefinition(
            name="lucy-latest",
            url_path="/v1/jobs/lucy-latest",
            fps=20,
            width=1088,
            height=624,
            input_schema=VideoEdit2Input,
        ),
        "lucy-vton-latest": ModelDefinition(
            name="lucy-vton-latest",
            url_path="/v1/jobs/lucy-vton-latest",
            fps=20,
            width=1088,
            height=624,
            input_schema=VideoEdit2Input,
        ),
        "lucy-restyle-latest": ModelDefinition(
            name="lucy-restyle-latest",
            url_path="/v1/jobs/lucy-restyle-latest",
            fps=22,
            width=1280,
            height=704,
            input_schema=VideoRestyleInput,
        ),
        "lucy-clip-latest": ModelDefinition(
            name="lucy-clip-latest",
            url_path="/v1/jobs/lucy-clip-latest",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoToVideoInput,
        ),
        # Deprecated names
        "lucy-pro-v2v": ModelDefinition(
            name="lucy-pro-v2v",
            url_path="/v1/jobs/lucy-pro-v2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoToVideoInput,
        ),
        "lucy-restyle-v2v": ModelDefinition(
            name="lucy-restyle-v2v",
            url_path="/v1/jobs/lucy-restyle-v2v",
            fps=22,
            width=1280,
            height=704,
            input_schema=VideoRestyleInput,
        ),
    },
    "image": {
        # Canonical names
        "lucy-image-2": ModelDefinition(
            name="lucy-image-2",
            url_path="/v1/generate/lucy-image-2",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToImageInput,
        ),
        # Latest alias (server-side resolution)
        "lucy-image-latest": ModelDefinition(
            name="lucy-image-latest",
            url_path="/v1/generate/lucy-image-latest",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToImageInput,
        ),
        # Deprecated names
        "lucy-pro-i2i": ModelDefinition(
            name="lucy-pro-i2i",
            url_path="/v1/generate/lucy-pro-i2i",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToImageInput,
        ),
    },
}


class Models:
    @staticmethod
    def realtime(model: RealTimeModels) -> RealTimeModelDefinition:
        """Get a realtime model definition for WebRTC streaming."""
        _warn_deprecated(model)
        try:
            return _MODELS["realtime"][model]  # type: ignore[return-value]
        except KeyError:
            raise ModelNotFoundError(model)

    @staticmethod
    def video(model: VideoModels) -> VideoModelDefinition:
        """
        Get a video model definition.
        Video models only support the queue API.

        Available models:
            - "lucy-clip" - Video-to-video
            - "lucy-2.1" - Video editing (newer, higher quality)
            - "lucy-2.5" - Video editing (latest generation)
            - "lucy-restyle-2" - Video restyling with prompt or reference image
        """
        _warn_deprecated(model)
        try:
            return _MODELS["video"][model]  # type: ignore[return-value]
        except KeyError:
            raise ModelNotFoundError(model)

    @staticmethod
    def image(model: ImageModels) -> ImageModelDefinition:
        """
        Get an image model definition.
        Image models only support the process (sync) API.

        Available models:
            - "lucy-image-2" - Image-to-image
        """
        _warn_deprecated(model)
        try:
            return _MODELS["image"][model]  # type: ignore[return-value]
        except KeyError:
            raise ModelNotFoundError(model)


models = Models()
