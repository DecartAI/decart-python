import warnings
from typing import Literal, Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .errors import ModelNotFoundError
from .types import FileInput, MotionTrajectoryInput


RealTimeModels = Literal[
    # Canonical names
    "lucy",
    "lucy-2",
    "lucy-2.1",
    "lucy-2.1-vton",
    "lucy-restyle",
    "lucy-restyle-2",
    "live-avatar",
    # Deprecated names
    "mirage",
    "mirage_v2",
    "lucy_v2v_720p_rt",
    "lucy_2_rt",
    "live_avatar",
]
VideoModels = Literal[
    # Canonical names
    "lucy-clip",
    "lucy-2",
    "lucy-2.1",
    "lucy-restyle-2",
    "lucy-motion",
    # Deprecated names
    "lucy-pro-v2v",
    "lucy-restyle-v2v",
    "lucy-2-v2v",
]
ImageModels = Literal[
    # Canonical names
    "lucy-image-2",
    # Deprecated names
    "lucy-pro-i2i",
]
Model = Literal[RealTimeModels, VideoModels, ImageModels]

MODEL_ALIASES: dict[str, str] = {
    # Realtime aliases
    "mirage": "lucy-restyle",
    "mirage_v2": "lucy-restyle-2",
    "lucy_v2v_720p_rt": "lucy",
    "lucy_2_rt": "lucy-2",
    "live_avatar": "live-avatar",
    # Video aliases
    "lucy-pro-v2v": "lucy-clip",
    "lucy-restyle-v2v": "lucy-restyle-2",
    "lucy-2-v2v": "lucy-2",
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
    input_schema: type[BaseModel]


# Type aliases for model definitions that support specific APIs
ImageModelDefinition = ModelDefinition[ImageModels]
"""Type alias for model definitions that support synchronous processing (process API)."""

VideoModelDefinition = ModelDefinition[VideoModels]
"""Type alias for model definitions that support queue processing (queue API)."""

RealTimeModelDefinition = ModelDefinition[RealTimeModels]
"""Type alias for model definitions that support realtime streaming."""


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


class ImageToMotionVideoInput(DecartBaseModel):
    data: FileInput
    trajectory: List[MotionTrajectoryInput] = Field(..., min_length=2, max_length=1000)
    seed: Optional[int] = None
    resolution: Optional[str] = None


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
    """Input for lucy-2-v2v model.

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
        "lucy": ModelDefinition(
            name="lucy",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "lucy-2": ModelDefinition(
            name="lucy-2",
            url_path="/v1/stream",
            fps=20,
            width=1280,
            height=720,
            input_schema=BaseModel,
        ),
        "lucy-2.1": ModelDefinition(
            name="lucy-2.1",
            url_path="/v1/stream",
            fps=20,
            width=1088,
            height=624,
            input_schema=BaseModel,
        ),
        "lucy-2.1-vton": ModelDefinition(
            name="lucy-2.1-vton",
            url_path="/v1/stream",
            fps=20,
            width=1088,
            height=624,
            input_schema=BaseModel,
        ),
        "lucy-restyle": ModelDefinition(
            name="lucy-restyle",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "lucy-restyle-2": ModelDefinition(
            name="lucy-restyle-2",
            url_path="/v1/stream",
            fps=22,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "live-avatar": ModelDefinition(
            name="live-avatar",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=720,
            input_schema=BaseModel,
        ),
        # Deprecated names
        "mirage": ModelDefinition(
            name="mirage",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "mirage_v2": ModelDefinition(
            name="mirage_v2",
            url_path="/v1/stream",
            fps=22,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "lucy_v2v_720p_rt": ModelDefinition(
            name="lucy_v2v_720p_rt",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "lucy_2_rt": ModelDefinition(
            name="lucy_2_rt",
            url_path="/v1/stream",
            fps=20,
            width=1280,
            height=720,
            input_schema=BaseModel,
        ),
        "live_avatar": ModelDefinition(
            name="live_avatar",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=720,
            input_schema=BaseModel,
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
        "lucy-2": ModelDefinition(
            name="lucy-2",
            url_path="/v1/jobs/lucy-2",
            fps=20,
            width=1280,
            height=720,
            input_schema=VideoEdit2Input,
        ),
        "lucy-2.1": ModelDefinition(
            name="lucy-2.1",
            url_path="/v1/jobs/lucy-2.1",
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
        "lucy-motion": ModelDefinition(
            name="lucy-motion",
            url_path="/v1/jobs/lucy-motion",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToMotionVideoInput,
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
        "lucy-2-v2v": ModelDefinition(
            name="lucy-2-v2v",
            url_path="/v1/jobs/lucy-2-v2v",
            fps=20,
            width=1280,
            height=720,
            input_schema=VideoEdit2Input,
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
            - "lucy-2" - Video editing with reference image support
            - "lucy-2.1" - Video editing (newer, higher quality)
            - "lucy-restyle-2" - Video restyling with prompt or reference image
            - "lucy-motion" - Image-to-motion-video
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
