from typing import Literal, Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .errors import ModelNotFoundError
from .types import FileInput, MotionTrajectoryInput


RealTimeModels = Literal["mirage", "mirage_v2", "lucy_v2v_720p_rt", "lucy_2_rt", "live_avatar"]
VideoModels = Literal[
    "lucy-pro-v2v",
    "lucy-motion",
    "lucy-restyle-v2v",
    "lucy-2-v2v",
]
ImageModels = Literal["lucy-pro-i2i"]
Model = Literal[RealTimeModels, VideoModels, ImageModels]

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
        "lucy-pro-v2v": ModelDefinition(
            name="lucy-pro-v2v",
            url_path="/v1/generate/lucy-pro-v2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoToVideoInput,
        ),
        "lucy-motion": ModelDefinition(
            name="lucy-motion",
            url_path="/v1/generate/lucy-motion",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToMotionVideoInput,
        ),
        "lucy-restyle-v2v": ModelDefinition(
            name="lucy-restyle-v2v",
            url_path="/v1/generate/lucy-restyle-v2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoRestyleInput,
        ),
        "lucy-2-v2v": ModelDefinition(
            name="lucy-2-v2v",
            url_path="/v1/generate/lucy-2-v2v",
            fps=20,
            width=1280,
            height=720,
            input_schema=VideoEdit2Input,
        ),
    },
    "image": {
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
            - "lucy-pro-v2v" - Video-to-video
            - "lucy-motion" - Image-to-motion-video
            - "lucy-restyle-v2v" - Video-to-video with prompt or reference image
            - "lucy-2-v2v" - Video-to-video editing (long-form, 720p)
        """
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
            - "lucy-pro-i2i" - Image-to-image
        """
        try:
            return _MODELS["image"][model]  # type: ignore[return-value]
        except KeyError:
            raise ModelNotFoundError(model)


models = Models()
