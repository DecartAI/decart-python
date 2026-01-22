from typing import Literal, Optional, List, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .errors import ModelNotFoundError
from .types import FileInput, MotionTrajectoryInput


RealTimeModels = Literal[
    "mirage", "mirage_v2", "lucy_v2v_720p_rt", "lucy_v2v_14b_rt", "avatar-live"
]
VideoModels = Literal[
    "lucy-dev-i2v",
    "lucy-fast-v2v",
    "lucy-pro-t2v",
    "lucy-pro-i2v",
    "lucy-pro-v2v",
    "lucy-pro-flf2v",
    "lucy-motion",
    "lucy-restyle-v2v",
]
ImageModels = Literal["lucy-pro-t2i", "lucy-pro-i2i"]
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


class TextToVideoInput(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    seed: Optional[int] = None
    resolution: Optional[str] = None
    orientation: Optional[str] = None


class ImageToVideoInput(DecartBaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    data: FileInput
    seed: Optional[int] = None
    resolution: Optional[str] = None


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


class FirstLastFrameInput(DecartBaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    start: FileInput
    end: FileInput
    seed: Optional[int] = None
    resolution: Optional[str] = None


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


class TextToImageInput(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    seed: Optional[int] = None
    resolution: Optional[str] = None
    orientation: Optional[str] = None


class ImageToImageInput(DecartBaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )
    data: FileInput
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
        "lucy_v2v_14b_rt": ModelDefinition(
            name="lucy_v2v_14b_rt",
            url_path="/v1/stream",
            fps=15,
            width=1280,
            height=704,
            input_schema=BaseModel,
        ),
        "avatar-live": ModelDefinition(
            name="avatar-live",
            url_path="/v1/avatar-live/stream",
            fps=25,
            width=1280,
            height=720,
            input_schema=BaseModel,
        ),
    },
    "video": {
        "lucy-dev-i2v": ModelDefinition(
            name="lucy-dev-i2v",
            url_path="/v1/generate/lucy-dev-i2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToVideoInput,
        ),
        "lucy-fast-v2v": ModelDefinition(
            name="lucy-fast-v2v",
            url_path="/v1/generate/lucy-fast-v2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoToVideoInput,
        ),
        "lucy-pro-t2v": ModelDefinition(
            name="lucy-pro-t2v",
            url_path="/v1/generate/lucy-pro-t2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=TextToVideoInput,
        ),
        "lucy-pro-i2v": ModelDefinition(
            name="lucy-pro-i2v",
            url_path="/v1/generate/lucy-pro-i2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=ImageToVideoInput,
        ),
        "lucy-pro-v2v": ModelDefinition(
            name="lucy-pro-v2v",
            url_path="/v1/generate/lucy-pro-v2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=VideoToVideoInput,
        ),
        "lucy-pro-flf2v": ModelDefinition(
            name="lucy-pro-flf2v",
            url_path="/v1/generate/lucy-pro-flf2v",
            fps=25,
            width=1280,
            height=704,
            input_schema=FirstLastFrameInput,
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
    },
    "image": {
        "lucy-pro-t2i": ModelDefinition(
            name="lucy-pro-t2i",
            url_path="/v1/generate/lucy-pro-t2i",
            fps=25,
            width=1280,
            height=704,
            input_schema=TextToImageInput,
        ),
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
            - "lucy-pro-t2v" - Text-to-video
            - "lucy-pro-i2v" - Image-to-video
            - "lucy-pro-v2v" - Video-to-video
            - "lucy-pro-flf2v" - First-last-frame-to-video
            - "lucy-dev-i2v" - Image-to-video (Dev quality)
            - "lucy-fast-v2v" - Video-to-video (Fast quality)
            - "lucy-motion" - Image-to-motion-video
            - "lucy-restyle-v2v" - Video-to-video with prompt or reference image
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
            - "lucy-pro-t2i" - Text-to-image
            - "lucy-pro-i2i" - Image-to-image
        """
        try:
            return _MODELS["image"][model]  # type: ignore[return-value]
        except KeyError:
            raise ModelNotFoundError(model)


models = Models()
