from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from .errors import create_model_not_found_error
from .types import FileInput


RealTimeModels = Literal["mirage", "lucy_v2v_720p_rt"]
VideoModels = Literal[
    "lucy-dev-i2v",
    "lucy-dev-v2v",
    "lucy-pro-t2v",
    "lucy-pro-i2v",
    "lucy-pro-v2v",
    "lucy-pro-flf2v",
]
ImageModels = Literal["lucy-pro-t2i", "lucy-pro-i2i"]
Model = Literal[RealTimeModels, VideoModels, ImageModels]


class ModelDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    url_path: str
    fps: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    input_schema: type[BaseModel]


class TextToVideoInput(BaseModel):
    prompt: str = Field(..., min_length=1)
    seed: int | None = None
    resolution: str | None = None
    orientation: str | None = None


class ImageToVideoInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt: str = Field(..., min_length=1)
    data: FileInput
    seed: int | None = None
    resolution: str | None = None


class VideoToVideoInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt: str = Field(..., min_length=1)
    data: FileInput
    seed: int | None = None
    resolution: str | None = None
    enhance_prompt: bool | None = None
    num_inference_steps: int | None = None


class FirstLastFrameInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt: str = Field(..., min_length=1)
    start: FileInput
    end: FileInput
    seed: int | None = None
    resolution: str | None = None


class TextToImageInput(BaseModel):
    prompt: str = Field(..., min_length=1)
    seed: int | None = None
    resolution: str | None = None
    orientation: str | None = None


class ImageToImageInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    prompt: str = Field(..., min_length=1)
    data: FileInput
    seed: int | None = None
    resolution: str | None = None
    enhance_prompt: bool | None = None


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
        "lucy_v2v_720p_rt": ModelDefinition(
            name="lucy_v2v_720p_rt",
            url_path="/v1/stream",
            fps=25,
            width=1280,
            height=704,
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
        "lucy-dev-v2v": ModelDefinition(
            name="lucy-dev-v2v",
            url_path="/v1/generate/lucy-dev-v2v",
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
    def realtime(model: RealTimeModels) -> ModelDefinition:
        try:
            return _MODELS["realtime"][model]
        except KeyError:
            raise create_model_not_found_error(model)

    @staticmethod
    def video(model: VideoModels) -> ModelDefinition:
        try:
            return _MODELS["video"][model]
        except KeyError:
            raise create_model_not_found_error(model)

    @staticmethod
    def image(model: ImageModels) -> ModelDefinition:
        try:
            return _MODELS["image"][model]
        except KeyError:
            raise create_model_not_found_error(model)


models = Models()
