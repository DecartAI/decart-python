import warnings
import pytest
from decart import models, DecartSDKError, ModelDefinition, RealtimeSpeed
from decart.models import _MODELS, _warned_aliases


def test_canonical_realtime_models() -> None:
    model = models.realtime("lucy-restyle-2")
    assert model.name == "lucy-restyle-2"
    assert model.fps == 30
    assert model.width == 1280
    assert model.height == 704
    assert model.url_path == "/v1/stream"

    model = models.realtime("lucy-2.1")
    assert model.name == "lucy-2.1"
    assert model.fps == 30
    assert model.width == 1088
    assert model.height == 624

    model = models.realtime("lucy-2.5")
    assert model.name == "lucy-2.5"
    assert model.url_path == "/v1/stream"
    assert model.fps == 30
    assert model.width == 1280
    assert model.height == 720

    model = models.realtime("lucy-vton-3.5")
    assert model.name == "lucy-vton-3.5"
    assert model.url_path == "/v1/stream"
    assert model.fps == 30
    assert model.width == 1280
    assert model.height == 720


def test_canonical_video_models() -> None:
    model = models.video("lucy-clip")
    assert model.name == "lucy-clip"
    assert model.url_path == "/v1/jobs/lucy-clip"

    model = models.video("lucy-2.1")
    assert model.name == "lucy-2.1"
    assert model.url_path == "/v1/jobs/lucy-2.1"
    assert model.fps == 20
    assert model.width == 1088
    assert model.height == 624

    model = models.video("lucy-2.5")
    assert model.name == "lucy-2.5"
    assert model.url_path == "/v1/jobs/lucy-2.5"
    assert model.fps == 20
    assert model.width == 1280
    assert model.height == 720

    model = models.video("lucy-vton-3.5")
    assert model.name == "lucy-vton-3.5"
    assert model.url_path == "/v1/jobs/lucy-vton-3.5"
    assert model.fps == 20
    assert model.width == 1280
    assert model.height == 720

    model = models.video("lucy-restyle-2")
    assert model.name == "lucy-restyle-2"
    assert model.url_path == "/v1/jobs/lucy-restyle-2"


def test_deprecated_video_models() -> None:
    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.video("lucy-pro-v2v")
        assert model.name == "lucy-pro-v2v"
        assert len(w) == 1
        assert "lucy-clip" in str(w[0].message)

    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.video("lucy-restyle-v2v")
        assert model.name == "lucy-restyle-v2v"
        assert len(w) == 1
        assert "lucy-restyle-2" in str(w[0].message)


def test_canonical_image_models() -> None:
    model = models.image("lucy-image-2")
    assert model.name == "lucy-image-2"
    assert model.url_path == "/v1/generate/lucy-image-2"


def test_deprecated_image_models() -> None:
    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.image("lucy-pro-i2i")
        assert model.name == "lucy-pro-i2i"
        assert len(w) == 1
        assert "lucy-image-2" in str(w[0].message)


def test_deprecation_warning_only_once() -> None:
    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        models.video("lucy-pro-v2v")
        models.video("lucy-pro-v2v")
        models.video("lucy-pro-v2v")
        assert len(w) == 1


def test_latest_realtime_models() -> None:
    model = models.realtime("lucy-latest")
    assert model.name == "lucy-latest"
    assert model.url_path == "/v1/stream"
    assert model.fps == 30
    assert model.width == 1088
    assert model.height == 624

    model = models.realtime("lucy-vton-latest")
    assert model.name == "lucy-vton-latest"
    assert model.url_path == "/v1/stream"
    assert model.fps == 30
    assert model.width == 1280
    assert model.height == 720

    model = models.realtime("lucy-restyle-latest")
    assert model.name == "lucy-restyle-latest"
    assert model.url_path == "/v1/stream"
    assert model.fps == 30
    assert model.width == 1280
    assert model.height == 704


def test_latest_video_models() -> None:
    model = models.video("lucy-latest")
    assert model.name == "lucy-latest"
    assert model.url_path == "/v1/jobs/lucy-latest"
    assert model.fps == 20
    assert model.width == 1088
    assert model.height == 624

    model = models.video("lucy-vton-latest")
    assert model.name == "lucy-vton-latest"
    assert model.url_path == "/v1/jobs/lucy-vton-latest"
    assert model.fps == 20
    assert model.width == 1280
    assert model.height == 720

    model = models.video("lucy-restyle-latest")
    assert model.name == "lucy-restyle-latest"
    assert model.url_path == "/v1/jobs/lucy-restyle-latest"
    assert model.fps == 22

    model = models.video("lucy-clip-latest")
    assert model.name == "lucy-clip-latest"
    assert model.url_path == "/v1/jobs/lucy-clip-latest"
    assert model.fps == 25


def test_latest_image_models() -> None:
    model = models.image("lucy-image-latest")
    assert model.name == "lucy-image-latest"
    assert model.url_path == "/v1/generate/lucy-image-latest"


def test_latest_aliases_no_deprecation_warning() -> None:
    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        models.realtime("lucy-latest")
        models.realtime("lucy-vton-latest")
        models.realtime("lucy-restyle-latest")
        models.video("lucy-latest")
        models.video("lucy-vton-latest")
        models.video("lucy-restyle-latest")
        models.video("lucy-clip-latest")
        models.image("lucy-image-latest")
        assert len(w) == 0


FAST_REALTIME_MODELS = {"lucy-2.5", "lucy-latest", "lucy-vton-3.5", "lucy-vton-latest"}


def test_realtime_fast_speed_capability_pinned_to_lucy_2_5_and_vton_3_5() -> None:
    # Matches the JS SDK registry: only these realtime entries advertise speed="fast".
    fast_models = {
        name for name, model in _MODELS["realtime"].items() if "fast" in model.supported_speeds
    }
    assert fast_models == FAST_REALTIME_MODELS

    for name, model in _MODELS["realtime"].items():
        if name in FAST_REALTIME_MODELS:
            assert model.supported_speeds == ("fast",)
        else:
            assert model.supported_speeds == ()


def test_non_realtime_models_advertise_no_speeds() -> None:
    for surface in ("video", "image"):
        for model in _MODELS[surface].values():
            assert model.supported_speeds == ()


def test_realtime_speed_literal_is_fast_only() -> None:
    from typing import get_args

    assert get_args(RealtimeSpeed) == ("fast",)


def test_custom_model_definition_allows_arbitrary_model_names() -> None:
    model = ModelDefinition(
        name="lucy_2_rt_preview",
        url_path="/v1/stream",
        fps=20,
        width=1280,
        height=720,
    )

    assert model.name == "lucy_2_rt_preview"
    assert model.input_schema is None
    assert model.supported_speeds == ()


def test_invalid_model() -> None:
    with pytest.raises(DecartSDKError):
        models.video("invalid-model")
