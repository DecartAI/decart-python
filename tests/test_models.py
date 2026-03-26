import pytest
from decart import models, DecartSDKError


def test_realtime_models() -> None:
    model = models.realtime("mirage")
    assert model.name == "mirage"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 704
    assert model.url_path == "/v1/stream"

    model = models.realtime("mirage_v2")
    assert model.name == "mirage_v2"
    assert model.fps == 22
    assert model.width == 1280
    assert model.height == 704
    assert model.url_path == "/v1/stream"

    # avatar-live model
    model = models.realtime("live_avatar")
    assert model.name == "live_avatar"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 720
    assert model.url_path == "/v1/stream"


def test_video_models() -> None:
    model = models.video("lucy-pro-v2v")
    assert model.name == "lucy-pro-v2v"
    assert model.url_path == "/v1/generate/lucy-pro-v2v"

    # lucy-restyle-v2v model
    model = models.video("lucy-restyle-v2v")
    assert model.name == "lucy-restyle-v2v"
    assert model.url_path == "/v1/generate/lucy-restyle-v2v"


def test_image_models() -> None:
    model = models.image("lucy-pro-i2i")
    assert model.name == "lucy-pro-i2i"
    assert model.url_path == "/v1/generate/lucy-pro-i2i"


def test_lucy_2_v2v_model() -> None:
    model = models.video("lucy-2-v2v")
    assert model.name == "lucy-2-v2v"
    assert model.url_path == "/v1/generate/lucy-2-v2v"
    assert model.fps == 20
    assert model.width == 1280
    assert model.height == 720


def test_invalid_model() -> None:
    with pytest.raises(DecartSDKError):
        models.video("invalid-model")
