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
    model = models.realtime("avatar-live")
    assert model.name == "avatar-live"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 720
    assert model.url_path == "/v1/avatar-live/stream"


def test_video_models() -> None:
    model = models.video("lucy-pro-t2v")
    assert model.name == "lucy-pro-t2v"
    assert model.url_path == "/v1/generate/lucy-pro-t2v"

    model = models.video("lucy-pro-v2v")
    assert model.name == "lucy-pro-v2v"

    # lucy-restyle-v2v model
    model = models.video("lucy-restyle-v2v")
    assert model.name == "lucy-restyle-v2v"
    assert model.url_path == "/v1/generate/lucy-restyle-v2v"


def test_image_models() -> None:
    model = models.image("lucy-pro-t2i")
    assert model.name == "lucy-pro-t2i"
    assert model.url_path == "/v1/generate/lucy-pro-t2i"


def test_invalid_model() -> None:
    with pytest.raises(DecartSDKError):
        models.video("invalid-model")
