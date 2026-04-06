import warnings
import pytest
from decart import models, DecartSDKError
from decart.models import _warned_aliases


def test_canonical_realtime_models() -> None:
    model = models.realtime("lucy-restyle")
    assert model.name == "lucy-restyle"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 704
    assert model.url_path == "/v1/stream"

    model = models.realtime("lucy-restyle-2")
    assert model.name == "lucy-restyle-2"
    assert model.fps == 22
    assert model.width == 1280
    assert model.height == 704

    model = models.realtime("lucy")
    assert model.name == "lucy"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 704

    model = models.realtime("lucy-2")
    assert model.name == "lucy-2"
    assert model.fps == 20
    assert model.width == 1280
    assert model.height == 720

    model = models.realtime("lucy-2.1")
    assert model.name == "lucy-2.1"
    assert model.fps == 20
    assert model.width == 1088
    assert model.height == 624

    model = models.realtime("lucy-2.1-vton")
    assert model.name == "lucy-2.1-vton"
    assert model.fps == 20
    assert model.width == 1088
    assert model.height == 624

    model = models.realtime("live-avatar")
    assert model.name == "live-avatar"
    assert model.fps == 25
    assert model.width == 1280
    assert model.height == 720


def test_deprecated_realtime_models() -> None:
    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.realtime("mirage")
        assert model.name == "mirage"
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()
        assert "lucy-restyle" in str(w[0].message)

    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.realtime("mirage_v2")
        assert model.name == "mirage_v2"
        assert len(w) == 1
        assert "lucy-restyle-2" in str(w[0].message)

    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.realtime("live_avatar")
        assert model.name == "live_avatar"
        assert len(w) == 1
        assert "live-avatar" in str(w[0].message)


def test_canonical_video_models() -> None:
    model = models.video("lucy-clip")
    assert model.name == "lucy-clip"
    assert model.url_path == "/v1/jobs/lucy-clip"

    model = models.video("lucy-2")
    assert model.name == "lucy-2"
    assert model.url_path == "/v1/jobs/lucy-2"
    assert model.fps == 20
    assert model.width == 1280
    assert model.height == 720

    model = models.video("lucy-2.1")
    assert model.name == "lucy-2.1"
    assert model.url_path == "/v1/jobs/lucy-2.1"
    assert model.fps == 20
    assert model.width == 1088
    assert model.height == 624

    model = models.video("lucy-restyle-2")
    assert model.name == "lucy-restyle-2"
    assert model.url_path == "/v1/jobs/lucy-restyle-2"

    model = models.video("lucy-motion")
    assert model.name == "lucy-motion"
    assert model.url_path == "/v1/jobs/lucy-motion"


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

    _warned_aliases.clear()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = models.video("lucy-2-v2v")
        assert model.name == "lucy-2-v2v"
        assert len(w) == 1
        assert '"lucy-2"' in str(w[0].message)


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
        models.realtime("mirage")
        models.realtime("mirage")
        models.realtime("mirage")
        assert len(w) == 1


def test_invalid_model() -> None:
    with pytest.raises(DecartSDKError):
        models.video("invalid-model")
