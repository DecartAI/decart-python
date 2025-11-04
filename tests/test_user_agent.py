"""Tests for User-Agent header construction."""

from decart._user_agent import build_user_agent
from decart._version import __version__


def test_build_user_agent_without_integration():
    """Test User-Agent without integration parameter."""
    user_agent = build_user_agent()

    assert user_agent == f"decart-python-sdk/{__version__} lang/py"
    assert user_agent.startswith("decart-python-sdk/")
    assert "lang/py" in user_agent


def test_build_user_agent_with_integration():
    """Test User-Agent with integration parameter."""
    user_agent = build_user_agent("langchain/0.1.0")

    expected = f"decart-python-sdk/{__version__} lang/py langchain/0.1.0"
    assert user_agent == expected

    parts = user_agent.split(" ")
    assert len(parts) == 3
    assert parts[0].startswith("decart-python-sdk/")
    assert parts[1] == "lang/py"
    assert parts[2] == "langchain/0.1.0"
