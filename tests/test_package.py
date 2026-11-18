"""Placeholder test to verify pytest setup."""

from hsd import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
