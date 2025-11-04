"""SDK version information."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("decart")
except PackageNotFoundError:
    # Development version when package is not installed
    __version__ = "0.0.0-dev"
