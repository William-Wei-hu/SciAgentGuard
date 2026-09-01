"""Public package metadata for SciAgentGuard."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sciagentguard")
except PackageNotFoundError:  # pragma: no cover - only possible outside an installed package
    __version__ = "0.4.0.dev1"

__all__ = ["__version__"]
