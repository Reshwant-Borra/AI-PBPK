"""AI-PBPK package initializer.

Exposes version and simple utility to get repository paths.
"""

from importlib.metadata import version as _version
from pathlib import Path


__all__ = [
	"get_repo_root",
	"__version__",
]


def get_repo_root() -> Path:
	return Path(__file__).resolve().parent.parent


try:
	__version__ = _version("ai-pbpk-mpc")
except Exception:
	__version__ = "0.0.0"


