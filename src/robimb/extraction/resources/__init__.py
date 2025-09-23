"""Package resources for the property extraction engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_RESOURCE_DIR = Path(__file__).resolve().parent
_DEFAULT_FILENAME = "extractors.json"
_DEFAULT_PATH = _RESOURCE_DIR / _DEFAULT_FILENAME


def default_path() -> Path:
    """Return the path to the default extractors pack JSON."""
    return _DEFAULT_PATH


def load_default() -> Dict[str, Any]:
    """Load and return the default extractors pack bundled with the package."""
    with _DEFAULT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["default_path", "load_default"]
