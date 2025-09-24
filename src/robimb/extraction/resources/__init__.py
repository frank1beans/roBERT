"""Package resources for the property extraction engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_PACK_PATH = Path(__file__).resolve().parents[4] / "pack" / "current" / "pack.json"


def default_path() -> Path:
    """Return the path to the bundled knowledge pack JSON."""
    return _PACK_PATH


def load_default() -> Dict[str, Any]:
    """Load and return the default extractors pack bundled with the project."""
    with _PACK_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    extractors = payload.get("extractors")
    if isinstance(extractors, dict):
        return extractors
    patterns = payload.get("patterns")
    if patterns is None:
        return {}
    return {"patterns": patterns, "normalizers": payload.get("normalizers", {})}


__all__ = ["default_path", "load_default"]
