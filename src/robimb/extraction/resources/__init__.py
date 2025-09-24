"""Package resources for the property extraction engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "properties"
_EXTRACTORS_BASENAME = "extractors.json"


def _detect_default_path() -> Path:
    """Return the most suitable bundled extractors JSON file."""

    candidates = [
        _DATA_DIR / "extractors_extended.json",
        _DATA_DIR / _EXTRACTORS_BASENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback to historical location for backwards compatibility
    legacy = Path(__file__).resolve().parents[4] / "pack" / "current" / "pack.json"
    if legacy.exists():
        return legacy
    return candidates[-1]


_PACK_PATH = _detect_default_path()


def default_path() -> Path:
    """Return the path to the bundled extractors JSON."""
    return _PACK_PATH


def load_default() -> Dict[str, Any]:
    """Load and return the default extractors pack bundled with the project."""

    path = default_path()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    # When pointing to the legacy pack we still need to unwrap the "extractors" section.
    if path.name == "pack.json":
        extractors = payload.get("extractors")
        if isinstance(extractors, dict):
            return extractors
        patterns = payload.get("patterns")
        if patterns is None:
            return {}
        return {"patterns": patterns, "normalizers": payload.get("normalizers", {})}

    if isinstance(payload, Mapping):
        # Modern bundles expose the extractors pack directly.
        patterns = payload.get("patterns")
        if isinstance(patterns, list):
            result: Dict[str, Any] = {"patterns": patterns}
            if isinstance(payload.get("normalizers"), Mapping):
                result["normalizers"] = dict(payload["normalizers"])
            if isinstance(payload.get("defaults"), Mapping):
                result["defaults"] = dict(payload["defaults"])
            return result
        extractors = payload.get("extractors")
        if isinstance(extractors, Mapping):
            return dict(extractors)
    return {}


__all__ = ["default_path", "load_default"]
