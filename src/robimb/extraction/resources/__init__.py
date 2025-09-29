"""Package resources for the property extraction engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_DATA_DIR = _REPO_ROOT / "data" / "properties"
_DEFAULT_PACK_ROOT = _REPO_ROOT / "pack"
_EXTRACTORS_FILENAMES = (
    "extractors.json",
    "extractors_extended.json",
)


def _resolve_from_directory(base: Path) -> Optional[Path]:
    for name in _EXTRACTORS_FILENAMES:
        candidate = base / name
        if candidate.exists():
            return candidate
    pack_json = base / "pack.json"
    if pack_json.exists():
        return pack_json
    return None


def _detect_default_path() -> Path:
    """Return the most suitable bundled extractors JSON file."""

    env_override = os.getenv("ROBIMB_EXTRACTORS_PACK")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate

    env_current = os.getenv("ROBIMB_PACK_CURRENT")
    if env_current:
        candidate = Path(env_current)
        if candidate.is_dir():
            resolved = _resolve_from_directory(candidate)
            if resolved is not None:
                return resolved
        if candidate.exists():
            return candidate

    current_dir = _DEFAULT_PACK_ROOT / "current"
    if current_dir.exists():
        if current_dir.is_dir():
            resolved = _resolve_from_directory(current_dir)
            if resolved is not None:
                return resolved
        else:
            return current_dir

    if _DEFAULT_PACK_ROOT.exists():
        versioned = sorted(_DEFAULT_PACK_ROOT.glob("v*/extractors.json"), reverse=True)
        for candidate in versioned:
            if candidate.exists():
                return candidate

    legacy_candidates = [
        _LEGACY_DATA_DIR / "extractors_extended.json",
        _LEGACY_DATA_DIR / "extractors.json",
    ]
    for candidate in legacy_candidates:
        if candidate.exists():
            return candidate

    # As a last resort, point to the last legacy candidate (which may not exist)
    return legacy_candidates[-1]


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
