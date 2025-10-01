"""High-level loaders for knowledge pack lexicon resources."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

__all__ = [
    "CategoryNorms",
    "CategoryProducers",
    "NormEntry",
    "load_norms_by_category",
    "load_producers_by_category",
]

NormEntry = Dict[str, str]
CategoryNorms = Dict[str, Any]
CategoryProducers = Dict[str, Iterable[str]]


def _load_json_resource(default_path: str, path: str | Path | None) -> Dict[str, Any]:
    lexicon_path = Path(path or default_path)
    if not lexicon_path.exists():
        return {}
    data = json.loads(lexicon_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary at {lexicon_path}, got {type(data)!r}")
    return data


def load_norms_by_category(path: str | Path | None = None) -> Dict[str, Any]:
    """Return the catalogue of reference standards grouped by category."""

    return _load_json_resource("data/properties/lexicon/norms_by_category.json", path)


def load_producers_by_category(path: str | Path | None = None) -> Dict[str, Iterable[str]]:
    """Return the curated list of producers grouped by category."""

    data = _load_json_resource("data/properties/lexicon/producers_by_category.json", path)
    return {key: list(map(str, values)) for key, values in data.items()}
