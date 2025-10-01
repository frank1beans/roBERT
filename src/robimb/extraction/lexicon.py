"""High-level loaders for knowledge pack lexicon resources."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..config import get_settings

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


def _group_standards_by_category(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Any] = {}
    transversal: List[NormEntry] = []

    for entry in entries:
        canonical = entry.get("canonical")
        if not canonical:
            continue
        title = entry.get("title")
        record: NormEntry = {"code": canonical, "title": title}
        categories = list(entry.get("categories", []) or [])
        families = list(entry.get("families", []) or [])

        if "trasversale" in categories:
            transversal.append(dict(record))

        for category in categories:
            if category == "trasversale":
                continue
            bucket = grouped.setdefault(category, {})
            families_iter = families or ["varie"]
            for family in families_iter:
                values = bucket.setdefault(family, [])
                values.append(dict(record))

    for bucket in grouped.values():
        for family, items in bucket.items():
            items.sort(key=lambda item: item.get("code", ""))

    transversal.sort(key=lambda item: item.get("code", ""))
    grouped["normative_trasversali"] = transversal
    return grouped


def load_norms_by_category(path: str | Path | None = None) -> Dict[str, Any]:
    """Return the catalogue of reference standards grouped by category."""

    settings = get_settings()
    default_new = settings.standards_lexicon
    candidate = Path(path) if path else default_new

    if candidate.exists():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "standards" in payload:
            return _group_standards_by_category(payload.get("standards", []))
        if isinstance(payload, dict):
            return payload

    legacy_path = settings.standards_by_category
    return _load_json_resource(str(legacy_path), None)


def load_producers_by_category(path: str | Path | None = None) -> Dict[str, Iterable[str]]:
    """Return the curated list of producers grouped by category."""

    settings = get_settings()
    data = _load_json_resource(str(settings.producers_by_category), path)
    return {key: list(map(str, values)) for key, values in data.items()}
