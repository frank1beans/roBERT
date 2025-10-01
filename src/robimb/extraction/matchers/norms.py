"""Matcher for technical standards and regulatory references."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...config import get_settings

__all__ = [
    "StandardDefinition",
    "StandardMatch",
    "StandardMatcher",
    "load_standard_dataset",
]


@dataclass(frozen=True)
class StandardDefinition:
    """Definition of a standard including synonyms and category coverage."""

    id: str
    canonical: str
    synonyms: Tuple[str, ...]
    categories: Tuple[str, ...]
    families: Tuple[str, ...]
    title: Optional[str] = None


@dataclass(frozen=True)
class StandardMatch:
    """Match returned by :class:`StandardMatcher`."""

    value: str
    surface: str
    span: Tuple[int, int]
    score: float
    title: Optional[str]


def _normalize_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _normalize_text_with_mapping(text: str) -> Tuple[str, List[int]]:
    normalized_chars: List[str] = []
    mapping: List[int] = []
    for index, char in enumerate(text):
        for item in unicodedata.normalize("NFKD", char):
            if unicodedata.combining(item):
                continue
            normalized_chars.append(item.lower())
            mapping.append(index)
    return "".join(normalized_chars), mapping


def load_standard_dataset(path: str | Path | None = None) -> Sequence[StandardDefinition]:
    """Load the curated standards dataset from disk."""

    dataset_path = Path(path) if path is not None else get_settings().standards_lexicon
    if not dataset_path.exists():
        return ()

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    definitions: List[StandardDefinition] = []
    for entry in payload.get("standards", []):
        canonical = entry.get("canonical")
        synonyms = entry.get("synonyms", []) or []
        if not canonical:
            continue
        surfaces = tuple(dict.fromkeys([canonical, *synonyms]))
        definitions.append(
            StandardDefinition(
                id=str(entry.get("id", canonical)),
                canonical=canonical,
                synonyms=surfaces,
                categories=tuple(dict.fromkeys(entry.get("categories", []) or [])),
                families=tuple(dict.fromkeys(entry.get("families", []) or [])),
                title=entry.get("title"),
            )
        )
    return tuple(definitions)


class StandardMatcher:
    """Accent-insensitive matcher for technical standards."""

    def __init__(
        self,
        dataset: Optional[Sequence[StandardDefinition] | Dict[str, Iterable[str]]] = None,
    ) -> None:
        if dataset is None:
            self._definitions = list(load_standard_dataset())
        elif isinstance(dataset, dict):
            self._definitions = [
                StandardDefinition(
                    id=key,
                    canonical=key,
                    synonyms=tuple(dict.fromkeys([key, *surfaces])),
                    categories=tuple(),
                    families=tuple(),
                    title=None,
                )
                for key, surfaces in dataset.items()
            ]
        else:
            self._definitions = list(dataset)

        self._surface_index: List[Tuple[str, str, StandardDefinition]] = []
        for definition in self._definitions:
            for surface in definition.synonyms:
                normalized_surface = _normalize_token(surface)
                if not normalized_surface:
                    continue
                self._surface_index.append((normalized_surface, surface, definition))

        # favour longer surfaces to avoid shadowing more specific entries
        self._surface_index.sort(key=lambda item: len(item[0]), reverse=True)

    def find(
        self,
        text: str,
        *,
        category: Optional[str] = None,
    ) -> List[StandardMatch]:
        if not text:
            return []

        normalized_text, index_map = _normalize_text_with_mapping(text)
        matches: Dict[Tuple[str, int, int], StandardMatch] = {}

        for normalized_surface, _, definition in self._surface_index:
            start = normalized_text.find(normalized_surface)
            while start != -1:
                end = start + len(normalized_surface)
                if end > len(index_map):
                    break
                span = (index_map[start], index_map[end - 1] + 1)
                allowed_categories = set(definition.categories)
                is_global = "trasversale" in allowed_categories or not allowed_categories
                if category and not is_global and category not in allowed_categories:
                    start = normalized_text.find(normalized_surface, start + 1)
                    continue

                surface_text = text[span[0] : span[1]]
                canonical_normalized = _normalize_token(definition.canonical)
                surface_normalized = _normalize_token(surface_text)
                score = 1.0 if surface_normalized == canonical_normalized else 0.95
                key = (definition.canonical, span[0], span[1])
                existing = matches.get(key)
                if existing is None or existing.score < score:
                    matches[key] = StandardMatch(
                        value=definition.canonical,
                        surface=surface_text,
                        span=span,
                        score=score,
                        title=definition.title,
                    )
                start = normalized_text.find(normalized_surface, start + 1)

        return sorted(matches.values(), key=lambda item: (item.span[0], -item.score))
