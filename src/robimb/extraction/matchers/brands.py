"""Matchers and loaders for the brand lexicon."""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "BrandDefinition",
    "BrandDataset",
    "BrandMatcher",
    "load_brand_dataset",
]


@dataclass(frozen=True)
class BrandDefinition:
    """Definition for a single brand, including synonyms and categories."""

    id: str
    canonical: str
    synonyms: Tuple[str, ...]
    categories: Tuple[str, ...]


@dataclass(frozen=True)
class BrandDataset:
    """Container for the curated brand lexicon."""

    fallback: str
    brands: Tuple[BrandDefinition, ...]


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


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    def is_word_char(ch: str) -> bool:
        return ch.isalnum()

    if start > 0 and is_word_char(text[start - 1]):
        return False
    if end < len(text) and is_word_char(text[end]):
        return False
    return True


def load_brand_dataset(path: str | Path | None = None) -> BrandDataset:
    """Load the structured brand dataset from disk."""

    dataset_path = Path(path or "data/properties/lexicon/brands.json")
    if dataset_path.exists():
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        fallback = str(payload.get("fallback", "Generico"))
        definitions: List[BrandDefinition] = []
        for entry in payload.get("brands", []):
            canonical = entry.get("canonical")
            if not canonical:
                continue
            synonyms = entry.get("synonyms", []) or []
            categories = entry.get("categories", []) or []
            surfaces = tuple(dict.fromkeys([canonical, *synonyms]))
            definitions.append(
                BrandDefinition(
                    id=str(entry.get("id") or canonical),
                    canonical=canonical,
                    synonyms=surfaces,
                    categories=tuple(dict.fromkeys(categories)),
                )
            )
        return BrandDataset(fallback=fallback, brands=tuple(definitions))

    legacy_path = dataset_path.with_suffix(".txt")
    if legacy_path.exists():
        entries = [
            line.strip()
            for line in legacy_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        definitions = [
            BrandDefinition(
                id=entry.lower(),
                canonical=entry,
                synonyms=(entry,),
                categories=(),
            )
            for entry in entries
        ]
        return BrandDataset(fallback="Generico", brands=tuple(definitions))

    return BrandDataset(fallback="Generico", brands=())


class BrandMatcher:
    """Detect known brands with accent-insensitive matching and category filtering."""

    def __init__(
        self,
        dataset: Optional[
            BrandDataset | Sequence[BrandDefinition] | Iterable[str]
        ] = None,
    ) -> None:
        if isinstance(dataset, BrandDataset):
            brand_dataset = dataset
        elif dataset is None:
            brand_dataset = load_brand_dataset()
        else:
            items = list(dataset)
            if items and isinstance(items[0], BrandDefinition):
                brand_dataset = BrandDataset(
                    fallback="Generico", brands=tuple(items)
                )
            else:
                surfaces = [str(item) for item in items]
                brand_dataset = BrandDataset(
                    fallback="Generico",
                    brands=tuple(
                        BrandDefinition(
                            id=surface.lower(),
                            canonical=surface,
                            synonyms=(surface,),
                            categories=(),
                        )
                        for surface in surfaces
                    ),
                )

        self._definitions = list(brand_dataset.brands)
        self._fallback_value = brand_dataset.fallback

        self._brand_categories: Dict[str, Optional[set[str]]] = {}
        self._surface_index: List[Tuple[str, str, BrandDefinition]] = []

        for definition in self._definitions:
            categories = set(definition.categories) or None
            self._brand_categories[definition.canonical.lower()] = categories
            for surface in definition.synonyms:
                normalized_surface = _normalize_token(surface)
                if not normalized_surface:
                    continue
                self._surface_index.append((normalized_surface, surface, definition))

        self._surface_index.sort(key=lambda item: len(item[0]), reverse=True)

    @property
    def fallback_value(self) -> str:
        """Return the fallback brand value used when no match survives filtering."""

        return self._fallback_value

    def find(
        self,
        text: str,
        *,
        category: Optional[str] = None,
    ) -> List[Tuple[str, Tuple[int, int], float]]:
        """Return matches as ``(brand, span, score)`` tuples filtered by category."""

        if not text:
            return []

        normalized_text, index_map = _normalize_text_with_mapping(text)
        matches: Dict[Tuple[str, int, int], Tuple[str, Tuple[int, int], float]] = {}

        for normalized_surface, _, definition in self._surface_index:
            start = normalized_text.find(normalized_surface)
            while start != -1:
                end = start + len(normalized_surface)
                if end > len(index_map):
                    break
                span = (index_map[start], index_map[end - 1] + 1)
                if not _is_word_boundary(text, span[0], span[1]):
                    start = normalized_text.find(normalized_surface, start + 1)
                    continue

                allowed_categories = self._brand_categories.get(
                    definition.canonical.lower()
                )
                if category and allowed_categories is not None and category not in allowed_categories:
                    start = normalized_text.find(normalized_surface, start + 1)
                    continue

                surface_text = text[span[0] : span[1]]
                score = (
                    1.0
                    if _normalize_token(surface_text)
                    == _normalize_token(definition.canonical)
                    else 0.95
                )
                key = (definition.canonical, span[0], span[1])
                candidate = (definition.canonical, span, score)
                existing = matches.get(key)
                if existing is None or existing[2] < score:
                    matches[key] = candidate

                start = normalized_text.find(normalized_surface, start + 1)

        ordered_matches = sorted(matches.values(), key=lambda item: (item[1][0], item[1][1]))
        return ordered_matches
