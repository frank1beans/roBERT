"""Lexical matcher for materials and finishes."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ...config import get_settings

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MaterialDefinition",
    "MaterialMatch",
    "MaterialMatcher",
    "load_material_lexicon",
]


@dataclass(frozen=True)
class MaterialMatch:
    """Structure describing a material match within a text.

    ``value`` intentionally exposes the normalised identifier defined in the
    lexicon (e.g. ``acciaio_inox``) so that downstream consumers can map the
    match directly to schema enums. ``canonical`` preserves the human readable
    label for logging or UI needs.
    """

    value: str
    canonical: str
    surface: str
    span: Tuple[int, int]
    score: float


@dataclass(frozen=True)
class MaterialDefinition:
    """Material entry enriched with synonyms and optional regex."""

    id: str
    canonical: str
    synonyms: Tuple[str, ...]
    regex: Optional[str] = None


def _default_lexicon_paths() -> Tuple[Path, Path]:
    settings = get_settings()
    return settings.materials_lexicon, settings.materials_lexicon_legacy


def _normalize_token(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _normalize_text_with_mapping(text: str) -> Tuple[str, List[int]]:
    normalized_chars: List[str] = []
    mapping: List[int] = []
    for index, char in enumerate(text):
        decomposed = unicodedata.normalize("NFKD", char)
        for item in decomposed:
            if unicodedata.combining(item):
                continue
            normalized_chars.append(item.lower())
            mapping.append(index)
    return "".join(normalized_chars), mapping


def load_material_lexicon(path: str | Path | None = None) -> List[MaterialDefinition]:
    """Load materials, synonyms and regex patterns from disk."""

    json_path, legacy_path = _default_lexicon_paths()
    if path:
        candidate = Path(path)
        if candidate.suffix == ".json":
            json_path = candidate
            legacy_path = candidate
        else:
            legacy_path = candidate
            json_path = candidate.with_suffix(".json")

    definitions: List[MaterialDefinition] = []

    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for entry in payload.get("materials", []):
            canonical = entry.get("canonical")
            if not canonical:
                continue
            synonyms = entry.get("synonyms", []) or []
            surfaces = tuple(dict.fromkeys([canonical, *synonyms]))
            definitions.append(
                MaterialDefinition(
                    id=entry.get("id", canonical),
                    canonical=canonical,
                    synonyms=surfaces,
                    regex=entry.get("regex"),
                )
            )
        return definitions

    if legacy_path.exists():
        mapping: Dict[str, List[str]] = {}
        for line in legacy_path.read_text(encoding="utf-8").splitlines():
            normalized = line.strip()
            if not normalized or normalized.startswith("#"):
                continue
            tokens = [token.strip() for token in normalized.split(";") if token.strip()]
            if not tokens:
                continue
            canonical, *synonyms = tokens
            mapping[canonical] = [canonical, *synonyms]
        for canonical, surfaces in mapping.items():
            definitions.append(
                MaterialDefinition(
                    id=canonical,
                    canonical=canonical,
                    synonyms=tuple(dict.fromkeys(surfaces)),
                    regex=None,
                )
            )
    return definitions


class MaterialMatcher:
    """Detect mentions of known materials using lexical and regex cues."""

    def __init__(
        self,
        lexicon: Optional[Sequence[MaterialDefinition] | Dict[str, Sequence[str]]] = None,
    ) -> None:
        if lexicon is None:
            self._definitions = load_material_lexicon()
        elif isinstance(lexicon, dict):
            self._definitions = [
                MaterialDefinition(
                    id=canonical,
                    canonical=canonical,
                    synonyms=tuple(dict.fromkeys(surfaces)),
                    regex=None,
                )
                for canonical, surfaces in lexicon.items()
            ]
        else:
            self._definitions = list(lexicon)

        self._synonym_index: List[Tuple[str, str, MaterialDefinition]] = []
        self._regex_patterns: List[Tuple[re.Pattern[str], MaterialDefinition]] = []

        for definition in self._definitions:
            for surface in definition.synonyms:
                normalized_surface = _normalize_token(surface)
                if not normalized_surface:
                    continue
                self._synonym_index.append((normalized_surface, surface, definition))

            if definition.regex:
                try:
                    pattern = re.compile(definition.regex, re.IGNORECASE | re.UNICODE)
                except re.error as exc:  # pragma: no cover - defensive
                    LOGGER.warning("invalid material regex", extra={"id": definition.id, "error": str(exc)})
                    continue
                self._regex_patterns.append((pattern, definition))

        # Sort longer surfaces first to favour specific matches when spans overlap
        self._synonym_index.sort(key=lambda item: len(item[0]), reverse=True)

    def _merge_match(
        self,
        accumulator: Dict[Tuple[str, int, int], MaterialMatch],
        definition: MaterialDefinition,
        span: Tuple[int, int],
        surface: str,
        score: float,
    ) -> None:
        key = (definition.canonical, span[0], span[1])
        existing = accumulator.get(key)
        if existing is None or existing.score < score:
            accumulator[key] = MaterialMatch(
                value=definition.id,
                canonical=definition.canonical,
                surface=surface,
                span=span,
                score=score,
            )

    def find(self, text: str) -> List[MaterialMatch]:
        normalized_text, index_map = _normalize_text_with_mapping(text)
        matches: Dict[Tuple[str, int, int], MaterialMatch] = {}

        for normalized_surface, original_surface, definition in self._synonym_index:
            start = normalized_text.find(normalized_surface)
            while start != -1:
                end = start + len(normalized_surface)
                if start >= len(index_map) or end - 1 >= len(index_map):
                    break
                span = (index_map[start], index_map[end - 1] + 1)
                surface_text = text[span[0] : span[1]]
                canonical_normalized = _normalize_token(definition.canonical)
                surface_normalized = _normalize_token(surface_text)
                score = 1.0 if surface_normalized == canonical_normalized else 0.95
                self._merge_match(matches, definition, span, surface_text, score)
                start = normalized_text.find(normalized_surface, start + 1)

        for pattern, definition in self._regex_patterns:
            for match in pattern.finditer(text):
                span = match.span()
                surface_text = match.group(0)
                self._merge_match(matches, definition, span, surface_text, 0.9)

        return list(matches.values())


__all__ = [
    "MaterialDefinition",
    "MaterialMatch",
    "MaterialMatcher",
    "load_material_lexicon",
]
