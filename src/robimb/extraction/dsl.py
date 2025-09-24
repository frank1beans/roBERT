"""Typed primitives for the regex extraction DSL."""

from __future__ import annotations

from typing import Dict, List, Sequence, TypedDict


class PatternSpec(TypedDict, total=False):
    """Description of a regex pattern entry in the extractors pack."""

    property_id: str
    regex: Sequence[str]
    normalizers: Sequence[str]
    language: str
    confidence: float
    tags: Sequence[str]
    unit: str | None
    examples: Sequence[str]
    max_matches: int
    first_wins: bool


class ExtractorsDefaults(TypedDict, total=False):
    """Optional defaults applied to all patterns."""

    normalizers: Sequence[str]
    selection_strategy: str


class ExtractorsPack(TypedDict, total=False):
    """Minimal structure expected from an extractors JSON pack."""

    patterns: List[PatternSpec]
    normalizers: Dict[str, Dict[str, str]]
    defaults: ExtractorsDefaults
    metadata: Dict[str, object]


PatternSpecs = Sequence[PatternSpec]
"""Convenience alias used by the extraction engine."""
