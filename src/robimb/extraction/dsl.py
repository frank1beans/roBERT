"""Typed primitives for the regex extraction DSL."""

from __future__ import annotations

from typing import Dict, List, Sequence, TypedDict


class PatternSpec(TypedDict, total=False):
    """Description of a regex pattern entry in the extractors pack."""

    property_id: str
    regex: Sequence[str]
    normalizers: Sequence[str]


class ExtractorsPack(TypedDict, total=False):
    """Minimal structure expected from an extractors JSON pack."""

    patterns: List[PatternSpec]
    normalizers: Dict[str, Dict[str, str]]


PatternSpecs = Sequence[PatternSpec]
"""Convenience alias used by the extraction engine."""
