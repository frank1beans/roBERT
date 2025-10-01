"""Lexical matcher for materials and finishes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["MaterialMatch", "MaterialMatcher", "load_material_lexicon"]


@dataclass(frozen=True)
class MaterialMatch:
    """Structure describing a material match within a text."""

    value: str
    surface: str
    span: Tuple[int, int]
    score: float


def load_material_lexicon(path: str | Path | None = None) -> Dict[str, Sequence[str]]:
    """Load materials and optional synonyms from a lexicon file."""

    lexicon_path = Path(path or "data/properties/lexicon/materials.txt")
    if not lexicon_path.exists():
        return {}
    mapping: Dict[str, Sequence[str]] = {}
    for line in lexicon_path.read_text(encoding="utf-8").splitlines():
        normalized = line.strip()
        if not normalized or normalized.startswith("#"):
            continue
        tokens = [token.strip() for token in normalized.split(";") if token.strip()]
        if not tokens:
            continue
        canonical, *synonyms = tokens
        mapping[canonical] = [canonical, *synonyms]
    return mapping


class MaterialMatcher:
    """Detect mentions of known materials using substring search."""

    def __init__(self, lexicon: Optional[Dict[str, Sequence[str]]] = None) -> None:
        self._lexicon = lexicon or load_material_lexicon()
        self._lookup: Dict[str, str] = {}
        for canonical, surfaces in self._lexicon.items():
            for surface in surfaces:
                self._lookup[surface.lower()] = canonical

    def find(self, text: str) -> List[MaterialMatch]:
        lowered = text.lower()
        matches: List[MaterialMatch] = []
        for surface, canonical in self._lookup.items():
            start = lowered.find(surface)
            if start == -1:
                continue
            end = start + len(surface)
            matches.append(
                MaterialMatch(
                    value=canonical,
                    surface=text[start:end],
                    span=(start, end),
                    score=1.0 if surface == canonical.lower() else 0.8,
                )
            )
        return matches


__all__ = ["MaterialMatch", "MaterialMatcher", "load_material_lexicon"]
