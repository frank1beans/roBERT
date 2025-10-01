"""Lexical matchers for brand mentions."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

__all__ = ["BrandMatcher", "load_brand_lexicon"]


def load_brand_lexicon(path: str | Path | None = None) -> Set[str]:
    """Load the brand lexicon from the default pack or a custom path."""

    lexicon_path = Path(path or "data/properties/lexicon/brands.txt")
    if not lexicon_path.exists():
        return set()
    entries = {
        line.strip()
        for line in lexicon_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return entries


class BrandMatcher:
    """Naïve case-insensitive matcher for known brand names."""

    def __init__(self, lexicon: Optional[Iterable[str]] = None) -> None:
        self._lexicon = {item.lower(): item for item in (lexicon or load_brand_lexicon())}

    def find(self, text: str) -> List[Tuple[str, Tuple[int, int], float]]:
        """Return matches as ``(brand, span, score)`` tuples."""

        lowered = text.lower()
        results: List[Tuple[str, Tuple[int, int], float]] = []
        for key, surface in self._lexicon.items():
            start = lowered.find(key)
            if start == -1:
                continue
            end = start + len(key)
            results.append((surface, (start, end), 1.0))
        return results
