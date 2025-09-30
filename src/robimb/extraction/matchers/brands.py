"""Minimal lexical matcher for brand mentions."""
from __future__ import annotations

from typing import List, Optional

__all__ = ["BrandMatcher"]


class BrandMatcher:
    """Naïve case-insensitive matcher for known brand names."""

    def __init__(self, lexicon: Optional[set[str]] = None) -> None:
        self._lexicon = {item.lower(): item for item in (lexicon or set())}

    def find(self, text: str) -> list[tuple[str, tuple[int, int], float]]:
        """Return matches as ``(brand, span, score)`` tuples."""

        lowered = text.lower()
        results: List[tuple[str, tuple[int, int], float]] = []
        for key, surface in self._lexicon.items():
            start = lowered.find(key)
            if start == -1:
                continue
            end = start + len(key)
            results.append((surface, (start, end), 1.0))
        return results
