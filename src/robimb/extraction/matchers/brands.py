"""Lexical matchers for brand mentions."""
from __future__ import annotations

import re
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
    """Optimized case-insensitive matcher for known brand names using compiled regex."""

    def __init__(self, lexicon: Optional[Iterable[str]] = None) -> None:
        brands = list(lexicon or load_brand_lexicon())
        # Sort by length descending to match longer brands first (e.g., "Knauf Italia" before "Knauf")
        brands.sort(key=len, reverse=True)
        self._brand_map = {brand.lower(): brand for brand in brands}
        # Build a single regex pattern with word boundaries
        escaped_brands = [re.escape(brand) for brand in brands]
        pattern = r'\b(' + '|'.join(escaped_brands) + r')\b'
        self._pattern = re.compile(pattern, re.IGNORECASE)

    def find(self, text: str) -> List[Tuple[str, Tuple[int, int], float]]:
        """Return matches as ``(brand, span, score)`` tuples."""

        results: List[Tuple[str, Tuple[int, int], float]] = []
        seen_spans: Set[Tuple[int, int]] = set()

        for match in self._pattern.finditer(text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            seen_spans.add(span)

            matched_text = match.group(0)
            # Preserve original casing from lexicon
            canonical_brand = self._brand_map.get(matched_text.lower(), matched_text)
            results.append((canonical_brand, span, 1.0))

        return results
