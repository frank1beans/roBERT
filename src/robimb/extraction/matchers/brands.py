"""Lexical matchers for brand mentions."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

__all__ = ["BrandMatcher", "load_brand_lexicon", "load_brand_metadata"]


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


def load_brand_metadata(path: str | Path | None = None) -> Dict[str, object]:
    """Load metadata describing the categories supported by each brand."""

    metadata_path = Path(path or "data/properties/brands_by_category.json")
    if not metadata_path.exists():
        return {"fallback": "Generico", "brands": {}}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid brand metadata file: {metadata_path}") from exc
    fallback = payload.get("fallback", "Generico")
    brands = payload.get("brands", {})
    if not isinstance(brands, dict):  # pragma: no cover - defensive
        raise ValueError("Brand metadata must provide a mapping for 'brands'")
    return {"fallback": fallback, "brands": brands}


class BrandMatcher:
    """Optimized case-insensitive matcher for known brand names using compiled regex."""

    def __init__(
        self,
        lexicon: Optional[Iterable[str]] = None,
        *,
        metadata_path: str | Path | None = None,
    ) -> None:
        metadata = load_brand_metadata(metadata_path)
        brands = list(lexicon or load_brand_lexicon())
        # Sort by length descending to match longer brands first (e.g., "Knauf Italia" before "Knauf")
        brands.sort(key=len, reverse=True)
        self._brand_map = {brand.lower(): brand for brand in brands}
        metadata_brands: Dict[str, List[str]] = {
            key: value for key, value in metadata.get("brands", {}).items() if isinstance(value, list)
        }
        self._brand_categories: Dict[str, Optional[Set[str]]] = {}
        for brand in brands:
            metadata_key = brand
            categories = metadata_brands.get(metadata_key)
            if categories is None:
                categories = metadata_brands.get(metadata_key.lower())
            allowed = set(categories) if categories else None
            self._brand_categories[brand.lower()] = allowed

        self._fallback_value = str(metadata.get("fallback", "Generico"))

        # Build a single regex pattern with word boundaries
        escaped_brands = [re.escape(brand) for brand in brands]
        if escaped_brands:
            pattern = r"\b(" + "|".join(escaped_brands) + r")\b"
            self._pattern: Optional[re.Pattern[str]] = re.compile(pattern, re.IGNORECASE)
        else:  # pragma: no cover - defensive
            self._pattern = None

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

        if self._pattern is None:
            return []

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
            allowed_categories = self._brand_categories.get(canonical_brand.lower())
            if category and allowed_categories is not None and category not in allowed_categories:
                continue
            results.append((canonical_brand, span, 1.0))

        return results
