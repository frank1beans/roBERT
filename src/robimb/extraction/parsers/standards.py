"""Parser detecting technical standards codes such as UNI EN ISO."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from ...config import get_settings

__all__ = ["StandardMatch", "load_standard_prefixes", "parse_standards"]

_STANDARD_PATTERN = re.compile(
    r"\b([A-Z]{2,4})(?:\s+EN)?\s*(\d{2,5})(?:-[0-9A-Z]{1,3})?(?:[:/](\d{2,4}))?\b"
)


@dataclass(frozen=True)
class StandardMatch:
    prefix: str
    code: str
    year: Optional[str]
    span: tuple[int, int]
    description: Optional[str]


def load_standard_prefixes(path: str | Path | None = None) -> Dict[str, str]:
    lexicon_path = Path(path) if path is not None else get_settings().standards_prefixes
    if not lexicon_path.exists():
        return {}
    data = json.loads(lexicon_path.read_text(encoding="utf-8"))
    return {key.upper(): value for key, value in data.items()}


def parse_standards(text: str, lexicon: Optional[Dict[str, str]] = None) -> Iterable[StandardMatch]:
    prefixes = lexicon or load_standard_prefixes()
    for match in _STANDARD_PATTERN.finditer(text):
        prefix = match.group(1).upper()
        code = match.group(2)
        year = match.group(3)

        # Skip if this looks like a BIM/product code (e.g., "PX06", "LA009", "K8227FR")
        # These are typically preceded by "cod", "art", "modello" or at the start of text
        start_pos = match.start()
        if start_pos > 0:
            lookback_start = max(0, start_pos - 20)
            lookback = text[lookback_start:start_pos].lower()
            # Check for product code indicators
            if re.search(r'\b(?:cod(?:ice)?|art(?:icolo)?|modello|tipo|mod|riferimento|rif)\.?\s*$', lookback):
                continue

        # Skip if at the very beginning of text (likely a BIM code)
        if start_pos < 10 and not re.search(r'\b(?:norma|standard|uni|en|iso|din)\b', text[:start_pos].lower()):
            continue

        # Only yield if prefix is a known standard or contains keywords
        description = prefixes.get(prefix)
        if description or prefix in {'UNI', 'EN', 'ISO', 'DIN', 'ASTM', 'BS', 'ANSI'}:
            yield StandardMatch(prefix=prefix, code=code, year=year, span=(match.start(), match.end()), description=description)
