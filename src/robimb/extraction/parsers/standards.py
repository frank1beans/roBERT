"""Parser detecting technical standards codes such as UNI EN ISO."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

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
    lexicon_path = Path(path or "data/properties/lexicon/standards_prefixes.json")
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
        description = prefixes.get(prefix)
        yield StandardMatch(prefix=prefix, code=code, year=year, span=(match.start(), match.end()), description=description)
