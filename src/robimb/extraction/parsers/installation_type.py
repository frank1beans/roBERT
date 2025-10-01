"""Parser for installation type (e.g., 'a terra', 'a parete', 'sospeso')."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

__all__ = ["InstallationTypeMatch", "parse_installation_type"]


@dataclass(frozen=True)
class InstallationTypeMatch:
    """Match for installation type."""

    value: str
    raw: str
    span: Tuple[int, int]


# Mapping of recognized patterns to normalized values
_INSTALLATION_PATTERNS = [
    (r'\ba\s+terra\b', 'a terra'),
    (r'\bda\s+terra\b', 'a terra'),
    (r'\ba\s+parete\b', 'a parete'),
    (r'\bda\s+parete\b', 'a parete'),
    (r'\bfilo\s+parete\b', 'filo parete'),
    (r'\bsospeso\b', 'sospeso'),
    (r'\bsospesa\b', 'sospeso'),
    (r'\bda\s+appoggio\b', 'da appoggio'),
    (r'\bdi\s+appoggio\b', 'da appoggio'),
    (r'\bsoprapiano\b', 'soprapiano'),
    (r'\bsotto\s+piano\b', 'sottopiano'),
    (r'\bsottopiano\b', 'sottopiano'),
    (r'\bda\s+incasso\b', 'da incasso'),
    (r'\bincassato\b', 'da incasso'),
    (r'\bincassata\b', 'da incasso'),
]


def parse_installation_type(text: str) -> Iterator[InstallationTypeMatch]:
    """Yield installation type matches."""
    for pattern, normalized_value in _INSTALLATION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            yield InstallationTypeMatch(
                value=normalized_value,
                raw=match.group(0),
                span=(match.start(), match.end()),
            )
