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


# Mapping of recognized patterns to normalized enum values
_INSTALLATION_PATTERNS = [
    (r'\b(?:a|da|su)\s+pavimento\b', 'a_pavimento'),
    (r'\b(?:scarico|scarichi)\s+a\s+pavimento\b', 'a_pavimento'),
    (r'\bpavimento\s+ribassat[oa]\b', 'a_pavimento'),
    (r'\ba\s+terra\b', 'a_pavimento'),
    (r'\bda\s+terra\b', 'a_pavimento'),
    (r'\bda\s+appoggio\b', 'a_pavimento'),
    (r'\bdi\s+appoggio\b', 'a_pavimento'),
    (r'\ba\s+parete\b', 'a_parete'),
    (r'\bda\s+parete\b', 'a_parete'),
    (r'\bfilo\s+parete\b', 'a_parete'),
    (r'\ba\s+muro\b', 'a_parete'),
    (r'\bda\s+fissare(?:\s+(?:a|su)\s+(?:parete|muro|porta))?\b', 'a_parete'),
    (r'\bsu\s+porta\b', 'a_parete'),
    (r'\bsospes[oa]\b', 'sospesa'),
    (r'\ba\s+soffitt[oa]\b', 'sospesa'),
    (r'\bda\s+incasso\b', 'incasso'),
    (r'\bincassat[oa]\b', 'incasso'),
    (r'\bsotto\s+piano\b', 'incasso'),
    (r'\bsottopiano\b', 'incasso'),
    (r'\bsoprapiano\b', 'incasso'),
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
