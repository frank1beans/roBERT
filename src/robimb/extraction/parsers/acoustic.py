"""Parser for acoustic absorption coefficient (αw values)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

__all__ = ["AcousticMatch", "parse_acoustic_coefficient"]


@dataclass(frozen=True)
class AcousticMatch:
    """Match for acoustic absorption coefficient."""

    value: float
    raw: str
    span: Tuple[int, int]


# Pattern for αw coefficient (e.g., "αw=0,95", "alpha w = 0.85")
_ACOUSTIC_PATTERN = re.compile(
    r"""
    (?:
        α\s*w
        |
        alpha\s*w
        |
        coefficiente\s+(?:di\s+)?(?:assorbimento\s+)?α\s*w
        |
        coefficiente\s+(?:di\s+)?(?:assorbimento\s+)?alpha\s*w
    )
    \s*(?:[:=]|pari\s+a)?\s*
    (?P<value>0[.,]\d+|1[.,]0+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_acoustic_coefficient(text: str) -> Iterator[AcousticMatch]:
    """Yield acoustic absorption coefficient matches (αw values)."""
    for match in _ACOUSTIC_PATTERN.finditer(text):
        try:
            raw_value = match.group("value")
            # Convert comma to dot
            numeric_value = float(raw_value.replace(",", "."))

            # Validate range [0.0, 1.0]
            if not (0.0 <= numeric_value <= 1.0):
                continue

            yield AcousticMatch(
                value=numeric_value,
                raw=match.group(0),
                span=(match.start(), match.end()),
            )
        except (ValueError, TypeError):
            continue
