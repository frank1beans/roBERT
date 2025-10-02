"""Parser for airborne sound insulation values (Rw in dB)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

__all__ = ["SoundInsulationMatch", "parse_sound_insulation"]


@dataclass(frozen=True)
class SoundInsulationMatch:
    """Match describing a sound insulation value in decibel."""

    value: float
    raw: str
    span: Tuple[int, int]


# Examples handled:
#   Rw 40dB, Rw = 39 dB, isolamento acustico Rw ≥ 39 dB
_SOUND_PATTERN = re.compile(
    r"""
    (
        (?:isolamento|abbattimento)\s+acustico
        |potere\s+fonoisolante
        |abbattimento\s+fonoisolante
        |abbattimento\s+fonoacustico
        |(?:rw|ra|r'w)
    )
    \s*(?:\([^\)]+\)\s*)?
    (?:[:=]|[<>]=?|≥|≤|pari\s+a|di)?\s*
    (?P<value>\d+(?:[\.,]\d+)?)
    (?:\s*\([^\)]+\))?
    \s*(?:d\s?b)
    """,
    re.IGNORECASE | re.VERBOSE,
)



def parse_sound_insulation(text: str) -> Iterator[SoundInsulationMatch]:
    for match in _SOUND_PATTERN.finditer(text):
        raw = match.group(0)
        value_str = match.group("value")
        try:
            value = float(value_str.replace(",", "."))
        except (TypeError, ValueError):
            continue
        if not (5.0 <= value <= 80.0):
            continue
        yield SoundInsulationMatch(
            value=value,
            raw=raw.strip(),
            span=(match.start(), match.end()),
        )
