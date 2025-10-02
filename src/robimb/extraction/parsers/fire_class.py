"""Parser for fire reaction classes (Euroclasses)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

__all__ = ["FireClassMatch", "parse_fire_class"]


@dataclass(frozen=True)
class FireClassMatch:
    """Match for fire reaction class."""

    value: str
    raw: str
    span: Tuple[int, int]


# Pattern for Euroclasses (e.g., "A1", "A2-s1,d0", "B-s1,d0", "Euroclasse A1")
# Requires either "euroclasse"/"classe" prefix OR smoke/droplet suffix to avoid false positives
_FIRE_CLASS_PATTERN = re.compile(
    r"""
    (?:
        # Match with prefix and word boundary after class
        (?:euroclasse|classe)(?!\s+energetica)\s+(?P<class1>A1|A2|B|C|D|E|F)\b
        |
        # Match without prefix but with smoke/droplet suffix
        \b(?P<class2>A1|A2|B|C|D|E|F)(?P<smoke>-s[1-3])
    )
    (?P<droplets>,\s*[dD][0-2])?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_fire_class(text: str) -> Iterator[FireClassMatch]:
    """Yield fire reaction class matches (Euroclasses)."""
    for match in _FIRE_CLASS_PATTERN.finditer(text):
        try:
            # Get class from either group
            main_class = (match.group("class1") or match.group("class2")).upper()
            smoke = match.group("smoke") or ""
            droplets = match.group("droplets") or ""

            # Build normalized value
            value = main_class
            if smoke:
                value += smoke.lower()
            if droplets:
                value += droplets.lower().replace(" ", "")

            yield FireClassMatch(
                value=value,
                raw=match.group(0),
                span=(match.start(), match.end()),
            )
        except (ValueError, TypeError, AttributeError):
            continue
