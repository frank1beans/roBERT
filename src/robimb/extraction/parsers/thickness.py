"""Parser for explicitly labeled thickness values."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

from .numbers import parse_number_it

__all__ = ["ThicknessMatch", "parse_thickness"]


@dataclass(frozen=True)
class ThicknessMatch:
    """A thickness value with explicit label."""

    value_mm: float
    raw: str
    span: Tuple[int, int]
    unit: str = "mm"


# Unit conversion factors to millimeters
_UNIT_TO_MM = {
    "mm": 1.0,
    "millimetri": 1.0,
    "millimetro": 1.0,
    "cm": 10.0,
    "centimetri": 10.0,
    "centimetro": 10.0,
    "m": 1000.0,
    "metro": 1000.0,
    "metri": 1000.0,
}

# Pattern for labeled thickness (e.g., "sp. 20 mm", "spessore 1,5 cm", "sp.20mm")
_THICKNESS_PATTERN = re.compile(
    r"""
    (?P<label>
        sp\.?|spessore|spessori|thickness|thick\.?
    )\s*
    (?:(?P<unit_before>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)\s*)?
    (?P<value>[\d.,]+)\s*
    (?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_thickness(text: str) -> Iterator[ThicknessMatch]:
    """Yield explicitly labeled thickness matches (values in mm)."""
    for match in _THICKNESS_PATTERN.finditer(text):
        try:
            raw_value = match.group("value")
            unit_after = match.group("unit")
            unit_before = match.group("unit_before")

            numeric_value = parse_number_it(raw_value)

            # Convert to mm
            unit = unit_after or unit_before
            if unit:
                multiplier = _UNIT_TO_MM.get(unit.lower(), 1.0)
                value_mm = numeric_value * multiplier
            else:
                # Default to mm for thickness values
                value_mm = numeric_value

            yield ThicknessMatch(
                value_mm=value_mm,
                raw=match.group(0),
                span=(match.start(), match.end()),
                unit="mm",
            )
        except (ValueError, TypeError):
            continue
