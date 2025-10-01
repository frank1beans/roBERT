"""Parser for explicitly labeled dimensions (e.g., 'lunghezza 60 cm', 'larghezza 80 mm')."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

from .numbers import parse_number_it

__all__ = ["LabeledDimensionMatch", "parse_labeled_dimensions"]


@dataclass(frozen=True)
class LabeledDimensionMatch:
    """A dimension with an explicit label."""

    label: str  # e.g., "lunghezza", "larghezza", "altezza", "profondità"
    value_mm: float  # Always in millimeters
    raw: str
    span: Tuple[int, int]
    unit: str = "mm"


# Unit conversion factors to millimeters
_UNIT_TO_MM: Dict[str, float] = {
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

# Pattern for labeled dimensions
# Includes abbreviated forms like "L.", "H.", "P."
_LABELED_DIM_PATTERN = re.compile(
    r"""
    (?P<label>
        lunghezza|larghezza|altezza|profondità|profondita|spessore|
        diametro|raggio|length|width|height|depth|thickness|diameter|radius|
        l\.|h\.|p\.|d\.
    )\s*
    (?:di\s+|:|=|\s+)?
    (?P<value>[\d.,]+)\s*
    (?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_labeled_dimensions(text: str) -> Iterator[LabeledDimensionMatch]:
    """Yield explicitly labeled dimension matches (values in mm)."""
    # Mapping of abbreviated labels to full forms
    _LABEL_NORMALIZE = {
        'l.': 'lunghezza',
        'h.': 'altezza',
        'p.': 'profondità',
        'd.': 'diametro',
    }

    for match in _LABELED_DIM_PATTERN.finditer(text):
        try:
            raw_label = match.group("label").lower()
            raw_value = match.group("value")
            unit = match.group("unit").lower()

            # Normalize abbreviated labels
            label = _LABEL_NORMALIZE.get(raw_label, raw_label)

            numeric_value = parse_number_it(raw_value)
            multiplier = _UNIT_TO_MM.get(unit, 1.0)
            value_mm = numeric_value * multiplier

            yield LabeledDimensionMatch(
                label=label,
                value_mm=value_mm,
                raw=match.group(0),
                span=(match.start(), match.end()),
                unit="mm",
            )
        except (ValueError, TypeError):
            continue
