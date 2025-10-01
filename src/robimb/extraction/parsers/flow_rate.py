"""Parser for flow rate expressions (l/min, l/s, etc.)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

from .numbers import parse_number_it

__all__ = ["FlowRateMatch", "parse_flow_rate"]


@dataclass(frozen=True)
class FlowRateMatch:
    """Normalized flow rate parsed from text."""

    value: float  # Normalized to l/min
    raw: str
    span: Tuple[int, int]
    unit: str = "l/min"


# Pattern for flow rate: "5.7 l/min", "10 litri/minuto", "3,5 l/s", etc.
_FLOW_RATE_PATTERN = re.compile(
    r"""
    (?P<value>[\d.,]+)\s*
    (?P<unit>
        l/min|l/m|litri/min|litri/minuto|l/s|l/sec|litri/s|litri/secondo|
        m³/h|mc/h|m3/h|metri\s*cubi/h|metri\s*cubi/ora|
        gpm|galloni/min
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_to_l_per_min(value: float, unit: str) -> float:
    """Convert flow rate to l/min."""
    unit_lower = unit.lower().replace(" ", "")

    # Already in l/min
    if unit_lower in {"l/min", "l/m", "litri/min", "litri/minuto"}:
        return value

    # l/s to l/min (multiply by 60)
    if unit_lower in {"l/s", "l/sec", "litri/s", "litri/secondo"}:
        return value * 60.0

    # m³/h to l/min (1 m³ = 1000 l, 1 h = 60 min)
    if unit_lower in {"m³/h", "mc/h", "m3/h", "metricubi/h", "metricubi/ora"}:
        return value * 1000.0 / 60.0

    # gpm (gallons per minute) to l/min (1 gallon ≈ 3.785 l)
    if unit_lower == "gpm" or "galloni" in unit_lower:
        return value * 3.785

    return value


def parse_flow_rate(text: str) -> Iterator[FlowRateMatch]:
    """Yield normalized flow rate matches (values in l/min)."""
    for match in _FLOW_RATE_PATTERN.finditer(text):
        try:
            raw_value = match.group("value")
            unit = match.group("unit")
            numeric_value = parse_number_it(raw_value)
            normalized_value = _normalize_to_l_per_min(numeric_value, unit)

            yield FlowRateMatch(
                value=normalized_value,
                raw=match.group(0),
                span=(match.start(), match.end()),
                unit="l/min",
            )
        except (ValueError, TypeError):
            continue
