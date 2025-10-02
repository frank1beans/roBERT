"""Parser for thermal transmittance values (Uw, Uf, Ug)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Tuple

__all__ = ["ThermalTransmittanceMatch", "parse_thermal_transmittance"]


@dataclass(frozen=True)
class ThermalTransmittanceMatch:
    """Match describing a transmittance value in W/m²K."""

    value: float
    label: str
    raw: str
    span: Tuple[int, int]


# Pattern examples handled:
#   Uw = 1,3 W/(m²K)
#   trasmittanza termica Uw pari a 1.40 W/m2K
#   Uf 2,21 W/m²°K
_TRANS_PATTERN = re.compile(
    r"""
    (?P<label>u[wfg]|trasmittanza\s+termica(?:\s+u[wfg])?)  # label
    [^0-9]{0,40}
    (?P<value>\d+(?:[\.,]\d+)?)
    \s*
    (?P<unit>
        w
        \s*/?\s*
        (?:
            \(\s*m(?:²|2)\s*k\s*\)
            |
            m(?:²|2)\s*k
            |
            \(\s*m(?:²|2)\s*/\s*k\s*\)
            |
            /\s*m(?:²|2)\s*k
        )
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalise_label(label: str) -> str:
    label = label.lower().strip()
    if label.startswith("trasmittanza"):
        if "uw" in label:
            return "Uw"
        if "uf" in label:
            return "Uf"
        if "ug" in label:
            return "Ug"
        return "U"
    if label.endswith("w"):
        return "Uw"
    if label.endswith("f"):
        return "Uf"
    if label.endswith("g"):
        return "Ug"
    return label.upper()


def parse_thermal_transmittance(text: str) -> Iterator[ThermalTransmittanceMatch]:
    for match in _TRANS_PATTERN.finditer(text):
        raw = match.group(0)
        value_str = match.group("value")
        try:
            value = float(value_str.replace(",", "."))
        except (TypeError, ValueError):
            continue
        if not (0.1 <= value <= 15.0):
            continue
        label = _normalise_label(match.group("label") or "U")
        yield ThermalTransmittanceMatch(
            value=value,
            label=label,
            raw=raw.strip(),
            span=(match.start(), match.end()),
        )
