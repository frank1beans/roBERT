"""Canonicalise measurement units used in property extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

__all__ = ["UnitMatch", "normalize_unit", "scan_units"]

_CANONICAL_UNITS = {
    "mm": {
        "mm",
        "millimetro",
        "millimetri",
        "millimeter",
        "millimeters",
        "㎜",
    },
    "cm": {
        "cm",
        "centimetro",
        "centimetri",
        "㎝",
    },
    "m": {
        "m",
        "metro",
        "metri",
    },
    "m2": {
        "m2",
        "m^2",
        "m²",
        "mq",
        "metroquadro",
        "metriquadrati",
        "metri_quadrati",
    },
    "m3": {
        "m3",
        "m^3",
        "m³",
        "mc",
        "metricubi",
        "metri_cubi",
    },
    "kn/m2": {
        "kn/m2",
        "kn/mq",
        "knm2",
        "knmq",
        "knm^2",
        "kn/m²",
        "kilonewton/m2",
        "kilonewton/mq",
    },
    "kg/m2": {
        "kg/m2",
        "kg/mq",
        "kgm2",
        "kgmq",
        "kg/m²",
    },
    "%": {
        "%",
        "percento",
        "percentuale",
    },
    "db": {
        "db",
        "dB",
        "decibel",
    },
}

_SUPERSCRIPTS = str.maketrans({"²": "2", "³": "3", "㎜": "mm", "㎝": "cm"})

_UNIT_PATTERN = re.compile(
    r"\b(?:(?:k|m|c)?(?:n|g)?/?m[²2]?|m³|m3|mq|mc|mm|cm|kn/mq|kg/mq|percento|percentuale|db|dB|decibel|㎜|㎝)\b|%",
    flags=re.IGNORECASE,
)


def _sanitize(token: str) -> str:
    cleaned = token.translate(_SUPERSCRIPTS)
    cleaned = cleaned.replace("/ ", "/")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.lower()


def normalize_unit(token: Optional[str]) -> Optional[str]:
    """Return the canonical representation of ``token`` if recognised."""

    if token is None:
        return None
    cleaned = _sanitize(token)
    for canonical, aliases in _CANONICAL_UNITS.items():
        if cleaned in {alias.lower().replace(" ", "") for alias in aliases}:
            return canonical
    return None


@dataclass(frozen=True)
class UnitMatch:
    """Unit mention located in text."""

    unit: str
    raw: str
    start: int
    end: int


def _iter_unit_tokens(text: str) -> Iterator[re.Match[str]]:
    for match in _UNIT_PATTERN.finditer(text):
        yield match


def scan_units(text: str) -> Iterable[UnitMatch]:
    """Find and normalise measurement units inside ``text``."""

    for match in _iter_unit_tokens(text):
        raw = match.group(0)
        unit = normalize_unit(raw)
        if unit:
            yield UnitMatch(unit=unit, raw=raw, start=match.start(), end=match.end())
