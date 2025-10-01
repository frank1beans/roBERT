"""Normalization helpers for property extraction outputs."""
from __future__ import annotations

import math
import unicodedata
from typing import Iterable, Optional

__all__ = [
    "normalize_string",
    "normalize_boolean",
    "normalize_dimension_mm",
    "normalize_confidence",
]


def normalize_string(value: str) -> str:
    """Trim and collapse spaces while applying Unicode normalisation."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(normalized.split())
    return normalized.strip()


def normalize_boolean(value: str) -> Optional[bool]:
    """Convert Italian yes/no markers to boolean values."""

    lowered = normalize_string(value).lower()
    if lowered in {"si", "sì", "true", "vero", "presente"}:
        return True
    if lowered in {"no", "false", "assente"}:
        return False
    return None


def normalize_dimension_mm(values: Iterable[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Round millimetre dimensions to one decimal and pad missing axes with ``None``."""

    rounded = []
    for val in values:
        if val is None:
            rounded.append(None)
            continue
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            rounded.append(None)
            continue
        if math.isnan(numeric):
            rounded.append(None)
        else:
            rounded.append(round(numeric, 1))
    while len(rounded) < 3:
        rounded.append(None)
    return tuple(rounded[:3])  # type: ignore[return-value]


def normalize_confidence(value: float | None) -> float:
    """Clamp confidence scores between 0 and 1."""

    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))
