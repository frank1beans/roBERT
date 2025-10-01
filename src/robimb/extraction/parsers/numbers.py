"""Utilities to parse Italian-formatted numbers and locate them in text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

__all__ = ["NumberSpan", "extract_numbers", "parse_number_it"]

_SIGN = r"[+-]?"
_DECIMAL_BODY = r"(?:\d{1,3}(?:[\.\s]\d{3})+|\d+)(?:[\.,]\d+)?|\d+[\.,]\d+|\d+"
_NUMBER_PATTERN = re.compile(rf"{_SIGN}(?:{_DECIMAL_BODY})")

_STRIP_CHARS = str.maketrans({"%": "", "‰": "", "€": "", "£": "", "\u00A0": " "})
_THOUSANDS_PATTERN = re.compile(r"^\d{1,3}(?:[.]\d{3})+$")


@dataclass(frozen=True)
class NumberSpan:
    """Representation of a numeric value extracted from text."""

    value: float
    raw: str
    start: int
    end: int


def _normalize_numeric_string(raw: str) -> str:
    candidate = raw.translate(_STRIP_CHARS).strip()
    candidate = re.sub(r"[^0-9,\.\-\+]+", "", candidate)
    if not candidate:
        raise ValueError(f"Cannot parse numeric value from '{raw}'")

    if not any(ch.isdigit() for ch in candidate):
        raise ValueError(f"Cannot parse numeric value from '{raw}'")

    last_comma = candidate.rfind(",")
    last_dot = candidate.rfind(".")

    decimal_sep: Optional[str]
    thousands_sep: Optional[str]
    if last_comma != -1 and last_dot != -1:
        if last_comma > last_dot:
            decimal_sep, thousands_sep = ",", "."
        else:
            decimal_sep, thousands_sep = ".", ","
    elif last_comma != -1:
        decimal_sep, thousands_sep = ",", None
    elif last_dot != -1:
        if _THOUSANDS_PATTERN.match(candidate):
            decimal_sep, thousands_sep = None, "."
        else:
            decimal_sep, thousands_sep = ".", None
    else:
        decimal_sep, thousands_sep = None, None

    if thousands_sep:
        candidate = candidate.replace(thousands_sep, "")

    if decimal_sep and decimal_sep != ".":
        candidate = candidate.replace(decimal_sep, ".")

    candidate = candidate.replace(" ", "")
    return candidate


def parse_number_it(raw: str) -> float:
    """Parse an Italian-formatted number string into a float."""

    normalized = _normalize_numeric_string(raw)
    try:
        return float(normalized)
    except ValueError as exc:  # pragma: no cover - safety guard
        raise ValueError(f"Invalid numeric value: {raw!r}") from exc


def _iter_number_matches(text: str) -> Iterator[re.Match[str]]:
    for match in _NUMBER_PATTERN.finditer(text):
        yield match


def extract_numbers(text: str) -> Iterable[NumberSpan]:
    """Extract numeric spans from ``text`` using Italian number heuristics."""

    for match in _iter_number_matches(text):
        raw = match.group(0)
        try:
            value = parse_number_it(raw)
        except ValueError:
            continue
        yield NumberSpan(value=value, raw=raw, start=match.start(), end=match.end())
