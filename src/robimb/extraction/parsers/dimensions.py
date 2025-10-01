"""Deterministic parsers for dimensional expressions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import zip_longest
from typing import Iterable, Iterator, List, Sequence, Tuple

from .numbers import parse_number_it
from .units import normalize_unit

__all__ = ["DimensionMatch", "parse_dimensions"]

_UNIT_FACTORS = {
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

_CROSS_PATTERN = re.compile(
    r"""
    (?P<first>[\d.,]+)\s*(?P<first_unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?
    \s*[x×X]\s*
    (?P<second>[\d.,]+)\s*(?P<second_unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?
    (?:\s*[x×X]\s*(?P<third>[\d.,]+)\s*(?P<third_unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?)?
    (?:\s*[hH]\s*(?P<height>[\d.,]+)\s*(?P<height_unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?)?
    \s*(?P<global_unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LABELLED_PATTERN = re.compile(
    r"""
    (?:(?:[LHP]\s*[:=]?)?\s*[\d.,]+\s*(?:mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?\s*){2,3}
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class DimensionMatch:
    """Normalized dimension parsed from text."""

    values_mm: Tuple[float, ...]
    raw: str
    span: Tuple[int, int]
    unit: str = "mm"


def _convert_unitless_value(value: float, raw: str, sequence_max: float) -> float:
    if value <= 10 and ("," in raw or "." in raw):
        return value * _UNIT_FACTORS["m"]
    if sequence_max <= 400:
        return value * _UNIT_FACTORS["cm"]
    return value * _UNIT_FACTORS["mm"]


def _convert(values: Sequence[str], units: Sequence[str | None], fallback_unit: str, explicit_unit: bool) -> Tuple[float, ...]:
    # Filter out values without digits before parsing
    filtered_pairs = [(v, u) for v, u in zip_longest(values, units) if any(ch.isdigit() for ch in v)]
    if not filtered_pairs:
        raise ValueError("No valid numeric values found")
    filtered_values = [v for v, _ in filtered_pairs]
    filtered_units = [u for _, u in filtered_pairs]

    numeric_values = [parse_number_it(raw_value) for raw_value in filtered_values]
    sequence_max = max(numeric_values) if numeric_values else 0.0
    units_list = list(filtered_units) + [None] * (len(filtered_values) - len(filtered_units))
    results: List[float] = []
    for numeric, raw_unit, raw_value in zip(numeric_values, units_list, filtered_values):
        if raw_unit:
            normalized = normalize_unit(raw_unit) or raw_unit
            multiplier = _UNIT_FACTORS.get(normalized)
            if multiplier is None:
                raise ValueError(f"Unsupported unit: {raw_unit}")
            results.append(numeric * multiplier)
            continue
        if explicit_unit:
            normalized = normalize_unit(fallback_unit) or fallback_unit
            multiplier = _UNIT_FACTORS.get(normalized, 1.0)
            results.append(numeric * multiplier)
            continue
        results.append(_convert_unitless_value(numeric, raw_value, sequence_max))
    return tuple(results)


def _fallback_unit(global_unit: str | None, *units: str | None) -> str:
    for candidate in (*units, global_unit):
        if not candidate:
            continue
        normalized = normalize_unit(candidate)
        if normalized:
            return normalized
    return "mm"


def _iter_cross(text: str) -> Iterator[DimensionMatch]:
    for match in _CROSS_PATTERN.finditer(text):
        raw_match = match.group(0)

        # Skip if the pattern looks like a product code (numbers without proper separators)
        # e.g., ". 23 326 " should be rejected (spaces instead of 'x')
        if 'x' not in raw_match.lower():
            # No 'x' separator found - likely not a dimension
            continue

        # Skip if preceded by a letter, digit, or punctuation (likely a product code)
        if match.start() > 0:
            # Look back up to 10 chars to check for "cod", "art", "cat", etc.
            lookback_start = max(0, match.start() - 10)
            lookback = text[lookback_start:match.start()].lower()
            # Check for product code indicators
            if re.search(r'\b(?:cod(?:ice)?|art|cat|rif|ref)\.?\s*$', lookback):
                continue
            # Check immediate previous character
            prev_char = text[match.start() - 1]
            if prev_char.isalpha() or prev_char.isdigit():
                continue

        # Skip if followed by more digits (likely a product code like 23 326 000)
        if match.end() < len(text):
            # Look ahead up to 10 characters for more digits
            lookahead = text[match.end():match.end() + 10]
            # If we find digits after whitespace, it's likely a code
            if re.search(r'^\s*\d', lookahead):
                continue

        groups = match.groupdict()
        values = [groups["first"], groups["second"]]
        units = [groups.get("first_unit"), groups.get("second_unit")]

        # Handle height parameter (h190)
        if groups.get("height"):
            # If we have "WxL h H" format, the height replaces or becomes the third dimension
            values.append(groups["height"])
            units.append(groups.get("height_unit"))
        elif groups.get("third"):
            # Standard "WxLxH" format
            values.append(groups["third"])
            units.append(groups.get("third_unit"))

        base_unit = _fallback_unit(groups.get("global_unit"), *units)
        explicit_unit = bool(groups.get("global_unit")) or any(units)
        try:
            converted = _convert(values, units, base_unit, explicit_unit)
        except ValueError:
            continue
        yield DimensionMatch(values_mm=converted, raw=match.group(0), span=(match.start(), match.end()))


def _extract_numbers_from_labelled(raw: str) -> Tuple[List[str], List[str | None]]:
    tokens = re.findall(
        r"([\d.,]+)\s*(mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?",
        raw,
        flags=re.IGNORECASE,
    )
    values: List[str] = []
    units: List[str | None] = []
    for value, unit in tokens:
        if not any(char.isdigit() for char in value):
            continue
        values.append(value)
        units.append(unit)
    return values, units


def _iter_labelled(text: str) -> Iterator[DimensionMatch]:
    for match in _LABELLED_PATTERN.finditer(text):
        raw = match.group(0)

        # Skip if preceded by product code indicators
        if match.start() > 0:
            lookback_start = max(0, match.start() - 10)
            lookback = text[lookback_start:match.start()].lower()
            if re.search(r'\b(?:cod(?:ice)?|art|cat|rif|ref)\.?\s*$', lookback):
                continue

        # Skip if followed by more digits (likely continuation of a product code)
        if match.end() < len(text):
            lookahead = text[match.end():match.end() + 10]
            if re.search(r'^\s*\d', lookahead):
                continue

        values, units = _extract_numbers_from_labelled(raw)
        if len(values) < 2:
            continue
        base_unit = _fallback_unit(None, *units)
        explicit_unit = any(units)
        try:
            converted = _convert(values, units, base_unit, explicit_unit)
        except ValueError:
            continue
        yield DimensionMatch(values_mm=converted, raw=raw, span=(match.start(), match.end()))


def parse_dimensions(text: str) -> Iterable[DimensionMatch]:
    """Yield normalized dimensions (values in millimetres)."""

    seen: set[Tuple[int, int]] = set()
    for match in _iter_cross(text):
        if match.span in seen:
            continue
        seen.add(match.span)
        yield match
    for match in _iter_labelled(text):
        if match.span in seen:
            continue
        seen.add(match.span)
        yield match
