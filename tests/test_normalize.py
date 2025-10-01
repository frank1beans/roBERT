import math

from robimb.extraction.normalize import (
    normalize_boolean,
    normalize_confidence,
    normalize_dimension_mm,
    normalize_string,
)


def test_normalize_string_collapses_spaces() -> None:
    assert normalize_string("  Ciao   mondo  ") == "Ciao mondo"


def test_normalize_boolean_handles_variants() -> None:
    assert normalize_boolean("Sì") is True
    assert normalize_boolean("No") is False
    assert normalize_boolean("forse") is None


def test_normalize_dimension_mm_rounds_and_pads() -> None:
    result = normalize_dimension_mm([123.456, 78])
    assert result == (123.5, 78.0, None)


def test_normalize_confidence_clamps_values() -> None:
    assert normalize_confidence(1.5) == 1.0
    assert normalize_confidence(-0.2) == 0.0
    assert normalize_confidence(None) == 0.0
    assert math.isclose(normalize_confidence(0.42), 0.42)
