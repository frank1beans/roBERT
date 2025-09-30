import pytest

from robimb.extraction.parsers.units import UnitMatch, normalize_unit, scan_units


@pytest.mark.parametrize(
    "token, expected",
    [
        ("mm", "mm"),
        ("Millimetri", "mm"),
        ("㎜", "mm"),
        ("cm", "cm"),
        ("Centimetro", "cm"),
        ("m", "m"),
        ("Metri", "m"),
        ("m²", "m2"),
        ("m^2", "m2"),
        ("mq", "m2"),
        ("m³", "m3"),
        ("mc", "m3"),
        ("kN/m²", "kn/m2"),
        ("kN/mq", "kn/m2"),
        ("kg/m²", "kg/m2"),
        ("percentuale", "%"),
        ("dB", "db"),
        (None, None),
        ("unknown", None),
    ],
)
def test_normalize_unit(token, expected) -> None:
    assert normalize_unit(token) == expected


def test_scan_units_detects_all() -> None:
    text = "Pannello 600x600 mm, spessore 30 mm, peso 45 kg/m², isolamento 35 dB, carico 4 kN/mq"
    matches = list(scan_units(text))
    assert [match.unit for match in matches] == ["mm", "mm", "kg/m2", "db", "kn/m2"]
    for match in matches:
        assert isinstance(match, UnitMatch)
        assert text[match.start:match.end] == match.raw


def test_scan_units_handles_superscripts() -> None:
    text = "Struttura regolabile 120-600 mm, carico 3,5 kN/m², superficie 1,44 m²"
    matches = list(scan_units(text))
    assert {match.unit for match in matches} == {"mm", "kn/m2", "m2"}
