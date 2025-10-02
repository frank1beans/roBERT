import pytest

from robimb.extraction.parsers.thermal import parse_thermal_transmittance


def test_parse_thermal_transmittance_accepts_plain_m2_units() -> None:
    text = "Porta con Uw = 1,30 W/m2K garantito."
    matches = list(parse_thermal_transmittance(text))
    assert matches
    match = matches[0]
    assert match.label == "Uw"
    assert pytest.approx(match.value, rel=1e-3) == 1.30


def test_parse_thermal_transmittance_accepts_parenthesised_units() -> None:
    text = "La trasmittanza termica pari a 1,40 W/(m2K) conforme alla norma."
    matches = list(parse_thermal_transmittance(text))
    assert matches
    match = matches[0]
    assert match.label == "U"
    assert pytest.approx(match.value, rel=1e-3) == 1.40
