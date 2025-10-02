import pytest

from robimb.extraction.parsers.acoustic import parse_acoustic_coefficient


def test_parse_acoustic_requires_alpha_w_label() -> None:
    text = "Pannello certificato con αw = 0,65 secondo ISO 354."
    matches = list(parse_acoustic_coefficient(text))
    assert matches
    assert pytest.approx(matches[0].value, rel=1e-3) == 0.65


def test_parse_acoustic_ignores_unlabelled_values() -> None:
    text = "Il pannello offre NRC = 0,85 con ottime prestazioni."
    assert list(parse_acoustic_coefficient(text)) == []
