import pytest

from robimb.extraction.parsers.sound_insulation import parse_sound_insulation


def test_parse_sound_insulation_from_potere_fonoisolante() -> None:
    text = "Porta tagliafuoco con potere fonoisolante di 38 dB."
    matches = list(parse_sound_insulation(text))
    assert matches
    assert pytest.approx(matches[0].value, rel=1e-3) == 38.0


def test_parse_sound_insulation_from_abbattimento_acustico() -> None:
    text = "Infisso con abbattimento acustico 42 dB certificato."
    matches = list(parse_sound_insulation(text))
    assert matches
    assert pytest.approx(matches[0].value, rel=1e-3) == 42.0
