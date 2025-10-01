from robimb.extraction.matchers.materials import MaterialMatcher


def test_material_matcher_returns_synonym() -> None:
    matcher = MaterialMatcher()
    text = "Rivestimento in gres porcellanato Marazzi"
    matches = matcher.find(text)
    assert any(match.value == "grès porcellanato" for match in matches)
    gres_match = next(match for match in matches if match.value == "grès porcellanato")
    assert gres_match.surface.lower() in text.lower()


def test_material_matcher_handles_accent_and_synonyms() -> None:
    matcher = MaterialMatcher()
    text = "Pavimento in gres tecnico e battiscopa in acciaio inox satinato"
    matches = matcher.find(text)
    values = {match.value for match in matches}
    assert "grès porcellanato" in values
    assert "acciaio inox" in values
