from robimb.extraction.matchers.materials import MaterialMatcher


def test_material_matcher_returns_synonym() -> None:
    matcher = MaterialMatcher()
    text = "Rivestimento in gres porcellanato Marazzi"
    matches = matcher.find(text)
    assert any(match.value == "gres" for match in matches)
    gres_match = next(match for match in matches if match.value == "gres")
    assert gres_match.surface.lower() in text.lower()
