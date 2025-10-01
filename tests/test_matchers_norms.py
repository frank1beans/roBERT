from robimb.extraction.matchers.norms import StandardMatcher


def test_standard_matcher_matches_category_synonym() -> None:
    matcher = StandardMatcher()
    text = "Parete certificata EN 520 conforme alla norma di cartongesso."
    matches = matcher.find(text, category="opere_da_cartongessista")
    assert any(match.value == "UNI EN 520" for match in matches)


def test_standard_matcher_filters_by_category() -> None:
    matcher = StandardMatcher()
    text = "Rivestimento conforme alla EN 520 con finitura lucida."
    matches = matcher.find(text, category="opere_di_rivestimento")
    assert not any(match.value == "UNI EN 520" for match in matches)


def test_standard_matcher_supports_transversal_regulation() -> None:
    matcher = StandardMatcher()
    text = "Prodotto marcato secondo il Regolamento (UE) 305/2011 vigente."
    matches = matcher.find(text, category="opere_di_pavimentazione")
    assert any(match.value == "Regolamento UE 305/2011" for match in matches)
