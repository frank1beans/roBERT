from robimb.extraction.parsers.colors import parse_ral_colors


def test_parse_ral_colors_detects_code() -> None:
    text = "Finitura disponibile nei colori RAL 9010 e ral7016."
    matches = list(parse_ral_colors(text))
    codes = {match.code for match in matches}
    assert "RAL 9010" in codes
    assert "RAL 7016" in codes
