from robimb.extraction.parsers.standards import parse_standards


def test_parse_standards_extracts_prefix_code() -> None:
    text = "Porta certificata UNI EN 13501-2:2016 conforme alle normative."
    matches = list(parse_standards(text))
    assert matches
    match = matches[0]
    assert match.prefix == "UNI"
    assert match.code == "13501"
    assert match.year == "2016"
