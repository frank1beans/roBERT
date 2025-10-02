from robimb.extraction.parsers.fire_class import parse_fire_class


def test_parse_fire_class_allows_spaces_in_suffix() -> None:
    text = "Controsoffitto con classe di reazione al fuoco B-s1, d0 certificata."
    matches = list(parse_fire_class(text))
    assert matches
    assert matches[0].value == "B-s1,d0"


def test_parse_fire_class_ignores_classe_energetica() -> None:
    text = "La porta è in classe energetica B per il consumo elettrico."
    assert list(parse_fire_class(text)) == []
