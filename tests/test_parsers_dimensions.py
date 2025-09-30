import pytest

from robimb.extraction.parsers.dimensions import DimensionMatch, parse_dimensions


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Anta 90x210 cm", [(900.0, 2100.0)]),
        ("Porta 90 x 210 mm", [(90.0, 210.0)]),
        ("Porta 0,90×2,10 m", [(900.0, 2100.0)]),
        ("Lastra 1200x600", [(1200.0, 600.0)]),
        ("Formato 1,20 x 2,40 m", [(1200.0, 2400.0)]),
        ("Formato 600 x 600 mm, spessore 30 mm", [(600.0, 600.0)]),
        ("Pannello 600×600×40 mm", [(600.0, 600.0, 40.0)]),
        ("Dimensioni L90 H210", [(900.0, 2100.0)]),
        ("Dimensioni L 0,90 m H 2,10 m", [(900.0, 2100.0)]),
        ("L 90 cm H 210 cm P 45 mm", [(900.0, 2100.0, 45.0)]),
        ("Formato 90x210", [(900.0, 2100.0)]),
        ("Dimensioni 0,90x2,10", [(900.0, 2100.0)]),
        ("Formati disponibili: 60x60 cm e 120x60 cm", [(600.0, 600.0), (1200.0, 600.0)]),
        ("Taglio 1000 × 2500 mm", [(1000.0, 2500.0)]),
        ("Lastre 1250x2600x15", [(1250.0, 2600.0, 15.0)]),
        ("Formato modulare 2,4x2,4 m", [(2400.0, 2400.0)]),
        ("Telaio L 110 H 215", [(1100.0, 2150.0)]),
        ("Profilo 50 x 30 x 2 mm", [(50.0, 30.0, 2.0)]),
        ("Range 0,60×0,60 m", [(600.0, 600.0)]),
        ("Formati L1200 H300", [(1200.0, 300.0)]),
        ("Pannello 1,2 x 1 m", [(1200.0, 1000.0)]),
        ("Larghezza 900 mm, altezza 2100 mm", []),
        ("Testo senza dimensioni", []),
    ],
)
def test_parse_dimensions(text: str, expected: list[tuple[float, ...]]) -> None:
    matches = list(parse_dimensions(text))
    assert [tuple(round(value, 3) for value in match.values_mm) for match in matches] == [
        tuple(round(value, 3) for value in values) for values in expected
    ]
    for match in matches:
        assert isinstance(match, DimensionMatch)
        assert 0 <= match.span[0] <= match.span[1] <= len(text)
        assert match.unit == "mm"


def test_parse_dimensions_deduplicates_spans() -> None:
    text = "Dimensioni 90x210 cm (L90 H210 cm)"
    matches = list(parse_dimensions(text))
    spans = {match.span for match in matches}
    assert len(spans) == len(matches)
