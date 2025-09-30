import pytest

from robimb.extraction.parsers.numbers import NumberSpan, extract_numbers, parse_number_it


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0", 0.0),
        ("42", 42.0),
        ("-7", -7.0),
        ("+12", 12.0),
        ("1,5", 1.5),
        ("1.5", 1.5),
        ("1.234,56", 1234.56),
        ("12.345", 12345.0),
        ("12 345", 12345.0),
        ("1 234,7", 1234.7),
        ("1.234", 1234.0),
        ("€ 1.200,00", 1200.0),
        ("-1.200,50", -1200.5),
        ("0,001", 0.001),
        ("10%", 10.0),
        ("85‰", 85.0),
        ("3,14159", 3.14159),
        ("4.500", 4500.0),
        ("9,81", 9.81),
        ("1\u00A0234,50", 1234.5),
        ("5.000.000", 5000000.0),
        ("7,", 7.0),
        ("0,90", 0.9),
        ("1.000", 1000.0),
        ("-0,75", -0.75),
        ("+0,250", 0.25),
        ("123456", 123456.0),
        ("1.234.567,89", 1234567.89),
        ("3,50", 3.5),
        ("6,000", 6.0),
    ],
)
def test_parse_number_it(raw: str, expected: float) -> None:
    assert parse_number_it(raw) == pytest.approx(expected)


def test_parse_number_it_invalid() -> None:
    with pytest.raises(ValueError):
        parse_number_it("abc")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Anta 90x210 cm, peso 45 kg", [("90", 90.0), ("210", 210.0), ("45", 45.0)]),
        ("Coefficiente 0,75 e 1,25", [("0,75", 0.75), ("1,25", 1.25)]),
        ("Valori: 1.200,00; 900; 75%", [("1.200,00", 1200.0), ("900", 900.0), ("75", 75.0)]),
        ("Senza numeri", []),
        ("Livelli +3,50 / -0,25", [("+3,50", 3.5), ("-0,25", -0.25)]),
    ],
)
def test_extract_numbers(text: str, expected: list[tuple[str, float]]) -> None:
    spans = list(extract_numbers(text))
    assert [span.raw for span in spans] == [raw for raw, _ in expected]
    assert [span.value for span in spans] == pytest.approx([value for _, value in expected])
    for span in spans:
        assert isinstance(span, NumberSpan)
        assert 0 <= span.start <= span.end <= len(text)
