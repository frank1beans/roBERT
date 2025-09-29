from robimb.extraction.normalizers import BUILTIN_NORMALIZERS


_truncate = BUILTIN_NORMALIZERS["truncate_model_value"]


def test_truncate_model_value_removes_leading_stopwords():
    value = "il PX04"
    context = "modello il PX04 con finitura opaca"
    assert _truncate(value, context) == "PX04"


def test_truncate_model_value_stops_on_followup_tokens():
    value = "PX04 campionatura prevista"
    context = "modello PX04 campionatura prevista dalla D.L."
    assert _truncate(value, context) == "PX04"


def test_truncate_model_value_rejects_invalid_phrases():
    value = "saranno da verificare"
    context = "modello sarà da verificare con la direzione lavori"
    assert _truncate(value, context) is None


def test_truncate_model_value_handles_lists():
    value = ["il PX07", "saranno da verificare"]
    context = "modello il PX07 completo di accessori"
    assert _truncate(value, context) == ["PX07"]


def test_truncate_model_value_keeps_short_codes():
    value = "PX"
    context = "modello PX"
    assert _truncate(value, context) == "PX"
