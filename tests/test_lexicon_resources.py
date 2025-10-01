from robimb.extraction.lexicon import load_norms_by_category, load_producers_by_category


def test_norms_by_category_contains_expected_entries():
    norms = load_norms_by_category()
    assert "opere_da_cartongessista" in norms
    cartongesso_uni = norms["opere_da_cartongessista"]["uni"]
    assert any(entry["code"] == "UNI 11424:2011" for entry in cartongesso_uni)
    transversal = norms["normative_trasversali"]
    assert any(item["code"] == "D.Lgs. 81/2008" for item in transversal)


def test_producers_by_category_covers_verticals():
    producers = load_producers_by_category()
    assert "opere_da_serramentista" in producers
    assert {"Internorm Italia", "Schüco", "Rehau"}.issubset(producers["opere_da_serramentista"])
    sanitari = producers["apparecchi_sanitari_e_accessori"]
    assert "Grohe" in sanitari
    assert "Geberit" in sanitari
