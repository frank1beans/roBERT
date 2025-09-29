
import pytest

from robimb.extraction import extract_properties
from robimb.extraction.resources import load_default


PACK_V1 = load_default()

def test_basic_extraction():
    extractors_pack = {
        "version":"0.0.1",
        "patterns":[
            {"property_id":"geo.spessore_elemento", "regex":[r"sp\s*(\d+(?:[.,]\d+)?)\s*mm", r"(\d+(?:[.,]\d+)?)\s*cm\b"], "normalizers":["comma_to_dot","to_float","if_cm_to_mm"]},
            {"property_id":"frs.resistenza_fuoco_EI","regex":[r"\b(REI|EI)\s*(15|30|45|60|90|120|180|240)\b"], "normalizers":["EI_from_any"]},
            {"property_id":"aco.rw","regex":[r"\bRw\s*(\d{2})\s*dB\b"], "normalizers":["to_int"]},
            {"property_id":"opn.trasmittanza_uw","regex":[r"\bUw\s*=?\s*(\d+(?:[.,]\d+)?)\b"], "normalizers":["comma_to_dot","to_float"]},
            {"property_id":"flr.formato","regex":[r"\b(\d{2,4})\s*[x×]\s*(\d{2,4})\b"], "normalizers":["dims_join"]},
            {"property_id":"geo.foratura_laterizio","regex":[r"\b(forato|semi-?pieno|pieno)\b"], "normalizers":["lower","map_enum:laterizio_norm"]},
        ],
        "normalizers":{
            "laterizio_norm": {"forato":"forato","semi-pieno":"semipieno","semipieno":"semipieno","pieno":"pieno"}
        }
    }
    text = "Muratura in laterizio forato sp 25 cm REI120; parete Rw 54 dB; serr. Uw=1,30 W/m²K; form. 60x60"
    props = extract_properties(text, extractors_pack)
    assert abs(props["geo.spessore_elemento"] - 250.0) < 1e-6
    assert props["frs.resistenza_fuoco_EI"] == "EI 120"
    assert props["aco.rw"] == 54
    assert abs(props["opn.trasmittanza_uw"] - 1.30) < 1e-6
    assert props["flr.formato"] == "60×60"
    assert props["geo.foratura_laterizio"] == "forato"


def test_pack_v1_normalizers_examples():
    text = (
        "Unità di misura m²; Formato 60 x 120; EI120; laterizio semi pieno; "
        "sp 2,5 cm; Parete Rw 54 dB."
    )
    props = extract_properties(text, PACK_V1)

    formato_values = [value for key, value in props.items() if key.endswith(".formato")]
    assert formato_values
    assert any(
        (isinstance(value, list) and value == ["60", "120"]) or value == "60×120"
        for value in formato_values
    )

    classe_values = [value for key, value in props.items() if key.endswith("classe_ei")]
    assert "EI120" in classe_values

    spessore_values = [value for key, value in props.items() if key.endswith("spessore_mm")]
    assert any(abs(val - 25.0) < 1e-6 for val in spessore_values)

    rw_values = [value for key, value in props.items() if key.endswith("rw_db")]
    assert any(str(val) == "54" or pytest.approx(float(val), rel=1e-6) == 54.0 for val in rw_values)


def test_empty_groups_are_filtered():
    extractors_pack = {
        "version": "0.0.1",
        "patterns": [
            {
                "property_id": "safety.classe",
                "regex": [r"classe_EN795\s+(?:([A-Z])|([a-z]))"],
                "normalizers": [],
            }
        ],
    }
    text = "Dispositivo di ancoraggio classe_EN795 e"

    props = extract_properties(text, extractors_pack)

    assert props["safety.classe"] == "e"


def test_whitespace_captures_are_ignored():
    extractors_pack = {
        "version": "0.0.1",
        "patterns": [
            {
                "property_id": "debug.blank",
                "regex": [r"\bblank\b\s*(?P<val>\s+)"],
                "normalizers": ["strip"],
            }
        ],
    }
    text = "blank     END"

    props = extract_properties(text, extractors_pack)

    assert "debug.blank" not in props


def test_empty_lists_are_dropped():
    extractors_pack = {
        "version": "0.0.1",
        "patterns": [
            {
                "property_id": "debug.size",
                "regex": [
                    r"\bsize\b\s*(?P<val>\d+)\s*(?P<unit>mm)\b",
                    r"\bsize\b(?P<val>\s*)(?P<unit>\s*)",
                ],
                "normalizers": [],
            }
        ],
    }

    text_with_value = "size 120 mm"
    props = extract_properties(text_with_value, extractors_pack)
    assert props["debug.size"] == ["120", "mm"]

    text_without_value = "size     "
    props_empty = extract_properties(text_without_value, extractors_pack)
    assert "debug.size" not in props_empty
