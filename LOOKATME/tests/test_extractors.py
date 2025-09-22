
from robimb.features.extractors import extract_properties

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
