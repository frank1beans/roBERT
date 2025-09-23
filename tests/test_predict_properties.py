from types import SimpleNamespace

from robimb.inference.predict_properties import predict_properties


def _make_pack():
    return SimpleNamespace(
        catmap={
            "mappings": [
                {
                    "cat_label": "Roof",
                    "props_required": ["roof_color"],
                    "props_recommended": [],
                    "groups_required": [],
                    "groups_recommended": [],
                    "keynote_mapping": {},
                }
            ]
        },
        registry={"groups": {}},
        extractors={
            "patterns": [
                {"property_id": "roof_color", "regex": [r"roof color (\w+)"]},
                {"property_id": "wall_color", "regex": [r"wall color (\w+)"]},
            ]
        },
    )


def test_predict_properties_filters_by_category():
    pack = _make_pack()
    text = "roof color red and wall color blue"

    props = predict_properties(text, pack, "Roof")

    assert props == {"roof_color": "red"}
