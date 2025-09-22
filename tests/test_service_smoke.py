
def test_import_app():
    from robimb.service.app import app
    assert app is not None


def test_predict_filters_properties(monkeypatch):
    from robimb.service.app import PredictIn, predict

    class DummyPack:
        def __init__(self):
            self.extractors = {
                "patterns": [
                    {"property_id": "grpA.alpha", "regex": [r"\balpha\b"], "normalizers": []},
                    {"property_id": "grpB.beta", "regex": [r"\bbeta\b"], "normalizers": []},
                ]
            }
            self.registry = {
                "groups": {
                    "grpA": {"properties": ["grpA.alpha"]},
                    "grpB": {"properties": ["grpB.beta"]},
                }
            }
            self.catmap = {
                "mappings": [
                    {
                        "cat_id": "cat.a",
                        "cat_label": "Categoria A",
                        "groups_required": ["grpA"],
                        "groups_recommended": [],
                        "props_required": [],
                        "props_recommended": [],
                        "keynote_mapping": {},
                    },
                    {
                        "cat_id": "cat.b",
                        "cat_label": "Categoria B",
                        "groups_required": ["grpB"],
                        "groups_recommended": [],
                        "props_required": [],
                        "props_recommended": [],
                        "keynote_mapping": {},
                    },
                ]
            }
            self.validators = {"rules": []}
            self.templates = {"templates": []}

    dummy_pack = DummyPack()

    def fake_load_pack(path):
        return dummy_pack

    def fake_load_model(model_path, label_index_path, calibrator_path=None):
        return object(), object(), {0: "Categoria A", 1: "Categoria B"}, None

    def fake_predict_topk(text, model, tokenizer, id2label, topk=5, calibrator=None):
        results = [
            {"id": 0, "label": "Categoria A", "score": 0.9},
            {"id": 1, "label": "Categoria B", "score": 0.1},
        ]
        return results[0], results, [0.9, 0.1], [0.9, 0.1]

    monkeypatch.setattr("robimb.service.app._load_pack_once", fake_load_pack)
    monkeypatch.setattr("robimb.service.app._load_model_once", fake_load_model)
    monkeypatch.setattr("robimb.inference.predict_category.predict_topk", fake_predict_topk)

    payload = PredictIn(text="alpha beta", topk=2)
    response = predict(payload)
    assert response.category["label"] == "Categoria A"
    assert "grpA.alpha" in response.properties
    assert "grpB.beta" not in response.properties
