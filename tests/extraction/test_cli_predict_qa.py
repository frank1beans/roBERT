from __future__ import annotations

import json

from types import SimpleNamespace

from robimb.extraction import property_qa


def test_cli_predict_qa(monkeypatch, capsys, tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "properties": {
                    "properties": {
                        "properties": {
                            "materiale": {
                                "$ref": "#/$defs/value"
                            }
                        }
                    }
                },
                "$defs": {"value": {"type": "object"}},
            }
        ),
        encoding="utf-8",
    )
    registry_payload = {
        "categories": [
            {
                "id": "cat1",
                "name": "Categoria 1",
                "schema": str(schema_path),
                "properties": [
                    {
                        "id": "materiale",
                        "title": "Materiale",
                        "type": "string",
                        "sources": ["matcher", "qa"],
                    }
                ],
            }
        ]
    }
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")

    class DummyModel(SimpleNamespace):
        def eval(self):  # pragma: no cover - trivial stub
            return self

        def to(self, _device):  # pragma: no cover - trivial stub
            return self

    class DummyTokenizer(SimpleNamespace):
        cls_token_id = 0

    def fake_model_loader(*_args, **_kwargs):  # pragma: no cover - stub
        return DummyModel()

    def fake_tokenizer_loader(*_args, **_kwargs):  # pragma: no cover - stub
        return DummyTokenizer()

    def fake_predict(_model, _tokenizer, examples, **_kwargs):
        return {
            examples[0].property_id: {"span": "cartongesso", "start": 10, "end": 21, "score": 0.9}
        }

    monkeypatch.setattr(property_qa, "AutoModelForQuestionAnswering", SimpleNamespace(from_pretrained=fake_model_loader))
    monkeypatch.setattr(property_qa, "AutoTokenizer", SimpleNamespace(from_pretrained=fake_tokenizer_loader))
    monkeypatch.setattr(property_qa, "predict_with_encoder", fake_predict)

    args = [
        "predict",
        "--model-dir",
        str(tmp_path / "dummy"),
        "--text",
        "parete in cartongesso",
        "--category",
        "cat1",
        "--registry",
        str(registry_path),
        "--null-th",
        "0.25",
    ]

    property_qa.main(args)
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["materiale"]["span"] == "cartongesso"
