import json

import pytest

from robimb.extraction.llm_adapter import SchemaField, StructuredLLMAdapter


def test_structured_llm_adapter_builds_prompt():
    adapter = StructuredLLMAdapter({"roof.color": SchemaField(type="string", description="Colore principale")})
    prompt = adapter.build_prompt("roof color red")
    assert "roof.color" in prompt
    assert "Colore principale" in prompt


def test_structured_llm_adapter_parses_response():
    schema = {
        "roof.color": SchemaField(type="string"),
        "roof.score": SchemaField(type="number"),
    }
    adapter = StructuredLLMAdapter(schema)
    response = json.dumps(
        {
            "roof.color": {"value": "red", "confidence": 0.75},
            "roof.score": 0.5,
        }
    )
    stage = adapter.build_stage(response)
    assert len(stage.candidates) == 2
    color_candidate = [c for c in stage.candidates if c.property_id == "roof.color"][0]
    assert color_candidate.value == "red"
    assert color_candidate.confidence == pytest.approx(0.75)


def test_structured_llm_adapter_rejects_invalid_json():
    adapter = StructuredLLMAdapter({"roof.color": "string"})
    with pytest.raises(ValueError):
        adapter.parse_response("not-json")

