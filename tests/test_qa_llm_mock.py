import json
from pathlib import Path

from robimb.extraction.fuse import Fuser, FusePolicy
from robimb.extraction.orchestrator import Orchestrator, OrchestratorConfig
from robimb.extraction.qa_llm import MockLLM


def _registry_llm_only(tmp_path: Path) -> Path:
    category_id = "cat_llm"
    schema_path = tmp_path / "schema.json"
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Schema QA",
        "type": "object",
        "properties": {
            "category": {"const": category_id},
            "properties": {
                "type": "object",
                "properties": {
                    "descrizione": {
                        "properties": {
                            "value": {"type": ["string", "null"]}
                        }
                    }
                },
            },
        },
    }
    schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    registry_path = tmp_path / "registry.json"
    registry = {
        "categories": [
            {
                "id": category_id,
                "name": "Categoria QA",
                "schema": str(schema_path),
                "required": [],
                "properties": [
                    {"id": "descrizione", "title": "Descrizione", "type": "string", "sources": ["qa_llm"]}
                ],
            }
        ]
    }
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    return registry_path


def test_mock_llm_returns_null(tmp_path: Path) -> None:
    registry_path = _registry_llm_only(tmp_path)
    cfg = OrchestratorConfig(registry_path=str(registry_path))
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=MockLLM(),
        cfg=cfg,
    )

    doc = {"text_id": "qa", "categoria": "cat_llm", "text": "Dato non presente."}
    result = orchestrator.extract_document(doc)
    prop = result["properties"]["descrizione"]

    assert prop["value"] is None
    assert prop["source"] is None
    assert prop["confidence"] == 0.0
    assert "no_valid_candidate" in prop["errors"]
