import json
from pathlib import Path

import pytest

from robimb.extraction.fuse import Fuser, FusePolicy
from robimb.extraction.orchestrator import Orchestrator, OrchestratorConfig
from robimb.extraction.qa_llm import MockLLM


def _write_registry(tmp_path: Path) -> Path:
    category_id = "categoria_test"
    schema_path = tmp_path / "schema.json"
    schema_payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Schema test",
        "type": "object",
        "properties": {
            "category": {"const": category_id},
            "properties": {
                "type": "object",
                "properties": {
                    "dimensioni": {
                        "properties": {
                            "value": {
                                "type": "object",
                                "properties": {
                                    "width_mm": {"type": "number"},
                                    "height_mm": {"type": "number"},
                                },
                                "required": ["width_mm", "height_mm"],
                            }
                        }
                    }
                },
            },
        },
    }
    schema_path.write_text(json.dumps(schema_payload, ensure_ascii=False), encoding="utf-8")

    registry_path = tmp_path / "registry.json"
    registry_payload = {
        "categories": [
            {
                "id": category_id,
                "name": "Categoria test",
                "schema": str(schema_path),
                "required": [],
                "properties": [
                    {
                        "id": "dimensioni",
                        "title": "Dimensioni",
                        "type": "object",
                        "sources": ["parser", "qa_llm"],
                    }
                ],
            }
        ]
    }
    registry_path.write_text(json.dumps(registry_payload, ensure_ascii=False), encoding="utf-8")
    return registry_path


def test_orchestrator_basic(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    cfg = OrchestratorConfig(registry_path=str(registry_path))
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority)
    orchestrator = Orchestrator(fuse=fuser, llm=MockLLM(), cfg=cfg)

    doc = {
        "text_id": "doc-1",
        "categoria": "categoria_test",
        "text": "Porta dimensioni 90x210 cm con finitura bianca.",
    }

    result = orchestrator.extract_document(doc)
    properties = result["properties"]["dimensioni"]

    assert properties["source"] == "parser"
    assert properties["confidence"] >= 0.85
    assert pytest.approx(properties["value"]["width_mm"], rel=1e-3) == 900.0
    assert pytest.approx(properties["value"]["height_mm"], rel=1e-3) == 2100.0
    assert result["validation"]["status"] == "ok"


def test_orchestrator_accepts_category_alias(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    cfg = OrchestratorConfig(registry_path=str(registry_path))
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=MockLLM(),
        cfg=cfg,
    )

    doc = {
        "id": 42,
        "cat": "categoria_test",
        "text": "Pannello in cartongesso 60x60 cm spessore 12 mm.",
    }

    result = orchestrator.extract_document(doc)

    assert result["categoria"] == "categoria_test"
    assert result["text_id"] == "42"


def test_orchestrator_uses_super_category_as_fallback(tmp_path: Path) -> None:
    registry_path = _write_registry(tmp_path)
    cfg = OrchestratorConfig(registry_path=str(registry_path))
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=MockLLM(),
        cfg=cfg,
    )

    doc = {
        "text": "Specchio rettangolare 70x50 cm incluso fissaggio.",
        "super": "Categoria test",
        "cat": "Accessori per l'allestimento di servizi igienici",
    }

    result = orchestrator.extract_document(doc)

    assert result["categoria"] == "categoria_test"
    # Without an explicit identifier the orchestrator should keep the field empty
    assert result["text_id"] is None
