import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from robimb.cli.extract import app


def _prepare_registry(tmp_path: Path) -> Path:
    category_id = "categoria_cli"
    schema_path = tmp_path / "schema.json"
    schema_payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Schema CLI",
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
                                }
                            }
                        }
                    },
                    "isolamento_db": {
                        "properties": {
                            "value": {"type": "number"}
                        }
                    },
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
                "name": "Categoria CLI",
                "schema": str(schema_path),
                "required": [],
                "properties": [
                    {"id": "dimensioni", "title": "Dimensioni", "type": "object", "sources": ["parser"]},
                    {
                        "id": "isolamento_db",
                        "title": "Isolamento acustico",
                        "type": "number",
                        "unit": "db",
                        "sources": ["parser", "qa_llm"],
                    },
                ],
            }
        ]
    }
    registry_path.write_text(json.dumps(registry_payload, ensure_ascii=False), encoding="utf-8")
    return registry_path


def test_extract_cli_end_to_end(tmp_path: Path) -> None:
    registry_path = _prepare_registry(tmp_path)
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    docs = [
        {
            "text_id": "doc-1",
            "categoria": "categoria_cli",
            "text": "Pannello con dimensioni 0,90x2,10 m e isolamento ≥ 50 dB.",
        },
        {"text_id": "doc-2", "categoria": "categoria_cli", "text": "Nessuna informazione."},
    ]
    with input_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "properties",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--pack",
            str(pack_dir),
            "--schema",
            str(registry_path),
        ],
    )

    assert result.exit_code == 0
    assert "completed" in result.stdout

    lines = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(lines) == 2

    first = lines[0]
    dim = first["properties"]["dimensioni"]
    iso = first["properties"]["isolamento_db"]
    assert dim["value"]["width_mm"] == pytest.approx(900.0, rel=1e-3)
    assert dim["value"]["height_mm"] == pytest.approx(2100.0, rel=1e-3)
    assert iso["value"] == pytest.approx(50.0, rel=1e-3)
    assert first["validation"]["status"] == "ok"
    assert first["confidence_overall"] > 0.0

    second = lines[1]
    assert second["properties"]["dimensioni"]["value"] is None
    assert second["properties"]["isolamento_db"]["value"] is None
    assert second["confidence_overall"] == 0.0
