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


def test_parser_candidates_extract_length_value() -> None:
    cfg = OrchestratorConfig(
        source_priority=["parser"],
        enable_matcher=False,
        enable_llm=False,
        registry_path="",
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    text = "Lavabo formato 20x20 cm con finitura opaca."
    candidates = list(orchestrator._parser_candidates("dimensione_lunghezza", None, text))

    assert candidates, "expected at least one candidate for lunghezza"
    candidate = candidates[0]
    assert candidate["source"] == "parser"
    assert candidate["unit"] == "mm"
    assert pytest.approx(candidate["value"], rel=1e-3) == 200.0



def test_parser_candidates_keep_first_value_for_width_two_dimensions() -> None:
    cfg = OrchestratorConfig(
        source_priority=["parser"],
        enable_matcher=False,
        enable_llm=False,
        registry_path="",
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    text = "Porta con dimensioni 70x210 cm in legno massello."
    width_candidates = list(
        orchestrator._parser_candidates("dimensione_larghezza", None, text)
    )
    assert width_candidates, "expected at least one candidate for larghezza"
    width_candidate = width_candidates[0]
    assert width_candidate["source"] == "parser"
    assert width_candidate["unit"] == "mm"
    assert pytest.approx(width_candidate["value"], rel=1e-3) == 700.0

    height_candidates = list(
        orchestrator._parser_candidates("dimensione_altezza", None, text)
    )
    assert height_candidates, "expected at least one candidate for altezza"
    height_candidate = height_candidates[0]
    assert height_candidate["unit"] == "mm"
    assert pytest.approx(height_candidate["value"], rel=1e-3) == 2100.0

def test_orchestrator_extracts_normativa_riferimento() -> None:
    cfg = OrchestratorConfig()
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "opere_da_serramentista",
        "text": "Serramento conforme al Regolamento (UE) 305/2011 sulla marcatura CE.",
    }

    result = orchestrator.extract_document(doc)
    normative = result["properties"].get("normativa_riferimento", {})

    assert normative.get("value") == "Regolamento UE 305/2011"
    assert normative.get("source") == "matcher"


def _build_parser_only_orchestrator() -> Orchestrator:
    cfg = OrchestratorConfig(
        source_priority=["parser"],
        enable_matcher=False,
        enable_llm=False,
        registry_path="",
        use_qa=False,
    )
    return Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )


def test_parser_candidates_respect_height_marker() -> None:
    orchestrator = _build_parser_only_orchestrator()
    text = "Box doccia 90x70 h190 cm con profili lucidi."

    length = orchestrator._parser_candidates("dimensione_lunghezza", None, text)
    width = orchestrator._parser_candidates("dimensione_larghezza", None, text)
    height = orchestrator._parser_candidates("dimensione_altezza", None, text)

    length_value = next(iter(length))
    width_value = next(iter(width))
    height_value = next(iter(height))

    assert pytest.approx(length_value["value"], rel=1e-3) == 900.0
    assert pytest.approx(width_value["value"], rel=1e-3) == 700.0
    assert pytest.approx(height_value["value"], rel=1e-3) == 1900.0


def test_parser_candidates_two_dimensions_leave_height_empty() -> None:
    orchestrator = _build_parser_only_orchestrator()
    text = "Specchio rettangolare 235x70 cm con illuminazione."

    height_candidates = list(
        orchestrator._parser_candidates("dimensione_altezza", None, text)
    )
    width_candidates = list(
        orchestrator._parser_candidates("dimensione_larghezza", None, text)
    )

    assert not height_candidates
    assert width_candidates
    assert pytest.approx(width_candidates[0]["value"], rel=1e-3) == 700.0


def test_parser_candidates_skip_width_for_single_diameter() -> None:
    orchestrator = _build_parser_only_orchestrator()
    text = "Maniglione di sicurezza Ø35 mm in acciaio."

    width_candidates = list(
        orchestrator._parser_candidates("dimensione_larghezza", None, text)
    )
    height_candidates = list(
        orchestrator._parser_candidates("dimensione_altezza", None, text)
    )

    assert not width_candidates
    assert not height_candidates


def test_installation_type_parser_maps_floor_variants() -> None:
    cfg = OrchestratorConfig(
        registry_path="resources/data/properties/registry.json",
        enable_llm=False,
        use_qa=False,
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "apparecchi_sanitari_accessori",
        "text": "Vaso con scarico a pavimento ribassato e sedile incluso.",
    }

    result = orchestrator.extract_document(doc)
    installazione = result["properties"].get("tipologia_installazione", {})

    assert installazione.get("value") == "a_pavimento"
    assert installazione.get("source") == "parser"


def test_installation_type_parser_maps_wall_variants() -> None:
    cfg = OrchestratorConfig(
        registry_path="resources/data/properties/registry.json",
        enable_llm=False,
        use_qa=False,
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "apparecchi_sanitari_accessori",
        "text": "Maniglione da fissare su porta completa di viteria.",
    }

    result = orchestrator.extract_document(doc)
    installazione = result["properties"].get("tipologia_installazione", {})

    assert installazione.get("value") == "a_parete"
    assert installazione.get("source") == "parser"


def test_cartongesso_parser_extracts_classe_ei_and_isolante() -> None:
    cfg = OrchestratorConfig(
        registry_path="resources/data/properties/registry.json",
        enable_llm=False,
        use_qa=False,
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "opere_da_cartongessista",
        "text": "Controparete EI30 Classe A1 con lastra fonoisolante e termoisolante.",
    }

    result = orchestrator.extract_document(doc)
    classe_ei = result["properties"].get("classe_ei", {})
    isolante = result["properties"].get("presenza_isolante", {})

    assert classe_ei.get("value") == "EI30"
    assert classe_ei.get("source") == "parser"
    assert isolante.get("value") == "si"


def test_cartongesso_parser_detects_absence_of_isolante() -> None:
    cfg = OrchestratorConfig(
        registry_path="resources/data/properties/registry.json",
        enable_llm=False,
        use_qa=False,
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "opere_da_cartongessista",
        "text": "Parete tecnica priva di isolante con doppia lastra da 15 mm.",
    }

    result = orchestrator.extract_document(doc)
    isolante = result["properties"].get("presenza_isolante", {})

    assert isolante.get("value") == "no"
    assert isolante.get("source") == "parser"


def test_controssoffitti_acoustic_and_fire_parsing() -> None:
    cfg = OrchestratorConfig(
        registry_path="resources/data/properties/registry.json",
        enable_llm=False,
        use_qa=False,
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "controsoffitti",
        "text": (
            "Lamella 4akustik αw = 0,65 certificata ISO 354, classe di reazione al fuoco B-s1, d0."
        ),
    }

    result = orchestrator.extract_document(doc)
    acoustic = result["properties"].get("coefficiente_fonoassorbimento", {})
    fire = result["properties"].get("classe_reazione_al_fuoco", {})
    thickness = result["properties"].get("spessore_pannello_mm", {})

    assert pytest.approx(acoustic.get("value"), rel=1e-3) == 0.65
    assert acoustic.get("source") == "parser"
    assert fire.get("value") == "B-s1,d0"
    assert fire.get("source") == "parser"
    assert thickness.get("value") is None


def test_serramentista_transmittance_and_sound() -> None:
    cfg = OrchestratorConfig(
        registry_path="resources/data/properties/registry.json",
        enable_llm=False,
        use_qa=False,
    )
    orchestrator = Orchestrator(
        fuse=Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority),
        llm=None,
        cfg=cfg,
    )

    doc = {
        "categoria": "opere_da_serramentista",
        "text": "Porta tagliafuoco Uw = 1,30 W/m2K con potere fonoisolante di 38 dB.",
    }

    result = orchestrator.extract_document(doc)
    trans = result["properties"].get("trasmittanza_termica", {})
    sound = result["properties"].get("isolamento_acustico_db", {})

    assert pytest.approx(trans.get("value"), rel=1e-3) == 1.30
    assert trans.get("unit") == "W/m²K"
    assert trans.get("source") == "parser"
    assert pytest.approx(sound.get("value"), rel=1e-3) == 38.0
    assert sound.get("unit") == "dB"
    assert sound.get("source") == "parser"
