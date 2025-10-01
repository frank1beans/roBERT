import pytest

from robimb.extraction.fuse import Fuser, FusePolicy
from robimb.extraction.orchestrator import Orchestrator, OrchestratorConfig


@pytest.fixture()
def orchestrator() -> Orchestrator:
    cfg = OrchestratorConfig(enable_llm=False)
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority)
    return Orchestrator(fuse=fuser, llm=None, cfg=cfg)


def test_brand_matcher_returns_category_specific_match(orchestrator: Orchestrator) -> None:
    doc = {
        "categoria": "opere_da_cartongessista",
        "text": "Lastra in cartongesso Knauf Italia con elevate performance acustiche.",
    }

    result = orchestrator.extract_document(doc)

    marchio = result["properties"]["marchio"]
    assert marchio["source"] == "matcher"
    assert marchio["value"] == "Knauf Italia"
    assert marchio["confidence"] >= 0.7


def test_brand_matcher_emits_fallback_for_incompatible_category(orchestrator: Orchestrator) -> None:
    doc = {
        "categoria": "opere_da_cartongessista",
        "text": "Serratura Yale per portoncini blindati ad alta sicurezza.",
    }

    result = orchestrator.extract_document(doc)

    marchio = result["properties"]["marchio"]
    assert marchio["value"] == "Generico"
    assert marchio["source"] == "fallback"
    assert pytest.approx(marchio["confidence"], rel=1e-3) == 0.05
