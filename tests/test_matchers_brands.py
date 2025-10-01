import pytest

from robimb.extraction.fuse import Fuser, FusePolicy
from robimb.extraction.matchers.brands import BrandMatcher
from robimb.extraction.orchestrator import Orchestrator, OrchestratorConfig


@pytest.fixture()
def orchestrator() -> Orchestrator:
    cfg = OrchestratorConfig(enable_llm=False)
    fuser = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=cfg.source_priority)
    return Orchestrator(fuse=fuser, llm=None, cfg=cfg)


def test_brand_matcher_matches_canonical_and_synonym() -> None:
    matcher = BrandMatcher()
    text = "Sistema cartongesso Knauf Italia con profili Profilgessi"
    matches = matcher.find(text)
    values = {value for value, _, _ in matches}
    assert "Knauf Italia" in values
    assert "Profilgessi" in values


def test_brand_matcher_handles_accents_and_hyphen_variants() -> None:
    matcher = BrandMatcher()
    text = "Serramenti Schuco abbinati a lastre Saint Gobain Gyproc"
    matches = matcher.find(text)
    values = {value for value, _, _ in matches}
    assert "Schüco" in values
    assert "Saint-Gobain Gyproc" in values


def test_brand_matcher_category_filtering() -> None:
    matcher = BrandMatcher()
    text = "Miscelatore Hansgrohe per lavabo a parete"
    matches_app = matcher.find(text, category="apparecchi_sanitari_accessori")
    assert any(value == "Hansgrohe" for value, _, _ in matches_app)

    matches_cart = matcher.find(text, category="opere_da_cartongessista")
    assert not matches_cart


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
