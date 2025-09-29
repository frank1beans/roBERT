from types import SimpleNamespace

from robimb.extraction import ExtractionRouter, SpanTagger
from robimb.registry.schemas import CategoryDefinition, PropertySlot


def _make_pack(span_enabled: bool = False) -> SimpleNamespace:
    category = CategoryDefinition(
        key="roof|roof",
        super="Roof",
        category="Roof",
        slots={
            "roof.color": PropertySlot(property_id="roof.color", name="Color", type="text"),
            "roof.extra": PropertySlot(property_id="roof.extra", name="Extra", type="bool"),
        },
    )
    extractors = {
        "patterns": [
            {"property_id": "roof.color", "regex": [r"roof color (\w+)"], "confidence": 0.8},
        ]
    }
    catmap = {
        "mappings": [
            {
                "cat_label": "Roof",
                "props_required": ["roof.color"],
                "props_recommended": ["roof.extra"],
                "groups_required": [],
                "groups_recommended": [],
                "keynote_mapping": {},
            }
        ]
    }
    validators = {"rules": []}
    registry = {"groups": {}}
    pack = SimpleNamespace(
        extractors=extractors,
        catmap=catmap,
        validators=validators,
        registry=registry,
        category_models={category.key: category},
    )
    if span_enabled:
        pack.span_router = True  # sentinel for tests
    return pack


def test_router_runs_rule_stage_and_postprocess():
    pack = _make_pack()
    router = ExtractionRouter(pack)
    result = router.extract("roof color red", categories="Roof")

    assert result.values() == {"roof.color": "red"}
    assert result.extraction.stages[0].stage == "R0"
    assert result.extraction.stages[0].candidates[0].provenance == "rules:regex"


def test_router_merges_span_tagger_predictions():
    pack = _make_pack(span_enabled=True)

    def _predictor(text, allowed):
        yield {"property_id": "roof.extra", "value": "true", "confidence": 0.6}
        yield {"property_id": "ignored.prop", "value": 1}

    router = ExtractionRouter(pack, span_tagger=SpanTagger(_predictor))
    result = router.extract("roof color blue", categories="Roof")

    assert result.values()["roof.color"] == "blue"
    assert result.values()["roof.extra"] is True
    best = result.extraction.best_by_property()["roof.extra"]
    assert best.stage == "R1"
    assert result.postprocess.issues == [] or result.postprocess.issues is None

