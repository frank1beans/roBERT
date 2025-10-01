from __future__ import annotations

import pytest

from robimb.extraction.fusion_policy import FusionThresholds, fuse_property_candidates


def _candidate(value: str, score: float, source: str = "rules") -> dict:
    return {
        "value": value,
        "confidence": score,
        "source": source,
        "span": [0, len(value)],
        "errors": [],
    }


def test_rules_only_mode_prefers_rules() -> None:
    rules = _candidate("cartongesso", 0.8)
    qa = _candidate("acciaio", 0.9, source="qa")
    chosen, reason = fuse_property_candidates(rules, qa, fusion_mode="rules_only")
    assert chosen is rules
    assert reason == "rules_only"


@pytest.mark.parametrize(
    "score,expected_reason", [
        (0.70, "qa_high_confidence"),
        (0.30, "qa_preferred_missing_rules"),
    ],
)
def test_fuse_mode_prefers_qa_when_confident(score: float, expected_reason: str) -> None:
    qa = _candidate("cartongesso", score, source="qa")
    rules = None if expected_reason == "qa_preferred_missing_rules" else _candidate("acciaio", 0.5)
    chosen, reason = fuse_property_candidates(
        rules,
        qa,
        fusion_mode="fuse",
        thresholds=FusionThresholds(qa_min=0.25, qa_confident=0.60),
    )
    assert chosen is qa
    assert reason == expected_reason


def test_fuse_mode_retains_rules_for_low_score() -> None:
    qa = _candidate("cartongesso", 0.4, source="qa")
    rules = _candidate("acciaio", 0.7)
    chosen, reason = fuse_property_candidates(
        rules,
        qa,
        fusion_mode="fuse",
        thresholds=FusionThresholds(qa_min=0.25, qa_confident=0.60),
    )
    assert chosen is rules
    assert reason == "rules_preferred"


def test_qa_only_mode_filters_by_threshold() -> None:
    qa = _candidate("cartongesso", 0.2, source="qa")
    chosen, reason = fuse_property_candidates(
        None,
        qa,
        fusion_mode="qa_only",
        thresholds=FusionThresholds(qa_min=0.25, qa_confident=0.60),
    )
    assert chosen is None
    assert reason == "qa_only_below_threshold"
