"""Fusion policy between rule-based and QA candidates."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

LOGGER = logging.getLogger(__name__)


CandidateDict = Dict[str, Any]


@dataclass(frozen=True)
class FusionThresholds:
    """Threshold configuration for QA-based fusion."""

    qa_min: float = 0.25
    qa_confident: float = 0.60


def _is_valid(candidate: Optional[CandidateDict]) -> bool:
    return bool(candidate and candidate.get("value") not in (None, ""))


def fuse_property_candidates(
    rules_candidate: Optional[CandidateDict],
    qa_candidate: Optional[CandidateDict],
    *,
    fusion_mode: str = "fuse",
    thresholds: FusionThresholds = FusionThresholds(),
) -> Tuple[Optional[CandidateDict], str]:
    """Return the fused candidate and a textual reason for logging."""

    mode = fusion_mode.lower()
    if mode not in {"rules_only", "qa_only", "fuse"}:
        raise ValueError(f"Unsupported fusion mode '{fusion_mode}'")

    qa_score = float(qa_candidate.get("confidence", 0.0)) if qa_candidate else 0.0

    if mode == "rules_only":
        LOGGER.debug("fusion.rules_only", extra={"qa_score": qa_score})
        return rules_candidate, "rules_only"

    if mode == "qa_only":
        if _is_valid(qa_candidate) and qa_score >= thresholds.qa_min:
            LOGGER.debug(
                "fusion.qa_only.accept", extra={"qa_score": qa_score, "qa_min": thresholds.qa_min}
            )
            return qa_candidate, "qa_only"
        LOGGER.debug(
            "fusion.qa_only.reject",
            extra={"qa_score": qa_score, "qa_min": thresholds.qa_min},
        )
        return None, "qa_only_below_threshold"

    # Default "fuse" behaviour
    if not _is_valid(qa_candidate):
        LOGGER.debug("fusion.qa_missing", extra={"qa_score": qa_score})
        return rules_candidate, "qa_missing"

    if qa_score < thresholds.qa_min:
        LOGGER.debug(
            "fusion.qa_low_score",
            extra={"qa_score": qa_score, "qa_min": thresholds.qa_min},
        )
        return rules_candidate, "qa_below_min"

    if qa_score >= thresholds.qa_confident:
        LOGGER.debug(
            "fusion.qa_high_confidence",
            extra={"qa_score": qa_score, "qa_confident": thresholds.qa_confident},
        )
        return qa_candidate, "qa_high_confidence"

    if not _is_valid(rules_candidate):
        LOGGER.debug(
            "fusion.rules_missing",
            extra={"qa_score": qa_score, "qa_min": thresholds.qa_min, "qa_confident": thresholds.qa_confident},
        )
        return qa_candidate, "qa_preferred_missing_rules"

    LOGGER.debug(
        "fusion.rules_preferred",
        extra={
            "qa_score": qa_score,
            "qa_min": thresholds.qa_min,
            "qa_confident": thresholds.qa_confident,
        },
    )
    return rules_candidate, "rules_preferred"

