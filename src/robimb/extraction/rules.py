"""Stage R0: rule based extraction powered by the regex engine."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional

from .dsl import ExtractorsPack
from .engine import dry_run, extract_properties
from .formats import ExtractionCandidate, StageResult

__all__ = ["run_rules_stage"]


def _confidence_for(property_id: str, regex: str, extractors_pack: ExtractorsPack) -> Optional[float]:
    for pattern in extractors_pack.get("patterns", []):
        if pattern.get("property_id") != property_id:
            continue
        regexes = pattern.get("regex", [])
        if isinstance(regexes, Iterable) and regex in regexes:
            raw_conf = pattern.get("confidence")
            if isinstance(raw_conf, (int, float)):
                return float(raw_conf)
    return None


def run_rules_stage(
    text: str,
    extractors_pack: ExtractorsPack,
    *,
    allowed_properties: Optional[Iterable[str]] = None,
    target_tags: Optional[Iterable[str]] = None,
) -> StageResult:
    """Apply the legacy regex engine and capture provenance for each match."""

    extracted = extract_properties(
        text,
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    debug = dry_run(
        text,
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    matches = debug.get("matches", []) if isinstance(debug, Mapping) else []
    matches_by_prop: MutableMapping[str, list[Mapping[str, object]]] = {}
    for item in matches:
        property_id = str(item.get("property_id"))
        matches_by_prop.setdefault(property_id, []).append(item)

    stage = StageResult(stage="R0", extra={"matches": matches})
    for property_id, value in extracted.items():
        bucket = matches_by_prop.get(property_id) or []
        provenance = "rules:regex"
        confidence = None
        metadata = bucket[0] if bucket else None
        if bucket:
            confidence = _confidence_for(property_id, str(bucket[0].get("regex")), extractors_pack)
        stage.add(
            ExtractionCandidate(
                property_id=property_id,
                value=value,
                confidence=confidence,
                stage="R0",
                provenance=provenance,
                metadata=metadata,
            )
        )
    return stage

