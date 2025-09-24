"""Regex based property extraction engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .dsl import ExtractorsPack
from .normalizers import BUILTIN_NORMALIZERS, Normalizer, build_normalizer


@dataclass
class Pattern:
    """Compiled representation of a pattern entry in the pack."""

    property_id: str
    regex: Sequence[str]
    normalizers: Sequence[str]
    language: Optional[str] = None
    confidence: Optional[float] = None
    tags: Sequence[str] | None = None
    compiled_regex: List[re.Pattern[str]] = field(default_factory=list)


class PatternValidationError(ValueError):
    """Raised when at least one regex pattern cannot be compiled."""

    def __init__(self, errors: Sequence[Dict[str, Any]]):
        self.errors: List[Dict[str, Any]] = list(errors)
        summary = ", ".join(
            f"{err.get('property_id')}: {err.get('regex')} -> {err.get('error')}" for err in self.errors
        )
        super().__init__(f"Invalid extractor patterns: {summary}")


def _compile_patterns(
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
    target_tags: Optional[Iterable[str]] = None,
) -> List[Pattern]:
    pats: List[Pattern] = []
    errors: List[Dict[str, Any]] = []
    allowed: Optional[set[str]] = None
    if allowed_properties is not None:
        allowed = {p for p in allowed_properties if p}
    requested_tags: Optional[set[str]] = None
    if target_tags is not None:
        requested_tags = {t for t in target_tags if t}

    for item in extractors_pack.get("patterns", []):
        pid = item["property_id"]
        if allowed is not None and pid not in allowed:
            continue
        pattern_tags_raw = item.get("tags")
        tags_list: Optional[List[str]] = None
        if isinstance(pattern_tags_raw, Sequence):
            tags_list = [str(tag) for tag in pattern_tags_raw if isinstance(tag, str) and tag]
        if requested_tags is not None:
            pattern_tags = set(tags_list or [])
            if not pattern_tags or pattern_tags.isdisjoint(requested_tags):
                continue
        regex_list = list(item.get("regex", []))
        compiled_regex: List[re.Pattern[str]] = []
        for rx in regex_list:
            try:
                compiled_regex.append(re.compile(rx, flags=re.IGNORECASE))
            except re.error as exc:  # pragma: no cover - defensive guard
                errors.append({"property_id": pid, "regex": rx, "error": str(exc)})
        pats.append(
            Pattern(
                property_id=pid,
                regex=regex_list,
                normalizers=list(item.get("normalizers", [])),
                language=item.get("language"),
                confidence=item.get("confidence"),
                tags=tags_list,
                compiled_regex=compiled_regex,
            )
        )
    if errors:
        raise PatternValidationError(errors)
    return pats


def _apply_normalizers(
    value: Any,
    matched_text: str,
    normalizers: Sequence[str],
    extractors_pack: ExtractorsPack,
) -> Any:
    v = value
    for name in normalizers:
        fn: Normalizer = build_normalizer(name, extractors_pack)
        v = fn(v, matched_text)
    return v


def _coerce_capture(match: re.Match[str]) -> Any:
    """Normalize match groups dropping empty values."""

    # No groups: return entire matched text
    if match.lastindex is None:
        return match.group(0)

    groups = [match.group(i) for i in range(1, match.lastindex + 1)]

    def _is_empty(value: Any) -> bool:
        return value is None or (isinstance(value, str) and value == "")

    cleaned = [value for value in groups if not _is_empty(value)]

    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned


def extract_properties(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
    *,
    target_tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply patterns declared in ``extractors_pack`` to ``text``."""

    out: Dict[str, Any] = {}
    pats = _compile_patterns(
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    for pat in pats:
        has_collect = "collect_many" in pat.normalizers
        for _, compiled_rx in zip(pat.regex, pat.compiled_regex):
            for m in compiled_rx.finditer(text):
                cap = _coerce_capture(m)
                val = _apply_normalizers(cap, m.group(0), pat.normalizers, extractors_pack)
                if has_collect:
                    prev = out.get(pat.property_id, [])
                    if not isinstance(prev, list):
                        prev = [prev]
                    prev.append(val)
                    out[pat.property_id] = prev
                else:
                    if pat.property_id not in out:
                        out[pat.property_id] = val
            if pat.property_id in out and not has_collect:
                break
        if has_collect and pat.property_id in out and "unique_list" in pat.normalizers:
            out[pat.property_id] = BUILTIN_NORMALIZERS["unique_list"](out[pat.property_id], "")
    return out


def dry_run(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
    *,
    target_tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return debug information with raw matches and normalized output."""

    details: List[Dict[str, Any]] = []
    pats = _compile_patterns(
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    for pat in pats:
        for rx_str, compiled in zip(pat.regex, pat.compiled_regex):
            for m in compiled.finditer(text):
                details.append(
                    {
                        "property_id": pat.property_id,
                        "regex": rx_str,
                        "match": m.group(0),
                        "groups": m.groups(),
                    }
                )
    extracted = extract_properties(
        text,
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    return {"matches": details, "extracted": extracted}


def validate_extractors_pack(extractors_pack: ExtractorsPack) -> None:
    """Validate that all regex patterns in the pack compile successfully."""

    _compile_patterns(extractors_pack)


__all__ = ["Pattern", "PatternValidationError", "extract_properties", "dry_run", "validate_extractors_pack"]
