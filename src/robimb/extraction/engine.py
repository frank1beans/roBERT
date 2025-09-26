"""Regex based property extraction engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    unit: Optional[str] = None
    examples: Sequence[str] | None = None
    max_matches: Optional[int] = None
    first_wins: Optional[bool] = None
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
                compiled_regex.append(re.compile(rx, flags=re.IGNORECASE | re.UNICODE))
            except re.error as exc:  # pragma: no cover - defensive guard
                errors.append({"property_id": pid, "regex": rx, "error": str(exc)})

        max_matches: Optional[int] = None
        raw_max = item.get("max_matches")
        if isinstance(raw_max, int) and raw_max > 0:
            max_matches = raw_max
        elif isinstance(raw_max, str) and raw_max.isdigit():
            max_matches = int(raw_max)

        first_wins: Optional[bool] = None
        if "first_wins" in item:
            first_wins = bool(item.get("first_wins"))

        pats.append(
            Pattern(
                property_id=pid,
                regex=regex_list,
                normalizers=list(item.get("normalizers", [])),
                language=item.get("language"),
                confidence=item.get("confidence"),
                tags=tags_list,
                unit=item.get("unit"),
                examples=list(item.get("examples", [])) if item.get("examples") else None,
                max_matches=max_matches,
                first_wins=first_wins,
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

    named = match.groupdict()
    if named:
        for key in ("val", "value"):
            if key in named and named[key] not in (None, ""):
                primary = named[key]
                extras = []
                for extra_key in ("unit", "second", "unit2", "min", "max", "value2"):
                    extra_val = named.get(extra_key)
                    if extra_val not in (None, ""):
                        extras.append(extra_val)
                if extras:
                    return [primary, *extras]
                return primary
        values = [v for v in named.values() if v not in (None, "")]
        if len(values) == 1:
            return values[0]
        if values:
            return values

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


def _extract_properties_internal(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
    *,
    target_tags: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Return both extracted properties and their confidences."""

    out: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}
    pats = _compile_patterns(
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )

    defaults = extractors_pack.get("defaults", {}) or {}
    default_norms_raw = defaults.get("normalizers", [])
    default_normalizers: List[str] = []
    if isinstance(default_norms_raw, Sequence) and not isinstance(default_norms_raw, (str, bytes)):
        default_normalizers = [str(n) for n in default_norms_raw]

    selection_default = str(defaults.get("selection_strategy", "first_wins")).lower()
    if selection_default not in {"first_wins", "best_confidence"}:
        selection_default = "first_wins"

    property_modes: Dict[str, str] = {}
    collect_properties: Set[str] = set()

    for pat in pats:
        property_id = pat.property_id
        combined_normalizers = list(default_normalizers) + list(pat.normalizers)
        has_collect = "collect_many" in combined_normalizers
        if has_collect:
            collect_properties.add(property_id)
            existing = out.get(property_id)
            if not isinstance(existing, list):
                out[property_id] = [] if existing is None else [existing]

        selection_mode = property_modes.get(property_id)
        if selection_mode is None:
            if pat.first_wins is not None:
                selection_mode = "first_wins" if pat.first_wins else "best_confidence"
            else:
                selection_mode = selection_default
            property_modes[property_id] = selection_mode

        matched_count = 0
        for compiled_rx in pat.compiled_regex:
            if has_collect and pat.max_matches is not None and matched_count >= pat.max_matches:
                break
            if not has_collect and selection_mode == "first_wins" and property_id in out:
                break
            for m in compiled_rx.finditer(text):
                cap = _coerce_capture(m)
                if cap is None:
                    continue
                value = _apply_normalizers(cap, m.group(0), combined_normalizers, extractors_pack)
                if has_collect:
                    bucket = out.setdefault(property_id, [])
                    if isinstance(value, list):
                        for item in value:
                            if pat.max_matches is not None and matched_count >= pat.max_matches:
                                break
                            bucket.append(item)
                            matched_count += 1
                    else:
                        bucket.append(value)
                        matched_count += 1
                    if pat.max_matches is not None and matched_count >= pat.max_matches:
                        break
                else:
                    confidence = pat.confidence if pat.confidence is not None else 0.0
                    if selection_mode == "best_confidence":
                        previous = confidences.get(property_id, float("-inf"))
                        if property_id not in out or confidence > previous:
                            out[property_id] = value
                            confidences[property_id] = confidence
                    else:
                        if property_id not in out:
                            out[property_id] = value
                            confidences[property_id] = confidence
                    if selection_mode == "first_wins":
                        break
            else:
                continue
            break

    for prop in collect_properties:
        if prop in out:
            out[prop] = BUILTIN_NORMALIZERS["unique_list"](out[prop], "")
    return out, confidences


def extract_properties(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
    *,
    target_tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply patterns declared in ``extractors_pack`` to ``text``."""

    out, _ = _extract_properties_internal(
        text,
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    return out


@dataclass(frozen=True)
class PropertyExtractionResult:
    """Container exposing both values and confidences for extracted properties."""

    values: Dict[str, Any]
    confidences: Dict[str, float]


def extract_properties_with_confidences(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
    *,
    target_tags: Optional[Iterable[str]] = None,
) -> PropertyExtractionResult:
    """Return extracted properties alongside their confidence scores."""

    values, confidences = _extract_properties_internal(
        text,
        extractors_pack,
        allowed_properties=allowed_properties,
        target_tags=target_tags,
    )
    return PropertyExtractionResult(values=values, confidences=confidences)


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
                        "groups": m.groupdict() or m.groups(),
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


__all__ = [
    "Pattern",
    "PatternValidationError",
    "PropertyExtractionResult",
    "extract_properties",
    "extract_properties_with_confidences",
    "dry_run",
    "validate_extractors_pack",
]

