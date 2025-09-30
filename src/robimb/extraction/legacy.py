"""Regex-based property extraction engine used for backwards compatibility.

This module rebuilds the legacy regex matcher so that existing
utilities such as dataset conversion can continue to operate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

__all__ = [
    "Pattern",
    "extract_properties",
    "dry_run",
    "validate_extractors_pack",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    """Compiled representation of a property extraction rule."""

    property_id: str
    regexes: Tuple[re.Pattern[str], ...]
    tags: Tuple[str, ...]
    normalizers: Tuple[str, ...]
    collect_many: bool = False
    priority: int = 0

    def matches_tags(self, target_tags: Optional[Sequence[str]]) -> bool:
        if not self.tags:
            return True
        if not target_tags:
            return False
        target = {tag.lower() for tag in target_tags}
        return all(tag.lower() in target for tag in self.tags)


Flag = int

_FLAG_ALIASES: Mapping[str, Flag] = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}

_DEFAULT_FLAGS = re.IGNORECASE | re.MULTILINE


def _ensure_sequence(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _compile_regex(pattern: str, flags: Flag) -> re.Pattern[str]:
    return re.compile(pattern, flags)


def _normalize_pattern_spec(spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    property_id = spec.get("property_id") or spec.get("property") or spec.get("id")
    if not isinstance(property_id, str) or not property_id:
        return None
    regexes: List[str] = []
    for key in ("regex", "pattern", "patterns", "expressions"):
        raw = spec.get(key)
        if isinstance(raw, str):
            regexes.append(raw)
        elif isinstance(raw, Sequence):
            regexes.extend(str(item) for item in raw if isinstance(item, (str, bytes)))
        if regexes:
            break
    if not regexes:
        return None
    tags = [tag for tag in _ensure_sequence(spec.get("tags")) if isinstance(tag, str)]
    normalizers = [
        name for name in _ensure_sequence(spec.get("normalizers")) if isinstance(name, str) and name
    ]
    collect_many = bool(spec.get("collect_many") or spec.get("collectMany"))
    priority = int(spec.get("priority", 0))
    return {
        "property_id": property_id,
        "regex": regexes,
        "tags": tags,
        "normalizers": normalizers,
        "collect_many": collect_many,
        "priority": priority,
    }


def _compile_patterns(
    pack: Mapping[str, Any],
    allowed_properties: Optional[Sequence[str]],
    target_tags: Optional[Sequence[str]],
) -> Tuple[Pattern, ...]:
    defaults = pack.get("defaults") if isinstance(pack, Mapping) else None
    default_norms = []
    default_flags = _DEFAULT_FLAGS
    default_collect_many = False
    if isinstance(defaults, Mapping):
        default_norms = [
            name for name in _ensure_sequence(defaults.get("normalizers")) if isinstance(name, str) and name
        ]
        raw_flags = _ensure_sequence(defaults.get("flags"))
        flags_value = 0
        for item in raw_flags:
            if isinstance(item, str) and item.upper() in _FLAG_ALIASES:
                flags_value |= _FLAG_ALIASES[item.upper()]
            elif isinstance(item, int):
                flags_value |= int(item)
        if flags_value:
            default_flags = flags_value
        default_collect_many = bool(defaults.get("collect_many"))

    allowed: Optional[set[str]] = None
    if allowed_properties:
        allowed = {str(prop) for prop in allowed_properties}

    compiled: List[Pattern] = []
    raw_patterns = pack.get("patterns") if isinstance(pack, Mapping) else None
    if isinstance(raw_patterns, Sequence):
        for spec in raw_patterns:
            if not isinstance(spec, Mapping):
                continue
            normalized = _normalize_pattern_spec(spec)
            if not normalized:
                continue
            property_id = normalized["property_id"]
            if allowed is not None and property_id not in allowed:
                continue
            tags = tuple(normalized["tags"] or [])
            if tags and target_tags and not Pattern(property_id, tuple(), tags, tuple()).matches_tags(target_tags):
                continue
            flags = default_flags
            if "flags" in spec:
                custom_flags = 0
                for item in _ensure_sequence(spec.get("flags")):
                    if isinstance(item, str) and item.upper() in _FLAG_ALIASES:
                        custom_flags |= _FLAG_ALIASES[item.upper()]
                    elif isinstance(item, int):
                        custom_flags |= int(item)
                if custom_flags:
                    flags = custom_flags
            regexes = tuple(_compile_regex(pattern, flags) for pattern in normalized["regex"])
            normals = tuple(normalized["normalizers"] or default_norms)
            collect_many = bool(normalized["collect_many"] or default_collect_many)
            compiled.append(
                Pattern(
                    property_id=property_id,
                    regexes=regexes,
                    tags=tags,
                    normalizers=normals,
                    collect_many=collect_many,
                    priority=int(normalized["priority"]),
                )
            )
    compiled.sort(key=lambda pat: pat.priority)
    return tuple(compiled)


def _cache_key(
    pack: Mapping[str, Any],
    allowed_properties: Optional[Sequence[str]],
    target_tags: Optional[Sequence[str]],
) -> Tuple[int, Tuple[str, ...], Tuple[str, ...]]:
    allowed_key: Tuple[str, ...] = tuple(sorted(str(prop) for prop in allowed_properties)) if allowed_properties else ()
    tags_key: Tuple[str, ...] = tuple(sorted(str(tag) for tag in target_tags)) if target_tags else ()
    return (id(pack), allowed_key, tags_key)


_COMPILED_CACHE: MutableMapping[Tuple[int, Tuple[str, ...], Tuple[str, ...]], Tuple[Pattern, ...]] = {}


def _extract_value(match: re.Match[str]) -> Optional[str]:
    if "value" in match.groupdict():
        value = match.group("value")
        if value:
            return value
    groups = match.groups()
    for group in groups:
        if group:
            return group
    return match.group(0) if match.group(0) else None


Normalizer = Callable[[Any], Any]


def _apply_map_enum(value: Any, mapping: Mapping[str, Any]) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    for key, mapped in mapping.items():
        if not isinstance(key, str):
            continue
        if lower == key.lower():
            return mapped
    return mapping.get(text, text)


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"si", "sì", "yes", "true", "1"}:
        return True
    if text in {"no", "false", "0"}:
        return False
    return value


def _to_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return value
    cleaned = text.replace("%", "").replace("\u202f", " ").replace("\xa0", " ")
    cleaned = cleaned.replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError:
        return value
    if abs(number - round(number)) < 1e-6:
        return int(round(number))
    return number


def _to_mm(value: Any, match: Optional[re.Match[str]]) -> Any:
    number = _to_number(value)
    if isinstance(number, str):
        return number
    if match is None:
        return number
    unit = match.groupdict().get("unit")
    if not unit:
        return number
    unit = unit.lower()
    if unit in {"mm", "millimetri"}:
        return number
    if unit in {"cm", "centimetri"}:
        return float(number) * 10
    if unit in {"m", "metri"}:
        return float(number) * 1000
    return number


def _normalize_spaces(value: Any) -> Any:
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_lower(value: Any) -> Any:
    return str(value).strip().lower()


def _normalize_upper(value: Any) -> Any:
    return str(value).strip().upper()


def _normalize_ei(value: Any) -> Any:
    text = str(value).upper()
    digits = re.findall(r"\d+", text)
    suffix = digits[0] if digits else ""
    return f"EI {suffix}".strip()


def _normalize_fire_reaction(value: Any) -> Any:
    text = str(value).upper().replace(" ", "")
    text = text.replace("S", "S").replace("D", "D")
    return text


def _normalize_pei(value: Any) -> Any:
    text = str(value).upper().replace(" ", "")
    if not text.startswith("PEI"):
        text = "PEI" + text
    return text


def _normalize_slip_class(value: Any) -> Any:
    text = str(value).upper().replace(" ", "")
    return text


def _normalize_format(value: Any, match: Optional[re.Match[str]]) -> Any:
    text = _normalize_spaces(value)
    unit = match.groupdict().get("unit") if match else None
    unit = unit.lower() if isinstance(unit, str) else None
    text = text.replace("×", "x")
    if unit:
        return f"{text} {unit}"
    return text


_BUILTIN_NORMALIZERS: Mapping[str, Callable[[Any, Optional[re.Match[str]]], Any]] = {
    "strip": lambda value, match: str(value).strip() if value is not None else value,
    "collapse_spaces": lambda value, match: _normalize_spaces(value) if value is not None else value,
    "lower": lambda value, match: _normalize_lower(value) if value is not None else value,
    "upper": lambda value, match: _normalize_upper(value) if value is not None else value,
    "to_number": lambda value, match: _to_number(value),
    "to_mm": lambda value, match: _to_mm(value, match),
    "to_bool_strict": lambda value, match: _normalize_bool(value),
    "normalize_ei": lambda value, match: _normalize_ei(value),
    "normalize_fire_reaction": lambda value, match: _normalize_fire_reaction(value),
    "normalize_pei": lambda value, match: _normalize_pei(value),
    "normalize_slip_class": lambda value, match: _normalize_slip_class(value),
    "normalize_format": lambda value, match: _normalize_format(value, match),
}


def _apply_normalizers(
    value: Any,
    normalizers: Sequence[str],
    pack_normalizers: Mapping[str, Any],
    match: Optional[re.Match[str]],
    property_id: str,
) -> Any:
    current = value
    for name in normalizers:
        if current is None:
            break
        normalizer = name.strip()
        if not normalizer:
            continue
        if normalizer.startswith("map_enum:"):
            key = normalizer.split(":", 1)[1].strip()
            mapping = pack_normalizers.get(key)
            if isinstance(mapping, Mapping):
                current = _apply_map_enum(current, mapping)
            continue
        func = _BUILTIN_NORMALIZERS.get(normalizer)
        if func is not None:
            current = func(current, match)
            continue
        mapping = pack_normalizers.get(normalizer)
        if isinstance(mapping, Mapping):
            current = _apply_map_enum(current, mapping)
    return current


def _iter_matches(text: str, pattern: Pattern) -> Iterable[Tuple[re.Match[str], Any]]:
    for regex in pattern.regexes:
        for match in regex.finditer(text):
            value = _extract_value(match)
            if value is None:
                continue
            yield match, value


def extract_properties(
    text: str,
    pack: Mapping[str, Any],
    *,
    allowed_properties: Optional[Sequence[str]] = None,
    target_tags: Optional[Sequence[str]] = None,
    collect_many: bool = False,
) -> Dict[str, Any]:
    """Extract property values from *text* using the provided *pack*."""

    cache_key = _cache_key(pack, allowed_properties, target_tags)
    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is None:
        compiled = _compile_patterns(pack, allowed_properties, target_tags)
        _COMPILED_CACHE[cache_key] = compiled

    results: Dict[str, Any] = {}
    pack_normalizers = pack.get("normalizers", {}) if isinstance(pack, Mapping) else {}
    target_tags_tuple = tuple(target_tags) if target_tags else None
    for pattern in compiled:
        if target_tags_tuple and not pattern.matches_tags(target_tags_tuple):
            continue
        collected: List[Any] = []
        for match, raw_value in _iter_matches(text, pattern):
            value = _apply_normalizers(raw_value, pattern.normalizers, pack_normalizers, match, pattern.property_id)
            if value is None or value == "":
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            collected.append(value)
        if not collected:
            continue
        if collect_many or pattern.collect_many:
            existing = results.setdefault(pattern.property_id, [])
            for item in collected:
                if item not in existing:
                    existing.append(item)
        else:
            results.setdefault(pattern.property_id, collected[0])
    return results


def dry_run(
    samples: Iterable[Any],
    pack: Mapping[str, Any],
    *,
    allowed_properties: Optional[Sequence[str]] = None,
    target_tags: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Run the extractor on *samples* returning the collected properties."""

    outputs: List[Dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, Mapping):
            text = str(sample.get("text", ""))
        else:
            text = str(sample)
        outputs.append(
            extract_properties(
                text,
                pack,
                allowed_properties=allowed_properties,
                target_tags=target_tags,
                collect_many=True,
            )
        )
    return outputs


def validate_extractors_pack(pack: Mapping[str, Any]) -> List[str]:
    """Validate the pack returning a list of problems (empty if valid)."""

    errors: List[str] = []
    patterns = pack.get("patterns") if isinstance(pack, Mapping) else None
    if not isinstance(patterns, Sequence) or not patterns:
        return ["extractors pack must define a non-empty 'patterns' list"]
    defaults = pack.get("defaults") if isinstance(pack, Mapping) else None
    flags = _DEFAULT_FLAGS
    if isinstance(defaults, Mapping):
        raw_flags = _ensure_sequence(defaults.get("flags"))
        for item in raw_flags:
            if isinstance(item, str) and item.upper() in _FLAG_ALIASES:
                flags |= _FLAG_ALIASES[item.upper()]
            elif isinstance(item, int):
                flags |= int(item)
    for idx, raw_spec in enumerate(patterns, start=1):
        if not isinstance(raw_spec, Mapping):
            errors.append(f"pattern {idx}: expected a mapping, got {type(raw_spec).__name__}")
            continue
        normalized = _normalize_pattern_spec(raw_spec)
        if not normalized:
            errors.append(f"pattern {idx}: missing property_id or regex definitions")
            continue
        custom_flags = flags
        if "flags" in raw_spec:
            custom_flags = 0
            for item in _ensure_sequence(raw_spec.get("flags")):
                if isinstance(item, str) and item.upper() in _FLAG_ALIASES:
                    custom_flags |= _FLAG_ALIASES[item.upper()]
                elif isinstance(item, int):
                    custom_flags |= int(item)
            if not custom_flags:
                custom_flags = flags
        for pattern in normalized["regex"]:
            try:
                _compile_regex(pattern, custom_flags)
            except re.error as exc:
                errors.append(
                    f"pattern {idx} ({normalized['property_id']}): invalid regex '{pattern}': {exc}"
                )
    return errors
