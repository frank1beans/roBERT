"""Utilities for loading registry definitions and extractor packs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..registry import RegistryLoader, load_pack
from ..registry.schemas import CategoryDefinition, PropertySlot

__all__ = [
    "ExtractorsPack",
    "load_property_registry",
    "build_registry_extractors",
    "load_extractors_pack",
    "merge_extractors_pack",
]


@dataclass(frozen=True)
class ExtractorsPack:
    """Typed representation of an extractors configuration."""

    patterns: List[Any] = field(default_factory=list)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            key: value for key, value in self.extras.items() if key != "patterns"
        }
        payload["patterns"] = [pattern for pattern in self.patterns]
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExtractorsPack":
        patterns: List[Any] = []
        raw_patterns = payload.get("patterns", [])
        if isinstance(raw_patterns, Iterable):
            for item in raw_patterns:
                if isinstance(item, Mapping):
                    patterns.append(dict(item))
                else:
                    patterns.append(item)
        extras = {key: value for key, value in payload.items() if key != "patterns"}
        return cls(patterns=patterns, extras=extras)

    def merge(self, other: "ExtractorsPack") -> "ExtractorsPack":
        combined_patterns: List[Any] = list(self.patterns) + list(other.patterns)
        extras: Dict[str, Any] = {
            key: value for key, value in self.extras.items() if key != "patterns"
        }
        for key, value in other.extras.items():
            if key == "normalizers" and isinstance(value, Mapping):
                base = extras.get("normalizers")
                merged = dict(base) if isinstance(base, Mapping) else {}
                merged.update(value)
                extras["normalizers"] = merged
            elif key not in extras:
                extras[key] = value
        return ExtractorsPack(patterns=combined_patterns, extras=extras)


def _resolve_pack_json(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / "pack.json"
        if candidate.exists():
            return candidate
    return path


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_property_registry(path: Path) -> Optional[Dict[str, CategoryDefinition]]:
    """Load a property registry returning :class:`CategoryDefinition` objects."""

    path = Path(path)
    original_path = path
    if path.is_dir():
        for name in (
            "registry.json",
            "properties_registry_extended.json",
            "properties_registry.json",
        ):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break

    try:
        payload = _load_json(path)
    except OSError:
        payload = None

    if isinstance(payload, Mapping) and "files" in payload:
        files = payload.get("files", {})
        if isinstance(files, Mapping):
            registry_ref = files.get("registry")
            if isinstance(registry_ref, str):
                registry_path = (path.parent / registry_ref).resolve()
                if registry_path.exists():
                    return load_property_registry(registry_path)

    registry: Dict[str, CategoryDefinition] = {}
    try:
        loader = RegistryLoader(path)
    except FileNotFoundError:
        loader = None
    except Exception:  # pragma: no cover - defensive fallback
        loader = None
    if loader is not None:
        try:
            registry = loader.load_registry()
        except Exception:  # pragma: no cover - defensive fallback
            registry = {}
        if registry:
            return registry

    fallback = _load_flat_registry(path)
    if fallback:
        return fallback

    if original_path != path:
        return load_property_registry(original_path)
    return None


def _load_flat_registry(path: Path) -> Dict[str, CategoryDefinition]:
    """Build a registry from the simplified schema-first resources."""

    if not path.exists():
        return {}

    try:
        payload = _load_json(path)
    except OSError:
        return {}

    categories_payload = payload.get("categories") if isinstance(payload, Mapping) else None
    if not isinstance(categories_payload, list):
        return {}

    base_dir = path.parent
    patterns_map = _load_flat_patterns(base_dir)

    registry: Dict[str, CategoryDefinition] = {}
    for entry in categories_payload:
        if not isinstance(entry, Mapping):
            continue
        cat_id = str(entry.get("id") or "")
        if not cat_id:
            continue
        cat_name = str(entry.get("name") or cat_id)
        key = f"{cat_name}|{cat_name}"

        slots: Dict[str, PropertySlot] = {}
        properties = entry.get("properties")
        if isinstance(properties, list):
            for prop in properties:
                if not isinstance(prop, Mapping):
                    continue
                prop_id = str(prop.get("id") or "")
                if not prop_id:
                    continue
                slot_kwargs: Dict[str, Any] = {
                    "property_id": prop_id,
                    "name": str(prop.get("title") or prop_id),
                }
                prop_type = prop.get("type")
                if isinstance(prop_type, str):
                    slot_kwargs["type"] = prop_type
                unit = prop.get("unit")
                if isinstance(unit, str):
                    slot_kwargs["unit"] = unit
                enum_values = prop.get("enum")
                if isinstance(enum_values, list):
                    slot_kwargs["values"] = list(enum_values)
                default = prop.get("default")
                if default is not None:
                    slot_kwargs["default"] = default
                tags = prop.get("tags")
                if isinstance(tags, list):
                    slot_kwargs["tags"] = [str(tag) for tag in tags if isinstance(tag, str)]
                description = prop.get("description")
                if isinstance(description, str):
                    slot_kwargs["description"] = description
                sources = prop.get("sources")
                if isinstance(sources, list):
                    slot_kwargs["sources"] = [str(src) for src in sources if isinstance(src, str)]
                slots[prop_id] = PropertySlot(**slot_kwargs)

        definition = CategoryDefinition(
            key=key,
            super=cat_name,
            category=cat_name,
            slots=slots,
            patterns={},
            metadata={"required": list(entry.get("required", []))},
        )
        registry[key] = definition

    if not registry:
        return {}

    for prop_id, regexes in patterns_map.items():
        for definition in registry.values():
            if prop_id in definition.slots:
                definition.patterns[prop_id] = list(regexes)

    return registry


def _load_flat_patterns(base_dir: Path) -> Dict[str, List[str]]:
    """Load regex patterns from the schema-first extractors pack."""

    candidates = [
        base_dir / "extractors.json",
        base_dir.parent / "extractors.json",
        base_dir.parent / "pack" / "current" / "extractors.json",
        base_dir.parent.parent / "pack" / "current" / "extractors.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = _load_json(candidate)
        except OSError:
            continue
        patterns = payload.get("patterns") if isinstance(payload, Mapping) else None
        if not isinstance(patterns, list):
            continue
        mapping: Dict[str, List[str]] = {}
        for entry in patterns:
            if not isinstance(entry, Mapping):
                continue
            prop_id = entry.get("property_id")
            regexes = entry.get("regex")
            if not isinstance(prop_id, str) or not isinstance(regexes, list):
                continue
            cleaned = [str(rx) for rx in regexes if isinstance(rx, str) and rx]
            if cleaned:
                mapping[prop_id] = cleaned
        if mapping:
            return mapping
    return {}


def _normalize_extractors_payload(payload: Any) -> Optional[ExtractorsPack]:
    if isinstance(payload, Mapping):
        if "extractors" in payload and isinstance(payload["extractors"], Mapping):
            return _normalize_extractors_payload(payload["extractors"])
        patterns = payload.get("patterns") if "patterns" in payload else None
        if isinstance(patterns, list):
            pack: Dict[str, Any] = {"patterns": patterns}
            if "normalizers" in payload and isinstance(payload["normalizers"], Mapping):
                pack["normalizers"] = dict(payload["normalizers"])
            if "defaults" in payload and isinstance(payload["defaults"], Mapping):
                pack["defaults"] = dict(payload["defaults"])
            return ExtractorsPack.from_payload(pack)
    elif isinstance(payload, list):
        return ExtractorsPack.from_payload({"patterns": list(payload)})
    return None


def build_registry_extractors(
    registry: Mapping[str, CategoryDefinition]
) -> Optional[ExtractorsPack]:
    patterns: List[Dict[str, Any]] = []
    for key, category in registry.items():
        if not isinstance(category, CategoryDefinition):
            continue
        schema_patterns = category.patterns
        if not schema_patterns:
            continue
        slots = category.slots
        tags: List[str] = []
        if category.super_label:
            tags.append(f"category:{category.super_label}")
        if category.category_label:
            tags.append(f"subcategory:{category.category_label}")
        for prop_id, regexes in schema_patterns.items():
            if not isinstance(prop_id, str):
                continue
            if not isinstance(regexes, (list, tuple, set)):
                continue
            cleaned = [str(rx) for rx in regexes if isinstance(rx, str) and rx]
            if not cleaned:
                continue
            pattern_spec: Dict[str, Any] = {"property_id": prop_id, "regex": cleaned}
            if tags:
                pattern_spec["tags"] = list(tags)
            slot_info = slots.get(prop_id) if isinstance(slots, Mapping) else None
            if slot_info is not None:
                slot_payload = (
                    slot_info if isinstance(slot_info, Mapping) else slot_info.model_dump()
                )
                normals = _infer_slot_normalizers(slot_payload)
                if normals:
                    pattern_spec["normalizers"] = normals
            patterns.append(pattern_spec)
    if not patterns:
        return None
    return ExtractorsPack.from_payload({"patterns": patterns})


def _infer_slot_normalizers(slot: Mapping[str, Any]) -> List[str]:
    slot_type = str(slot.get("type", "")).strip().lower()
    if slot_type in {"float", "number", "numeric", "ratio"}:
        return ["to_number"]
    if slot_type in {"int", "integer"}:
        return ["to_number"]
    if slot_type in {"bool", "boolean"}:
        return ["to_bool_strict"]
    if slot_type in {"enum", "text"}:
        return ["strip"]
    return []


def load_extractors_pack(path: Path) -> Optional[ExtractorsPack]:
    """Load an extractors pack from raw JSON or a knowledge pack."""

    path = Path(path)
    if path.is_dir():
        for name in (
            "extractors_extended.json",
            "extractors.json",
        ):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break

    path = _resolve_pack_json(path)
    try:
        payload = _load_json(path)
    except OSError:
        return None

    pack = _normalize_extractors_payload(payload)
    if pack is not None:
        return pack

    if isinstance(payload, Mapping) and "files" in payload:
        files = payload.get("files", {})
        if isinstance(files, Mapping):
            extractors_ref = files.get("extractors")
            if isinstance(extractors_ref, str):
                extractors_path = (path.parent / extractors_ref).resolve()
                if extractors_path.exists():
                    nested_payload = _load_json(extractors_path)
                    normalized = _normalize_extractors_payload(nested_payload)
                    if normalized is not None:
                        return normalized

    try:
        pack_obj = load_pack(str(path))
    except Exception:  # pragma: no cover - defensive fallback
        return None
    extractors_payload = getattr(pack_obj, "extractors", None)
    if isinstance(extractors_payload, Mapping):
        return ExtractorsPack.from_payload(extractors_payload)
    return None


def merge_extractors_pack(
    primary: Optional[ExtractorsPack],
    secondary: Optional[ExtractorsPack],
) -> Optional[ExtractorsPack]:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    return primary.merge(secondary)
