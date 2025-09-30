"""Pydantic models representing registry categories and property slots."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from pydantic import BaseModel, ConfigDict, Field

import re
import unicodedata

__all__ = [
    "PropertySlot",
    "CategoryDefinition",
    "slugify",
    "build_category_key",
    "build_property_id",
]

# ---------------------------------------------------------------------------
# Slug helpers reused across the registry stack
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Return a deterministic slug suitable for identifiers."""

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower()
    slug = _SLUG_RE.sub("_", lowered).strip("_")
    if slug:
        return slug
    digest = unicodedata.normalize("NFKD", str(value)).encode("utf-8")
    return "slot_" + re.sub(r"[^a-f0-9]", "", digest.hex())[:8]


def build_category_key(super_name: str, category_name: str) -> str:
    """Compose the canonical key used to identify a registry category."""

    return f"{super_name}|{category_name}"


def build_property_id(super_name: str, category_name: str, slot_name: str, *, inherited: bool = False) -> str:
    """Return the canonical property identifier used across the pipeline."""

    super_slug = slugify(super_name)
    slot_slug = slugify(slot_name)
    if inherited:
        return f"{super_slug}.__global__.{slot_slug}"
    category_slug = slugify(category_name)
    return f"{super_slug}.{category_slug}.{slot_slug}"


class PropertySlot(BaseModel):
    """Description of a single property declared inside the registry."""

    property_id: str = Field(..., description="Canonical identifier used downstream")
    name: str = Field(..., description="Human friendly slot label")
    type: Optional[str] = Field(default=None, description="Logical type of the property")
    priority: Optional[int] = Field(default=None, description="Relative importance used by UI components")
    tags: List[str] = Field(default_factory=list, description="List of category tags associated with the slot")
    unit: Optional[str] = Field(default=None, description="Measurement unit (if any)")
    values: Optional[List[Any]] = Field(default=None, description="Explicit enumeration of admissible values")
    enum: Optional[str] = Field(default=None, description="Reference to shared enumeration declared in the registry")
    expected_atoms: List[str] = Field(default_factory=list, description="List of atoms consumed by the extraction pipeline")
    default: Optional[Any] = Field(default=None, description="Optional default value")
    inherited: bool = Field(default=False, description="Whether the property comes from the _global block")

    model_config = ConfigDict(extra="allow")


class CategoryDefinition(BaseModel):
    """Flattened representation of a category schema."""

    key: str = Field(..., description="Composite key super|category")
    super_label: str = Field(..., alias="super")
    category_label: str = Field(..., alias="category")
    slots: MutableMapping[str, PropertySlot] = Field(default_factory=dict)
    patterns: MutableMapping[str, List[str]] = Field(default_factory=dict)
    groups: MutableMapping[str, Any] = Field(default_factory=dict)
    enums: MutableMapping[str, List[Any]] = Field(default_factory=dict)
    validators: MutableMapping[str, Any] = Field(default_factory=dict)
    normalizers: MutableMapping[str, Any] = Field(default_factory=dict)
    plugins: MutableMapping[str, List[str]] = Field(default_factory=dict)
    metadata: MutableMapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def property_ids(self) -> List[str]:
        return list(self.slots.keys())

    def json_schema(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description of the category."""

        slots: Dict[str, Any] = {}
        slot_meta: Dict[str, Dict[str, Any]] = {}
        for prop_id, slot in self.slots.items():
            payload = slot.model_dump(
                exclude={"property_id", "name", "inherited"},
                exclude_none=True,
                exclude_defaults=True,
            )
            slots[prop_id] = payload
            slot_meta[prop_id] = {"name": slot.name, "inherited": slot.inherited}
        schema: Dict[str, Any] = {"slots": slots}
        if self.patterns:
            schema["patterns"] = dict(self.patterns)
        if self.groups:
            schema["groups"] = dict(self.groups)
        if self.enums:
            schema["enums"] = dict(self.enums)
        if self.validators:
            schema["validators"] = dict(self.validators)
        if self.normalizers:
            schema["normalizers"] = dict(self.normalizers)
        metadata = dict(self.metadata)
        if slot_meta:
            if not isinstance(metadata.get("slots"), dict):
                metadata["slots"] = {}
            metadata["slots"].update(slot_meta)
        if metadata:
            schema["metadata"] = metadata
        if self.plugins:
            schema["plugins"] = {kind: list(values) for kind, values in self.plugins.items()}
        return schema

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_payload(
        cls,
        *,
        key: str,
        super_label: str,
        category_label: str,
        payload: Mapping[str, Any],
        inherited_slots: Optional[Mapping[str, Mapping[str, Any]]] = None,
        inherited_patterns: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> "CategoryDefinition":
        slots: Dict[str, PropertySlot] = {}
        metadata: Dict[str, Any] = {}
        inherited_slots = inherited_slots or {}
        inherited_patterns = inherited_patterns or {}

        for prop_id, slot_payload in inherited_slots.items():
            if not isinstance(slot_payload, Mapping):
                continue
            slot = PropertySlot(
                property_id=prop_id,
                name=str(slot_payload.get("name") or slot_payload.get("label") or prop_id),
                inherited=True,
                **{k: v for k, v in slot_payload.items() if k not in {"name", "label"}},
            )
            slots[prop_id] = slot

        raw_slots = payload.get("slots")
        if isinstance(raw_slots, Mapping):
            for slot_name, slot_payload in raw_slots.items():
                if not isinstance(slot_name, str) or not isinstance(slot_payload, Mapping):
                    continue
                prop_id = build_property_id(super_label, category_label, slot_name, inherited=False)
                slot = PropertySlot(
                    property_id=prop_id,
                    name=str(slot_payload.get("name") or slot_payload.get("label") or slot_name),
                    inherited=False,
                    **{k: v for k, v in slot_payload.items() if k not in {"name", "label"}},
                )
                slots[prop_id] = slot

        patterns: Dict[str, List[str]] = {}
        for prop_id, pattern_list in inherited_patterns.items():
            if isinstance(pattern_list, Iterable):
                patterns[prop_id] = [str(item) for item in pattern_list]

        raw_patterns = payload.get("patterns")
        if isinstance(raw_patterns, Mapping):
            for slot_name, pattern_list in raw_patterns.items():
                if not isinstance(slot_name, str):
                    continue
                prop_id = build_property_id(super_label, category_label, slot_name, inherited=False)
                values = [str(item) for item in pattern_list] if isinstance(pattern_list, Iterable) else []
                if values:
                    patterns[prop_id] = values

        plugins_block: Dict[str, List[str]] = {}
        raw_plugins = payload.get("plugins")
        if isinstance(raw_plugins, Mapping):
            for kind, entries in raw_plugins.items():
                if not isinstance(entries, Iterable):
                    continue
                plugins_block[str(kind)] = [str(item) for item in entries if isinstance(item, str)]

        for key_name, value in payload.items():
            if key_name in {"slots", "patterns"}:
                continue
            metadata[key_name] = value

        return cls(
            key=key,
            super=super_label,
            category=category_label,
            slots=slots,
            patterns=patterns,
            metadata=metadata,
            plugins=plugins_block,
        )


def merge_inherited_structures(
    *,
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
    super_label: str,
    category_label: str,
) -> CategoryDefinition:
    """Create a :class:`CategoryDefinition` merging global and category payloads."""

    inherited_slots: Dict[str, Mapping[str, Any]] = {}
    inherited_patterns: Dict[str, Iterable[str]] = {}

    if isinstance(base, Mapping):
        raw_slots = base.get("slots")
        if isinstance(raw_slots, Mapping):
            for slot_name, slot_payload in raw_slots.items():
                if not isinstance(slot_name, str) or not isinstance(slot_payload, Mapping):
                    continue
                prop_id = build_property_id(super_label, category_label, slot_name, inherited=True)
                inherited_slots[prop_id] = slot_payload
        raw_patterns = base.get("patterns")
        if isinstance(raw_patterns, Mapping):
            for slot_name, pattern_list in raw_patterns.items():
                if not isinstance(slot_name, str):
                    continue
                prop_id = build_property_id(super_label, category_label, slot_name, inherited=True)
                inherited_patterns[prop_id] = pattern_list

    payload: Dict[str, Any] = {}
    if isinstance(base, Mapping):
        payload.update({k: v for k, v in base.items() if k not in {"slots", "patterns"}})
    if isinstance(override, Mapping):
        payload.update(override)

    key = build_category_key(super_label, category_label)

    return CategoryDefinition.from_payload(
        key=key,
        super_label=super_label,
        category_label=category_label,
        payload=payload,
        inherited_slots=inherited_slots,
        inherited_patterns=inherited_patterns,
    )
