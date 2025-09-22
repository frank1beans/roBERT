"""Utilities to work with the extended property registry."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple


__all__ = [
    "PropertySlot",
    "PropertyGroup",
    "PropertyRegistry",
]


@dataclass(slots=True)
class PropertySlot:
    """Represents a single property slot within a group."""

    name: str
    slot_type: str
    description: Optional[str] = None
    enum_values: Optional[List[str]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    range: Optional[Tuple[float, float]] = None
    unknown_token: Optional[str] = None
    unit: Optional[str] = None

    def normalize(self, value: str) -> str:
        """Normalize raw text according to the slot definition."""

        value = value.strip()
        if not value:
            return self.unknown_token or value
        if self.slot_type == "text":
            return value
        if self.slot_type == "bool":
            lowered = value.lower()
            if lowered in {"true", "t", "si", "sì", "yes", "1"}:
                return "true"
            if lowered in {"false", "f", "no", "0"}:
                return "false"
            return self.unknown_token or value
        if self.slot_type == "enum" and self.enum_values:
            lowered = value.lower()
            for choice in self.enum_values:
                if lowered == choice.lower():
                    return choice
            return self.unknown_token or value
        # numeric values
        cleaned = value.replace(",", ".")
        try:
            num = float(cleaned)
        except ValueError:
            return self.unknown_token or value
        if self.slot_type == "int":
            return str(int(round(num)))
        return str(num)


@dataclass(slots=True)
class PropertyGroup:
    """A group identifies the Super|Cat combination in the registry."""

    key: str
    priority: List[str] = field(default_factory=list)
    slots: Dict[str, PropertySlot] = field(default_factory=dict)
    patterns: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)

    def compile_patterns(self, flags: int = re.IGNORECASE) -> Dict[str, List[re.Pattern]]:
        compiled: Dict[str, List[re.Pattern]] = {}
        for slot_name, pat_list in self.patterns.items():
            compiled[slot_name] = [re.compile(pattern, flags) for pattern in pat_list]
        return compiled

    def iter_slots(self) -> Iterator[PropertySlot]:
        return iter(self.slots.values())

    def add_slot(self, slot: PropertySlot) -> None:
        self.slots[slot.name] = slot


class PropertyRegistry:
    """Collection of property groups keyed by "Super|Cat"."""

    def __init__(self, schema: Mapping[str, object], groups: Mapping[str, PropertyGroup]):
        self._schema = dict(schema)
        self._groups = dict(groups)

    @property
    def schema(self) -> Mapping[str, object]:
        return self._schema

    def __contains__(self, key: str) -> bool:
        return key in self._groups

    def __len__(self) -> int:
        return len(self._groups)

    def keys(self) -> Iterable[str]:
        return self._groups.keys()

    def get(self, key: str) -> Optional[PropertyGroup]:
        return self._groups.get(key)

    def iter_groups(self) -> Iterator[PropertyGroup]:
        return iter(self._groups.values())

    @classmethod
    def from_json(cls, path: str | Path) -> "PropertyRegistry":
        raw = json.load(open(path, "r", encoding="utf-8"))
        schema = raw.get("_schema", {})
        groups: Dict[str, PropertyGroup] = {}
        for key, value in raw.items():
            if key.startswith("_"):
                continue
            groups[key] = cls._build_group(key, value, schema)
        return cls(schema=schema, groups=groups)

    @staticmethod
    def _build_group(key: str, data: Mapping[str, object], schema: Mapping[str, object]) -> PropertyGroup:
        priority = list(data.get("priority", []))
        slots: Dict[str, PropertySlot] = {}
        raw_slots = data.get("slots", {})
        unknown_token = schema.get("unknown_token") if isinstance(schema, Mapping) else None
        for slot_name, slot_data in raw_slots.items():
            slot = PropertySlot(
                name=slot_name,
                slot_type=str(slot_data.get("type", "text")),
                description=slot_data.get("description"),
                enum_values=list(slot_data.get("values", []) or []),
                minimum=slot_data.get("min"),
                maximum=slot_data.get("max"),
                range=tuple(slot_data.get("range", [])) if slot_data.get("range") else None,
                unknown_token=slot_data.get("unknown", unknown_token),
                unit=slot_data.get("unit"),
            )
            slots[slot_name] = slot
        patterns = {
            slot_name: list(patterns)
            for slot_name, patterns in (data.get("patterns", {}) or {}).items()
        }
        metadata = {
            key: str(value)
            for key, value in (data.get("metadata", {}) or {}).items()
        }
        return PropertyGroup(key=key, priority=priority, slots=slots, patterns=patterns, metadata=metadata)

    def merge(self, other: "PropertyRegistry", override: bool = True) -> None:
        """Merge another registry in-place."""

        if override:
            self._schema.update(other.schema)
        else:
            for key, value in other.schema.items():
                self._schema.setdefault(key, value)

        for key, group in other._groups.items():
            if key not in self._groups or override:
                self._groups[key] = group
            else:
                base = self._groups[key]
                base.priority = list(dict.fromkeys(base.priority + group.priority))
                base.patterns.update(group.patterns)
                for slot in group.iter_slots():
                    base.slots.setdefault(slot.name, slot)

    def to_json(self, path: str | Path) -> None:
        data: Dict[str, object] = {"_schema": self._schema}
        for key, group in self._groups.items():
            slots = {}
            for slot_name, slot in group.slots.items():
                slot_data = {
                    "type": slot.slot_type,
                }
                if slot.description:
                    slot_data["description"] = slot.description
                if slot.enum_values:
                    slot_data["values"] = slot.enum_values
                if slot.minimum is not None:
                    slot_data["min"] = slot.minimum
                if slot.maximum is not None:
                    slot_data["max"] = slot.maximum
                if slot.range is not None:
                    slot_data["range"] = list(slot.range)
                if slot.unknown_token is not None:
                    slot_data["unknown"] = slot.unknown_token
                if slot.unit:
                    slot_data["unit"] = slot.unit
                slots[slot_name] = slot_data
            entry = {
                "priority": group.priority,
                "slots": slots,
                "patterns": group.patterns,
            }
            if group.metadata:
                entry["metadata"] = group.metadata
            data[key] = entry
        json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
