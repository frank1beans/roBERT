"""Property extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional

from .registry import PropertyGroup, PropertyRegistry, PropertySlot

__all__ = ["PropertyExtractionResult", "RegexPropertyExtractor"]


@dataclass(slots=True)
class PropertyExtractionResult:
    group_key: str
    slot: PropertySlot
    value: str
    span: Optional[tuple[int, int]] = None
    pattern: Optional[str] = None


class RegexPropertyExtractor:
    """Simple regex-based property extractor aware of the registry schema."""

    def __init__(self, registry: PropertyRegistry, flags: int = re.IGNORECASE):
        self.registry = registry
        self.flags = flags
        self._compiled: Dict[str, Dict[str, List[re.Pattern]]] = {}

    def _get_group(self, key: str) -> PropertyGroup:
        group = self.registry.get(key)
        if group is None:
            raise KeyError(f"Unknown property group '{key}'")
        return group

    def _compiled_for(self, key: str) -> Dict[str, List[re.Pattern]]:
        if key not in self._compiled:
            self._compiled[key] = self._get_group(key).compile_patterns(self.flags)
        return self._compiled[key]

    def extract(self, text: str, group_key: str, slots: Optional[Iterable[str]] = None) -> List[PropertyExtractionResult]:
        group = self._get_group(group_key)
        compiled = self._compiled_for(group_key)
        results: List[PropertyExtractionResult] = []

        slot_names = set(slots) if slots else set(group.slots.keys())
        for slot_name in slot_names:
            slot = group.slots.get(slot_name)
            if slot is None:
                continue
            patterns = compiled.get(slot_name, [])
            if not patterns:
                continue
            for pattern in patterns:
                for match in pattern.finditer(text):
                    raw_value = match.group(1) if match.groups() else match.group(0)
                    normalized = slot.normalize(raw_value)
                    results.append(
                        PropertyExtractionResult(
                            group_key=group_key,
                            slot=slot,
                            value=normalized,
                            span=match.span(),
                            pattern=pattern.pattern,
                        )
                    )
                    # prefer first match per slot to avoid duplicates
                    break
                else:
                    continue
                break
        # fallback for slots without patterns: assign unknown token
        for slot_name in slot_names:
            if any(res.slot.name == slot_name for res in results):
                continue
            slot = group.slots.get(slot_name)
            if slot is None:
                continue
            results.append(
                PropertyExtractionResult(
                    group_key=group_key,
                    slot=slot,
                    value=slot.unknown_token or "",
                    span=None,
                    pattern=None,
                )
            )
        # enforce priority ordering if available
        priority = group.priority
        if priority:
            results.sort(key=lambda item: priority.index(item.slot.name) if item.slot.name in priority else len(priority))
        return results
