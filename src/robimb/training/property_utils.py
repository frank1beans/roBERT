"""Utility helpers to deal with property prediction targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["PropertyMetadata", "build_property_metadata", "build_property_targets"]


SUPPORTED_SCHEMA_TYPES = {"float", "int", "number", "enum", "bool", "boolean"}
NUMERIC_TYPES = {"float", "int", "number"}


@dataclass
class PropertyMetadata:
    """Aggregated information about property slots present in the dataset."""

    slot_to_idx: Mapping[str, int]
    slot_names: Sequence[str]
    numeric_mask: np.ndarray
    cat_property_mask: np.ndarray

    @property
    def num_slots(self) -> int:
        return len(self.slot_names)

    def has_properties(self) -> bool:
        return self.num_slots > 0


def _normalise_schema_type(raw: Optional[Mapping[str, object]]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.get("type") if isinstance(raw, Mapping) else None
    if not isinstance(value, str):
        return None
    lowered = value.lower().strip()
    if lowered not in SUPPORTED_SCHEMA_TYPES:
        return None
    return lowered


def build_property_metadata(
    dataframes: Iterable[pd.DataFrame],
    num_cat: int,
) -> PropertyMetadata:
    """Collect slot metadata from the provided dataframes."""

    slot_to_idx: Dict[str, int] = {}
    slot_names: List[str] = []
    numeric_mask: List[bool] = []
    cat_to_slots: List[set[int]] = [set() for _ in range(num_cat)]

    for df in dataframes:
        if "property_schema" not in df.columns or "cat_label" not in df.columns:
            continue
        for cat_label, schema in zip(df["cat_label"], df["property_schema"]):
            if not isinstance(schema, Mapping):
                continue
            slots = schema.get("slots")
            if not isinstance(slots, Mapping):
                continue
            try:
                cat_idx = int(cat_label)
            except (TypeError, ValueError):
                continue
            if cat_idx < 0 or cat_idx >= num_cat:
                continue
            for slot_name, slot_schema in slots.items():
                if not isinstance(slot_name, str):
                    continue
                norm_type = _normalise_schema_type(slot_schema)
                if norm_type is None:
                    continue
                if slot_name not in slot_to_idx:
                    slot_to_idx[slot_name] = len(slot_names)
                    slot_names.append(slot_name)
                    numeric_mask.append(norm_type in NUMERIC_TYPES)
                idx = slot_to_idx[slot_name]
                cat_to_slots[cat_idx].add(idx)

    if not slot_names:
        empty_mask = np.zeros((num_cat, 0), dtype=np.float32)
        return PropertyMetadata({}, [], np.zeros(0, dtype=bool), empty_mask)

    cat_property_mask = np.zeros((num_cat, len(slot_names)), dtype=np.float32)
    for cat_idx, slot_indices in enumerate(cat_to_slots):
        if not slot_indices:
            continue
        cat_property_mask[cat_idx, list(slot_indices)] = 1.0

    return PropertyMetadata(
        slot_to_idx=slot_to_idx,
        slot_names=slot_names,
        numeric_mask=np.asarray(numeric_mask, dtype=bool),
        cat_property_mask=cat_property_mask,
    )


def _clean_value(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        stripped = stripped.replace(",", ".")
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def build_property_targets(
    batch_properties: Sequence[MutableMapping[str, object] | Mapping[str, object] | None],
    batch_cat_labels: Sequence[int],
    metadata: PropertyMetadata,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create masks and targets for property prediction."""

    num_props = metadata.num_slots
    batch_size = len(batch_cat_labels)
    mask = np.zeros((batch_size, num_props), dtype=np.float32)
    presence = np.zeros((batch_size, num_props), dtype=np.float32)
    regression_targets = np.zeros((batch_size, num_props), dtype=np.float32)
    regression_mask = np.zeros((batch_size, num_props), dtype=np.float32)

    if num_props == 0:
        return mask, presence, regression_targets, regression_mask

    numeric_mask = metadata.numeric_mask
    cat_mask = metadata.cat_property_mask

    for row_idx, cat_label in enumerate(batch_cat_labels):
        try:
            cat_id = int(cat_label)
        except (TypeError, ValueError):
            continue
        if cat_id < 0 or cat_id >= cat_mask.shape[0]:
            continue
        allowed = cat_mask[cat_id]
        mask[row_idx] = allowed

        props = batch_properties[row_idx] if row_idx < len(batch_properties) else None
        if not isinstance(props, Mapping):
            continue
        for prop_name, raw_value in props.items():
            idx = metadata.slot_to_idx.get(prop_name)
            if idx is None:
                continue
            if allowed[idx] == 0:
                continue
            presence[row_idx, idx] = 1.0
            if numeric_mask[idx]:
                value = _clean_value(raw_value)
                if value is not None:
                    regression_targets[row_idx, idx] = value
                    regression_mask[row_idx, idx] = 1.0

    return mask, presence, regression_targets, regression_mask
