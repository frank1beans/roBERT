"""Final normalisation and validation hooks shared across router clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

from robimb.registry import CategoryDefinition, validate

__all__ = ["PostProcessResult", "apply_postprocess"]

NUMERIC_TYPES = {"float", "int", "number"}
BOOL_TYPES = {"bool", "boolean"}


@dataclass
class PostProcessResult:
    values: Dict[str, Any]
    issues: Optional[list[Dict[str, Any]]] = None


def _coerce_numeric(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return value


def _coerce_boolean(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "si", "sì", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _apply_schema(values: MutableMapping[str, Any], category: CategoryDefinition) -> None:
    for property_id, slot in category.slots.items():
        if property_id not in values:
            continue
        value = values[property_id]
        slot_type = slot.type or ""
        if slot_type in NUMERIC_TYPES:
            if isinstance(value, list):
                values[property_id] = [_coerce_numeric(item) for item in value]
            else:
                values[property_id] = _coerce_numeric(value)
        elif slot_type in BOOL_TYPES:
            if isinstance(value, list):
                values[property_id] = [_coerce_boolean(item) for item in value]
            else:
                values[property_id] = _coerce_boolean(value)


def apply_postprocess(
    values: Mapping[str, Any],
    *,
    category: Optional[CategoryDefinition] = None,
    validators: Optional[Mapping[str, Any]] = None,
    category_label: str = "",
    context: Optional[Mapping[str, Any]] = None,
    cat_entry: Optional[Mapping[str, Any]] = None,
) -> PostProcessResult:
    """Apply schema-driven normalisation and validator rules."""

    normalized: Dict[str, Any] = dict(values)
    if category is not None:
        _apply_schema(normalized, category)

    issues: Optional[list[Dict[str, Any]]] = None
    if validators is not None:
        issues = validate(category_label, normalized, context or {}, validators, cat_entry=cat_entry)

    return PostProcessResult(values=normalized, issues=issues)

