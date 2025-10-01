"""Validation helpers for schema-first property extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .fuse import CandidateSource
from .schema_registry import CategorySchema, PropertySpec, load_registry
from ..config import get_settings

__all__ = [
    "ALLOWED_SOURCES",
    "ValidationIssue",
    "ValidationResult",
    "PropertyPayload",
    "validate_properties",
]

ALLOWED_SOURCES: frozenset[str] = frozenset(source.value for source in CandidateSource)


class PropertyPayload(BaseModel):
    """Pydantic model describing the payload of a single property."""

    value: Any
    unit: str | None = None
    source: CandidateSource = Field(..., description="Origin of the extracted value")
    raw: str | None = None
    span: tuple[int, int] | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    evidence: str | None = None
    validation: Dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("source")
    @classmethod
    def _source_allowed(cls, value: CandidateSource | str) -> CandidateSource:
        try:
            resolved = CandidateSource(value)
        except ValueError as exc:  # pragma: no cover - handled via explicit tests
            raise ValueError(f"source '{value}' is not supported") from exc
        return resolved

    @field_validator("span")
    @classmethod
    def _span_format(cls, value: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError("span must contain exactly two offsets")
            value = (value[0], value[1])
        if len(value) != 2 or any(not isinstance(item, int) for item in value):
            raise ValueError("span must be a pair of integers")
        start, end = value
        if start < 0 or end < 0:
            raise ValueError("span offsets must be non-negative")
        if start > end:
            raise ValueError("span start must be less or equal to end")
        return (start, end)


@dataclass(frozen=True)
class ValidationIssue:
    property_id: str
    code: str
    message: str
    severity: str


@dataclass
class ValidationResult:
    category: CategorySchema
    normalized: Dict[str, PropertyPayload]
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]

    @property
    def ok(self) -> bool:  # pragma: no cover - trivial property
        return not self.errors


def _coerce_value(property_spec: PropertySpec, payload: PropertyPayload, errors: list[ValidationIssue]) -> Any:
    value = payload.value
    expected = property_spec.type
    prop_id = property_spec.id

    def emit(code: str, message: str) -> None:
        errors.append(ValidationIssue(prop_id, code, message, "error"))

    if value is None:
        return None

    try:
        if expected in {"number", "float"}:
            return float(value)
        if expected == "integer":
            return int(value)
        if expected == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "si", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            if isinstance(value, (int, float)):
                return bool(value)
            emit("type_boolean", f"Valore '{value}' non interpretabile come booleano")
            return None
        if expected in {"array", "list"}:
            if isinstance(value, (list, tuple)):
                return list(value)
            emit("type_array", f"Valore '{value}' non è una lista")
            return None
        # default to string semantics
        if expected in {"string", "text"}:
            if isinstance(value, str):
                return value
            return str(value)
    except (TypeError, ValueError) as exc:
        emit("type_conversion", f"Impossibile convertire '{value}' in {expected}: {exc}")
        return None

    return value


def _validate_enum(property_spec: PropertySpec, value: Any, errors: list[ValidationIssue]) -> None:
    if property_spec.enum and value is not None:
        allowed = set(property_spec.enum)
        if value not in allowed:
            errors.append(
                ValidationIssue(
                    property_spec.id,
                    "enum_mismatch",
                    f"Valore '{value}' non appartiene a {sorted(allowed)}",
                    "error",
                )
            )


def _validate_unit(property_spec: PropertySpec, payload: PropertyPayload, errors: list[ValidationIssue]) -> None:
    if property_spec.unit:
        if payload.unit is None:
            errors.append(
                ValidationIssue(
                    property_spec.id,
                    "unit_missing",
                    f"Unità attesa '{property_spec.unit}' mancante",
                    "error",
                )
            )
        elif payload.unit != property_spec.unit:
            errors.append(
                ValidationIssue(
                    property_spec.id,
                    "unit_mismatch",
                    f"Unità '{payload.unit}' diversa da '{property_spec.unit}'",
                    "error",
                )
            )


def _validate_required(category: CategorySchema, provided: Iterable[str], errors: list[ValidationIssue]) -> None:
    provided_set = set(provided)
    for required_id in category.required:
        if required_id not in provided_set:
            errors.append(
                ValidationIssue(required_id, "missing_required", "Proprietà obbligatoria assente", "error")
            )


def validate_properties(
    category_id: str,
    properties: Mapping[str, Mapping[str, Any]],
    *,
    registry_path: str | Path | None = None,
) -> ValidationResult:
    """Validate a property payload against the category schema."""

    effective_registry = registry_path or get_settings().registry_path
    registry = load_registry(effective_registry)
    category = registry.get(category_id)
    if category is None:
        raise ValueError(f"Categoria '{category_id}' non presente nel registry")

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    normalized: dict[str, PropertyPayload] = {}

    specs = {prop.id: prop for prop in category.properties}

    for prop_id, raw_payload in properties.items():
        spec = specs.get(prop_id)
        if spec is None:
            warnings.append(
                ValidationIssue(prop_id, "unknown_property", "Proprietà non definita nel registry", "warning")
            )
            continue
        try:
            payload = PropertyPayload.model_validate(raw_payload)
        except ValidationError as exc:  # pragma: no cover - exercised via tests
            for err in exc.errors():
                loc = ".".join(str(item) for item in err.get("loc", []))
                errors.append(
                    ValidationIssue(
                        prop_id,
                        "payload_invalid",
                        f"Errore campo '{loc}': {err.get('msg')}",
                        "error",
                    )
                )
            continue

        coerced = _coerce_value(spec, payload, errors)
        _validate_enum(spec, coerced, errors)
        _validate_unit(spec, payload, errors)

        if not any(issue.property_id == prop_id for issue in errors):
            normalized[prop_id] = PropertyPayload.model_validate({**payload.model_dump(), "value": coerced})

    _validate_required(category, properties.keys(), errors)

    return ValidationResult(category=category, normalized=normalized, errors=errors, warnings=warnings)
