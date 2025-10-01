"""Orchestrates multi-strategy property extraction with cascading fallbacks."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from .fuse import Candidate, Fuser
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .parsers import dimensions, numbers
from .parsers.colors import parse_ral_colors
from .parsers.standards import parse_standards
from .qa_llm import QALLM
from ..registry.schemas import slugify
from .schema_registry import PropertySpec, load_category_schema, load_registry
from .validators import validate_properties

LOGGER = logging.getLogger(__name__)

__all__ = ["Orchestrator", "OrchestratorConfig"]


class OrchestratorConfig(BaseModel):
    """Configuration for the property extraction orchestrator."""

    source_priority: List[str] = Field(default_factory=lambda: ["parser", "matcher", "qa_llm"])
    enable_matcher: bool = True
    enable_llm: bool = True
    registry_path: str = "data/properties/registry.json"


class Orchestrator:
    """Coordinate deterministic parsers, matchers and LLM fallbacks."""

    def __init__(self, fuse: Fuser, llm: Optional[QALLM], cfg: OrchestratorConfig) -> None:
        self._fuse = fuse
        self._llm = llm if cfg.enable_llm else None
        self._cfg = cfg
        self._brand_matcher = BrandMatcher()
        self._material_matcher = MaterialMatcher()

    def extract_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties for a single document."""

        text_id = self._resolve_text_id(doc)
        category_id = self._resolve_category_id(doc)
        text = doc.get("text", "") or ""

        if not category_id:
            raise ValueError("Input document is missing category information")

        try:
            category, schema = load_category_schema(category_id, registry_path=self._cfg.registry_path)
        except ValueError:
            alias = self._resolve_category_alias(doc, category_id)
            if not alias:
                raise
            category_id = alias
            category, schema = load_category_schema(category_id, registry_path=self._cfg.registry_path)
        property_specs = {prop.id: prop for prop in category.properties}
        schema_properties = self._resolve_schema_properties(schema)

        properties_payload: Dict[str, Dict[str, Any]] = {}
        for prop_id, prop_schema in schema_properties.items():
            spec = property_specs.get(prop_id)
            result = self._extract_property(text, category_id, prop_id, prop_schema, spec)
            properties_payload[prop_id] = result

        validation_input = {
            prop_id: {
                "value": payload["value"],
                "unit": payload.get("unit"),
                "source": payload["source"],
                "raw": payload.get("raw"),
                "span": payload.get("span"),
                "confidence": payload.get("confidence"),
            }
            for prop_id, payload in properties_payload.items()
            if payload.get("source")
        }

        validation = validate_properties(
            category_id,
            validation_input,
            registry_path=self._cfg.registry_path,
        )
        for issue in validation.errors:
            result = properties_payload.setdefault(
                issue.property_id,
                {
                    "value": None,
                    "source": None,
                    "unit": None,
                    "raw": None,
                    "span": None,
                    "confidence": 0.0,
                    "errors": [],
                },
            )
            result.setdefault("errors", []).append(issue.message)

        confidence_values = [
            float(payload.get("confidence") or 0.0)
            for payload in properties_payload.values()
            if payload.get("value") is not None
        ]
        confidence_overall = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

        LOGGER.info(
            "document_processed",
            extra={
                "text_id": text_id,
                "category": category_id,
                "confidence_overall": confidence_overall,
                "validation_status": "ok" if validation.ok else "failed",
            },
        )

        return {
            "text_id": text_id,
            "categoria": category_id,
            "properties": properties_payload,
            "validation": {
                "status": "ok" if validation.ok else "failed",
                "errors": [
                    {
                        "property_id": issue.property_id,
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in validation.errors
                ],
            },
            "confidence_overall": confidence_overall,
        }

    def _extract_property(
        self,
        text: str,
        cat: str,
        prop: str,
        prop_schema: Dict[str, Any],
        prop_spec: Optional[PropertySpec] = None,
    ) -> Dict[str, Any]:
        allowed_sources = self._determine_sources(prop_spec)
        candidates: List[Candidate] = []

        if "parser" in allowed_sources:
            candidates.extend(self._parser_candidates(prop, prop_spec, text))

        if self._cfg.enable_matcher and "matcher" in allowed_sources:
            candidates.extend(self._matcher_candidates(prop, text))

        if self._llm and "qa_llm" in allowed_sources:
            llm_candidate = self._llm_candidate(prop, text, prop_schema)
            if llm_candidate:
                candidates.append(llm_candidate)

        validator = self._build_validator(prop_spec)
        fused = self._fuse.fuse(candidates, validator)

        span = self._normalize_span(fused.get("span"))
        unit = fused.get("unit")
        errors = list(fused.get("errors", []))
        confidence = float(fused.get("confidence") or 0.0)
        source = fused.get("source")
        value = fused.get("value")
        raw = fused.get("raw")

        result: Dict[str, Any] = {
            "value": value,
            "source": source,
            "raw": raw,
            "span": span,
            "confidence": confidence,
            "unit": unit,
            "errors": errors,
        }

        LOGGER.info(
            "property_fused",
            extra={
                "category": cat,
                "property": prop,
                "candidates": candidates,
                "selected": result,
            },
        )

        return result

    def _determine_sources(self, spec: Optional[PropertySpec]) -> Sequence[str]:
        if spec and spec.sources:
            return [source for source in spec.sources if source in self._cfg.source_priority]
        return list(self._cfg.source_priority)

    def _normalize_span(self, span: Any) -> Optional[List[int]]:
        if span is None:
            return None
        if isinstance(span, list):
            return [int(span[0]), int(span[1])]
        if isinstance(span, tuple):
            return [int(span[0]), int(span[1])]
        raise TypeError(f"Unsupported span type: {type(span)!r}")

    def _resolve_text_id(self, doc: Dict[str, Any]) -> Optional[str]:
        for key in ("text_id", "id", "document_id", "doc_id", "uuid"):
            value = doc.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, (int, float)):
                return str(value)
        return None

    def _resolve_category_id(self, doc: Dict[str, Any]) -> Optional[str]:

        registry = load_registry(self._cfg.registry_path)
        for key in (
            "categoria",
            "cat",
            "category",
            "category_id",
            "categoria_id",
            "categoria_label",
            "cat_label",
        ):
            value = doc.get(key)
            resolved = self._match_category_value(value, registry)
            if resolved:
                return resolved

        for key in (
            "super",
            "supercategoria",
            "macro_categoria",
            "macrocategory",
            "macro",
        ):
            value = doc.get(key)
            resolved = self._match_category_value(value, registry)
            if resolved:
                return resolved

        schema_hint = self._resolve_category_from_schema(doc, registry)
        if schema_hint:
            return schema_hint

        return None

    def _match_category_value(self, value: Any, registry) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            value = str(int(value)) if float(value).is_integer() else str(value)
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate:
            return None

        if candidate in registry.categories:
            return candidate

        lowered = candidate.lower()
        for schema in registry.list():
            if schema.id.lower() == lowered or schema.name.lower() == lowered:
                return schema.id

        slug = slugify(candidate)
        for schema in registry.list():
            if slugify(schema.id) == slug or slugify(schema.name) == slug:
                return schema.id

        return None

    def _resolve_category_from_schema(self, doc: Dict[str, Any], registry) -> Optional[str]:
        hints: List[str] = []
        for key in ("property_schema", "properties"):
            value = doc.get(key)
            if not isinstance(value, dict):
                continue
            hints.extend([k for k in value.keys() if isinstance(k, str)])
            if key == "property_schema":
                slots = value.get("slots")
                if isinstance(slots, dict):
                    hints.extend([k for k in slots.keys() if isinstance(k, str)])
                metadata = value.get("metadata")
                if isinstance(metadata, dict):
                    slot_meta = metadata.get("slots")
                    if isinstance(slot_meta, dict):
                        hints.extend([k for k in slot_meta.keys() if isinstance(k, str)])
        for hint in hints:
            prefix = hint.split(".", 1)[0]
            resolved = self._match_category_value(prefix, registry)
            if resolved:
                return resolved
        return None

    def _resolve_category_alias(self, doc: Dict[str, Any], original: Optional[str]) -> Optional[str]:
        registry = load_registry(self._cfg.registry_path)
        candidates: List[Any] = []
        if original:
            candidates.append(original)
        candidates.extend(
            doc.get(key)
            for key in (
                "categoria",
                "cat",
                "category",
                "category_id",
                "categoria_id",
                "super",
                "supercategoria",
                "macro_categoria",
                "macrocategory",
                "macro",
            )
        )
        for candidate in candidates:
            resolved = self._match_category_value(candidate, registry)
            if resolved:
                return resolved
        return self._resolve_category_from_schema(doc, registry)


    def _parser_candidates(
        self, prop_id: str, spec: Optional[PropertySpec], text: str
    ) -> Iterable[Candidate]:
        results: List[Candidate] = []
        lowered = prop_id.lower()

        if any(token in lowered for token in ("dimension", "formato")):
            for match in dimensions.parse_dimensions(text):
                values = match.values_mm
                if not values:
                    continue

                # Improved dimension assignment heuristics
                if "lunghezza" in lowered or "length" in lowered:
                    # For length: prefer the largest value (typically the first in WxHxD format)
                    selected = max(values) if len(values) > 1 else values[0]
                elif "larghezza" in lowered or "width" in lowered:
                    # For width: first value in 2D (WxH), or first in 3D (WxHxD)
                    selected = values[0]
                elif "altezza" in lowered or "height" in lowered:
                    # For height: second value in 2D (WxH), third in 3D door format (WxHxD), or second in typical 3D
                    if len(values) == 2:
                        selected = values[1]
                    elif len(values) >= 3:
                        # If one value is significantly larger (e.g., door height 2100mm vs 800mm width),
                        # that's likely the height
                        if max(values) > 1500 and max(values) / min(values) > 2:
                            selected = max(values)
                        else:
                            selected = values[1]
                    else:
                        selected = values[0]
                elif "profond" in lowered or "depth" in lowered:
                    # For depth: third value in 3D, or smallest in set
                    if len(values) >= 3:
                        selected = values[2]
                    elif len(values) == 2:
                        selected = min(values)
                    else:
                        selected = None
                else:
                    # Generic dimension property: return all values as structured dict
                    keys = ["width_mm", "height_mm", "depth_mm"]
                    selected = {key: values[idx] for idx, key in enumerate(keys) if idx < len(values)}

                if selected is None:
                    continue
                results.append(
                    Candidate(
                        value=selected,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.90,
                        unit="mm" if isinstance(selected, (int, float)) else None,
                        errors=[],
                    )
                )

        elif any(token in lowered for token in ("spessore", "spessori")):
            for match in numbers.extract_numbers(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=(match.start, match.end),
                        confidence=0.90,
                        unit="mm",
                        errors=[],
                    )
                )

        elif "ral" in lowered or "colore" in lowered:
            for match in parse_ral_colors(text):
                raw = text[match.span[0] : match.span[1]]
                results.append(
                    Candidate(
                        value=match.code,
                        source="parser",
                        raw=raw,
                        span=match.span,
                        confidence=0.85,
                        unit=None,
                        errors=[],
                    )
                )

        elif "norma" in lowered or "standard" in lowered:
            for match in parse_standards(text):
                raw = text[match.span[0] : match.span[1]]
                label = match.prefix
                if match.code:
                    label += f" {match.code}"
                if match.year:
                    label += f":{match.year}"
                results.append(
                    Candidate(
                        value=label,
                        source="parser",
                        raw=raw,
                        span=match.span,
                        confidence=0.80,
                        unit=None,
                        errors=[],
                    )
                )

        elif "db" in lowered or "decibel" in lowered:
            for match in numbers.extract_numbers(text):
                window = text[match.end : min(len(text), match.end + 5)].lower()
                previous = text[max(0, match.start - 5) : match.start].lower()
                if "db" not in window and "db" not in previous:
                    continue
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=(match.start, match.end),
                        confidence=0.90,
                        unit="db",
                        errors=[],
                    )
                )

        return results

    def _matcher_candidates(self, prop_id: str, text: str) -> Iterable[Candidate]:
        lowered = prop_id.lower()
        results: List[Candidate] = []
        if lowered == "marchio":
            for brand, span, score in self._brand_matcher.find(text):
                results.append(
                    Candidate(
                        value=brand,
                        source="matcher",
                        raw=text[span[0] : span[1]],
                        span=span,
                        confidence=0.70 * float(score),
                        unit=None,
                        errors=[],
                    )
                )
        if "material" in lowered:
            for match in self._material_matcher.find(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="matcher",
                        raw=match.surface,
                        span=match.span,
                        confidence=0.65 * float(match.score),
                        unit=None,
                        errors=[],
                    )
                )
        return results

    def _llm_candidate(self, prop_id: str, text: str, prop_schema: Dict[str, Any]) -> Optional[Candidate]:
        value_schema = self._extract_value_schema(prop_schema)
        llm_schema = {
            "type": "object",
            "properties": {
                "value": value_schema,
                "confidence": {
                    "type": ["number", "null"],
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        }
        question = f"Estrai il valore della proprietà '{prop_id}'."
        try:
            response = self._llm.ask(text, question, llm_schema)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("llm_error", extra={"property": prop_id, "error": str(exc)})
            return None
        value = response.get("value")
        confidence = response.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.60
        span = response.get("span")
        candidate: Candidate = Candidate(
            value=value,
            source="qa_llm",
            raw=response.get("raw"),
            span=span if isinstance(span, (list, tuple)) else None,
            confidence=confidence_value,
            unit=response.get("unit"),
            errors=list(response.get("errors", [])) if isinstance(response.get("errors"), list) else [],
        )
        return candidate

    def _build_validator(self, spec: Optional[PropertySpec]):
        def _validator(candidate: Candidate) -> tuple[bool, List[str]]:
            errors: List[str] = []
            if spec is None:
                return True, errors
            value = candidate.get("value")
            if value is None:
                errors.append("value_missing")
                return False, errors
            expected = (spec.type or "string").lower()
            if expected in {"number", "float"} and not isinstance(value, (int, float)):
                errors.append("expected_number")
            elif expected == "integer" and not isinstance(value, int):
                errors.append("expected_integer")
            elif expected == "boolean" and not isinstance(value, bool):
                errors.append("expected_boolean")
            elif expected in {"array", "list"} and not isinstance(value, (list, tuple)):
                errors.append("expected_array")
            elif expected in {"object", "dict"} and not isinstance(value, dict):
                errors.append("expected_object")
            elif expected in {"string", "text"} and not isinstance(value, str):
                errors.append("expected_string")

            if spec.enum and value not in spec.enum:
                errors.append("enum_mismatch")

            if spec.unit and candidate.get("unit") not in {spec.unit, None}:
                errors.append("unit_mismatch")

            return (not errors), errors

        return _validator

    def _resolve_schema_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        props = schema.get("properties", {})
        return props.get("properties", {}).get("properties", {})

    def _extract_value_schema(self, prop_schema: Dict[str, Any]) -> Dict[str, Any]:
        if "properties" in prop_schema and "value" in prop_schema["properties"]:
            return prop_schema["properties"]["value"]
        all_of = prop_schema.get("allOf", [])
        for entry in all_of:
            if isinstance(entry, dict) and "properties" in entry and "value" in entry["properties"]:
                return entry["properties"]["value"]
        return {"type": ["string", "number", "boolean", "null", "object", "array"]}
