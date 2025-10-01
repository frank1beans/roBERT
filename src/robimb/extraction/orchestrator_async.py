"""Async orchestrator for parallel property extraction with LLM."""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .fuse import Candidate, CandidateSource, Fuser
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .orchestrator import OrchestratorConfig
from .parsers import dimensions, numbers
from .parsers.colors import parse_ral_colors
from .parsers.standards import parse_standards
from .qa_llm import AsyncHttpLLM
from .schema_registry import PropertySpec, load_category_schema, load_registry
from .validators import validate_properties
from ..registry.schemas import slugify

LOGGER = logging.getLogger(__name__)

__all__ = ["AsyncOrchestrator"]


class AsyncOrchestrator:
    """Async orchestrator for parallel LLM-based property extraction."""

    def __init__(self, fuse: Fuser, llm: AsyncHttpLLM, cfg: OrchestratorConfig) -> None:
        self._fuse = fuse
        self._llm = llm
        self._cfg = cfg
        self._brand_matcher = BrandMatcher()
        self._material_matcher = MaterialMatcher()

    async def extract_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties for a single document using async LLM calls."""
        from .orchestrator import Orchestrator

        text_id = Orchestrator._resolve_text_id(None, doc)
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
            result = await self._extract_property(text, category_id, prop_id, prop_schema, spec)
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

        result = {
            **doc,  # Include all original fields
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
        return result

    async def _extract_property(
        self,
        text: str,
        cat: str,
        prop: str,
        prop_schema: Dict[str, Any],
        prop_spec: Optional[PropertySpec] = None,
    ) -> Dict[str, Any]:
        from .orchestrator import Orchestrator

        allowed_sources = self._determine_sources(prop_spec)
        candidates: List[Candidate] = []

        if "parser" in allowed_sources:
            candidates.extend(self._parser_candidates(prop, prop_spec, text))

        if self._cfg.enable_matcher and "matcher" in allowed_sources:
            candidates.extend(self._matcher_candidates(cat, prop, text))

        if self._llm and "qa_llm" in allowed_sources:
            llm_candidate = await self._llm_candidate(prop, text, prop_schema)
            if llm_candidate:
                candidates.append(llm_candidate)

        validator = self._build_validator(prop_spec)
        fused = self._fuse.fuse(candidates, validator)

        span = Orchestrator._normalize_span(None, fused.get("span"))
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

    async def _llm_candidate(self, prop_id: str, text: str, prop_schema: Dict[str, Any]) -> Optional[Candidate]:
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
            response = await self._llm.ask(text, question, llm_schema)
            LOGGER.info(
                "llm_response",
                extra={
                    "property": prop_id,
                    "text_preview": text[:100] if len(text) > 100 else text,
                    "response": response,
                    "question": question,
                    "schema": llm_schema,
                }
            )
        except Exception as exc:
            LOGGER.warning("llm_error", extra={"property": prop_id, "error": str(exc)})
            return None
        value = response.get("value")
        confidence = response.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.60
        span = response.get("span")
        candidate: Candidate = Candidate(
            value=value,
            source=CandidateSource.QA_LLM,
            raw=response.get("raw"),
            span=span if isinstance(span, (list, tuple)) else None,
            confidence=confidence_value,
            unit=response.get("unit"),
            errors=list(response.get("errors", [])) if isinstance(response.get("errors"), list) else [],
        )
        return candidate

    # Reuse sync methods from Orchestrator
    def _determine_sources(self, spec: Optional[PropertySpec]) -> Sequence[str]:
        if spec and spec.sources:
            return [source for source in spec.sources if source in self._cfg.source_priority]
        return list(self._cfg.source_priority)

    def _parser_candidates(
        self, prop_id: str, spec: Optional[PropertySpec], text: str
    ) -> Iterable[Candidate]:
        from .orchestrator import Orchestrator
        return Orchestrator._parser_candidates(self, prop_id, spec, text)

    def _matcher_candidates(self, category: str, prop_id: str, text: str) -> Iterable[Candidate]:
        from .orchestrator import Orchestrator
        return Orchestrator._matcher_candidates(self, category, prop_id, text)

    def _build_validator(self, spec: Optional[PropertySpec]):
        from .orchestrator import Orchestrator
        return Orchestrator._build_validator(self, spec)

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
