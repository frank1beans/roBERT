"""Async orchestrator for parallel property extraction with LLM."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .fuse import Candidate, CandidateSource, Fuser
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .orchestrator_base import OrchestratorBase
from .parsers import dimensions, numbers
from .parsers.colors import parse_ral_colors
from .parsers.standards import parse_standards

from .qa_llm import AsyncHttpLLM
from .schema_registry import PropertySpec

LOGGER = logging.getLogger(__name__)

__all__ = ["AsyncOrchestrator"]


class AsyncOrchestrator(OrchestratorBase):
    """Async orchestrator for parallel LLM-based property extraction."""

    def __init__(self, fuse: Fuser, llm: AsyncHttpLLM, cfg: OrchestratorConfig) -> None:
        super().__init__(fuse=fuse, cfg=cfg)
        self._llm = llm

    async def extract_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties for a single document using async LLM calls."""

        text_id, category_id, text, property_specs, schema_properties = self._prepare_document(doc)

        properties_payload: Dict[str, Dict[str, Any]] = {}
        for prop_id, prop_schema in schema_properties.items():
            spec = property_specs.get(prop_id)
            result = await self._extract_property(text, category_id, prop_id, prop_schema, spec)
            properties_payload[prop_id] = result

        validation = self._validate_payload(category_id, properties_payload)

        return self._finalise_document(doc, text_id, category_id, properties_payload, validation)

    async def _extract_property(
        self,
        text: str,
        cat: str,
        prop: str,
        prop_schema: Dict[str, Any],
        prop_spec: Optional[PropertySpec] = None,
    ) -> Dict[str, Any]:
        allowed_sources, candidates = self._deterministic_candidates(cat, prop, prop_spec, text)

        if self._llm and "qa_llm" in allowed_sources:
            llm_candidate = await self._llm_candidate(prop, text, prop_schema)
            if llm_candidate:
                candidates.append(llm_candidate)

        validator = self._build_validator(prop_spec)
        fused = self._fuse.fuse(candidates, validator)

        return self._build_property_payload(cat, prop, candidates, fused)

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
                },
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
