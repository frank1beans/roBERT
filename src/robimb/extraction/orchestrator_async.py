"""Async orchestrator for parallel property extraction with LLM."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .fuse import Candidate, CandidateSource, Fuser
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .domain_heuristics import post_process_properties
from .orchestrator_base import OrchestratorBase, OrchestratorConfig
from .parsers import dimensions, numbers
from .parsers.colors import parse_ral_colors
from .parsers.standards import parse_standards
from .parsers.flow_rate import parse_flow_rate
from .parsers.labeled_dimensions import parse_labeled_dimensions
from .parsers.acoustic import parse_acoustic_coefficient
from .parsers.fire_class import parse_fire_class
from .parsers.thickness import parse_thickness
from .parsers.installation_type import parse_installation_type

from .qa_llm import AsyncHttpLLM
from .schema_registry import PropertySpec

LOGGER = logging.getLogger(__name__)

__all__ = ["AsyncOrchestrator"]

PROPERTY_EXTRA_HINTS = {
    "trasmittanza_termica": "Cerca valori di trasmittanza (Uw, Uf, Ug) espressi in W/m²K.",
    "isolamento_acustico_db": "Cerca valori di isolamento acustico (Rw, Ra) espressi in dB.",
}


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

        post_process_properties(text, category_id, properties_payload, logger=LOGGER)

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
            llm_candidate = await self._llm_candidate(prop, text, prop_schema, prop_spec)
            if llm_candidate:
                candidates.append(llm_candidate)

        validator = self._build_validator(prop_spec)
        fused = self._fuse.fuse(candidates, validator)

        return self._build_property_payload(cat, prop, candidates, fused)

    async def _llm_candidate(
        self,
        prop_id: str,
        text: str,
        prop_schema: Dict[str, Any],
        prop_spec: Optional[PropertySpec],
    ) -> Optional[Candidate]:
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
                "raw": {"type": ["string", "null"]},
                "span": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "unit": {"type": ["string", "null"]},
                "errors": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        }
        display_name = prop_spec.title if prop_spec and getattr(prop_spec, 'title', None) else prop_id
        hints: list[str] = []
        if prop_spec and getattr(prop_spec, 'description', None):
            hints.append(prop_spec.description.strip())
        if prop_spec and getattr(prop_spec, 'aliases', None):
            alias_text = ", ".join(prop_spec.aliases)
            hints.append(f"Nel testo potrebbe essere indicata come: {alias_text}.")
        extra_hint = PROPERTY_EXTRA_HINTS.get(prop_id)
        if extra_hint:
            hints.append(extra_hint)
        if prop_spec and prop_spec.enum:
            choices = ", ".join(prop_spec.enum)
            hints.append(f"Scegli esclusivamente tra: {choices}.")
        if prop_spec and prop_spec.unit:
            hints.append(
                f"Se il valore è numerico restituiscilo in {prop_spec.unit} e imposta il campo 'unit' su {prop_spec.unit}."
            )
        question = f"Estrai il valore della proprietà '{display_name}'."
        if hints:
            question = question + " " + " ".join(hints)
        question += " Se l'informazione non è presente e lo schema lo consente restituisci null."
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
        unit = response.get("unit")
        if isinstance(unit, str):
            unit = unit.strip() or None
        if prop_spec and prop_spec.unit:
            expected_unit = prop_spec.unit
            if isinstance(value, (int, float)) and (not unit or unit.lower() != expected_unit.lower()):
                unit = expected_unit
        errors_raw = response.get("errors")
        if errors_raw is None:
            errors_list = []
        elif isinstance(errors_raw, list):
            errors_list = list(errors_raw)
        else:
            errors_list = [str(errors_raw)]
        if unit and "unit_missing" in errors_list:
            errors_list = [err for err in errors_list if err != "unit_missing"]
        candidate: Candidate = Candidate(
            value=value,
            source=CandidateSource.QA_LLM,
            raw=response.get("raw"),
            span=span if isinstance(span, (list, tuple)) else None,
            confidence=confidence_value,
            unit=unit,
            errors=errors_list,
        )
        return candidate
