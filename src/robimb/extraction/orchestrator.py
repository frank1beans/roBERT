"""Orchestrates multi-strategy property extraction with cascading fallbacks."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .fuse import Candidate, Fuser
from .orchestrator_base import OrchestratorBase, OrchestratorConfig
from .qa_llm import QALLM

from ..config import get_settings
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
    registry_path: str = Field(default_factory=lambda: str(get_settings().registry_path))


class Orchestrator:

    """Coordinate deterministic parsers, matchers and LLM fallbacks."""

    def __init__(self, fuse: Fuser, llm: Optional[QALLM], cfg: OrchestratorConfig) -> None:
        super().__init__(fuse=fuse, cfg=cfg)
        if llm is not None and not cfg.enable_llm:
            LOGGER.info(
                "llm_disabled", extra={"reason": "config_disabled", "llm_type": llm.__class__.__name__}
            )
        self._llm = llm if cfg.enable_llm else None

    def extract_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties for a single document."""

        text_id, category_id, text, property_specs, schema_properties = self._prepare_document(doc)

        properties_payload: Dict[str, Dict[str, Any]] = {}
        for prop_id, prop_schema in schema_properties.items():
            spec = property_specs.get(prop_id)
            result = self._extract_property(text, category_id, prop_id, prop_schema, spec)
            properties_payload[prop_id] = result

        validation = self._validate_payload(category_id, properties_payload)

        return self._finalise_document(doc, text_id, category_id, properties_payload, validation)

    def _extract_property(
        self,
        text: str,
        cat: str,
        prop: str,
        prop_schema: Dict[str, Any],
        prop_spec: Optional[PropertySpec] = None,
    ) -> Dict[str, Any]:
        allowed_sources, candidates = self._deterministic_candidates(cat, prop, prop_spec, text)

        if self._llm and "qa_llm" in allowed_sources:
            llm_candidate = self._llm_candidate(prop, text, prop_schema)
            if llm_candidate:
                candidates.append(llm_candidate)

        validator = self._build_validator(prop_spec)
        fused = self._fuse.fuse(candidates, validator)

        return self._build_property_payload(cat, prop, candidates, fused)

    def _llm_candidate(self, prop_id: str, text: str, prop_schema: Dict[str, Any]) -> Optional[Candidate]:
        if not self._llm:
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
                    # For length: first value in 2D/3D (WxHxD), or largest if significantly bigger
                    if len(values) >= 2 and max(values) > 1500 and max(values) / min(values) > 2:
                        selected = max(values)
                    else:
                        selected = values[0]
                elif "larghezza" in lowered or "width" in lowered:

                    # For width keep the first value when only two dimensions are present.
                    if len(values) == 2:
                        selected = values[0]
                    elif len(values) >= 3:
                        # Heuristic for three dimensions: prefer the most plausible width
                        # among the remaining values when the first value resembles a height
                        # (e.g. door formats like 70x210x4 cm).
                        first = values[0]
                        if first > 1500:
                            # Choose the largest candidate below the typical door height
                            candidates = [v for v in values[1:] if v <= 1500]
                            selected = max(candidates) if candidates else first
                        else:
                            selected = first
                    else:
                        selected = values[0]
                elif "altezza" in lowered or "height" in lowered:
                    # For height: second value in 2D (WxH), third in 3D door format (WxHxD), or largest if door-like
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

    def _matcher_candidates(self, category: str, prop_id: str, text: str) -> Iterable[Candidate]:
        lowered = prop_id.lower()
        results: List[Candidate] = []
        if lowered == "marchio":
            matches = list(self._brand_matcher.find(text, category=category))
            for brand, span, score in matches:
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
            if not matches:
                results.append(
                    Candidate(
                        value=self._brand_matcher.fallback_value,
                        source="fallback",
                        raw=None,
                        span=None,
                        confidence=0.05,
                        unit=None,
                        errors=[],
                    )
                )
        if "material" in lowered or "materiale" in lowered:
            matches = list(self._material_matcher.find(text))
            LOGGER.info(f"Material matcher for property '{prop_id}': found {len(matches)} matches in text")
            for match in matches:
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
        if any(token in lowered for token in ("norma", "standard")):
            matches = list(self._standard_matcher.find(text, category=category))
            for match in matches:
                results.append(
                    Candidate(
                        value=match.value,
                        source="matcher",
                        raw=match.surface,
                        span=match.span,
                        confidence=0.75 * float(match.score),
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
