"""Shared infrastructure for property extraction orchestrators.

This module defines the :class:`OrchestratorBase` template that centralises the
deterministic extraction logic (parsers, matchers, validation, payload
construction) used by both synchronous and asynchronous orchestrators. Concrete
implementations only need to provide the primitives to query an LLM and perform
the fusion step, allowing future extensions (e.g. batch orchestrators) to reuse
the same core behaviour while customising the execution model.
"""

from __future__ import annotations

import logging
import re
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..config import get_settings
from ..registry.schemas import slugify
from .fuse import Candidate, Fuser
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .matchers.norms import StandardMatcher
from .schema_registry import PropertySpec, load_category_schema, load_registry
from .validators import validate_properties

LOGGER = logging.getLogger(__name__)

__all__ = ["OrchestratorBase", "OrchestratorConfig"]

_SKIRTING_PATTERN = re.compile(
    r"(?P<label>h|altezza)[\s\.:=]*?(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>mm|cm|m)",
    re.IGNORECASE,
)

_WIDTH_RANGE_PATTERN = re.compile(
    r"(?P<v1>\d+(?:[\.,]\d+)?)\s*(?:÷|-|–)\s*(?P<v2>\d+(?:[\.,]\d+)?)(?:\s*(?P<unit>mm|cm|m))?\s*(?:di\s+)?larghezza",
    re.IGNORECASE,
)

_LENGTH_RANGE_PATTERN = re.compile(
    r"(?P<v1>\d+(?:[\.,]\d+)?)\s*(?:÷|-|–)\s*(?P<v2>\d+(?:[\.,]\d+)?)(?:\s*(?P<unit>mm|cm|m))?\s*(?:di\s+)?lunghezza",
    re.IGNORECASE,
)

_TIPOLOGIA_KEYWORDS = [
    (
        "ignifuga",
        [r"ignifug", r"gkfi", r"fire", r"rei\s*\d+", r"classe\s*ei", r"ei\s*\d+"],
    ),
    (
        "idrofuga",
        [r"idrolastra", r"idrof", r"idrorep", r"gki", r"lastra\s+hidro"],
    ),
    (
        "acustica",
        [
            r"4akustik",
            r"lastra[^\n]{0,40}acust",
            r"lastra[^\n]{0,40}fono",
            r"pannell[^\n]{0,40}acust",
            r"fireboard\s+akust",
        ],
    ),
    (
        "fibrogesso",
        [r"fibrogesso", r"fibro-?gesso", r"fermacell", r"fibre\s+di\s+gesso"],
    ),
    (
        "accoppiata_isolante",
        [r"accoppiat", r"lastra\s+coibentata", r"lastra\s+isolante", r"sandwich"],
    ),
]

_TOTAL_THICKNESS_PATTERN = re.compile(
    r"sp(?:essore)?\s*(?:totale|complessivo|tot\.?)?\s*[:=]?\s*(\d+(?:[\.,]\d+)?)\s*(mm|cm|m)",
    re.IGNORECASE,
)

_SECTION_PROFILE_KEYWORDS = (
    "profilo",
    "profilati",
    "sezione",
    "angolare",
)

_ISOLANTE_PATTERN = re.compile(r"isolant|lana\s+minerale|naturboard|mineral\s+wool", re.IGNORECASE)
_ISOLANTE_NEGATIVE_PATTERN = re.compile(
    r"\b(?:senza|privo|priva|sprovvist[oa])\s+(?:di\s+)?isolant",
    re.IGNORECASE,
)

_ORDITURA_PATTERN = re.compile(r"orditura[^\n]{0,40}?(\d+(?:[\.,]\d+)?)\s*(mm|cm|m)", re.IGNORECASE)
_EI_CLASS_PATTERN = re.compile(r"\bEI\s*(\d{2,3})\b", re.IGNORECASE)


class OrchestratorConfig(BaseModel):
    """Configuration for the property extraction orchestrators."""

    source_priority: List[str] = Field(default_factory=lambda: ["parser", "matcher", "qa_llm"])
    enable_matcher: bool = True
    enable_llm: bool = True
    registry_path: str = Field(default_factory=lambda: str(get_settings().registry_path))
    use_qa: bool = True
    fusion_mode: str = "fuse"
    qa_null_threshold: float = 0.25
    qa_confident_threshold: float = 0.60


class OrchestratorBase(ABC):
    """Base class exposing shared extraction utilities.

    Sub-classes are expected to orchestrate the execution flow (synchronous or
    asynchronous) while relying on these helpers for deterministic candidate
    generation, validation and payload construction.
    """

    def __init__(self, fuse: Fuser, cfg: OrchestratorConfig) -> None:
        self._fuse = fuse
        self._cfg = cfg
        self._brand_matcher = BrandMatcher()
        self._material_matcher = MaterialMatcher()
        self._standard_matcher = StandardMatcher()

    # ------------------------------------------------------------------
    # Document level helpers
    # ------------------------------------------------------------------
    def _prepare_document(self, doc: Dict[str, Any]) -> Tuple[
        Optional[str],
        str,
        str,
        Dict[str, PropertySpec],
        Dict[str, Any],
    ]:
        """Resolve metadata and schema information for a document."""

        text_id = self._resolve_text_id(doc)
        category_id = self._resolve_category_id(doc)
        text = doc.get("text", "") or ""

        if not category_id:
            raise ValueError("Input document is missing category information")

        category_id, category, schema = self._load_category_schema(doc, category_id)
        property_specs = {prop.id: prop for prop in category.properties}
        schema_properties = self._resolve_schema_properties(schema)
        return text_id, category_id, text, property_specs, schema_properties

    def _load_category_schema(self, doc: Dict[str, Any], category_id: str):
        """Load schema for a category, attempting alias resolution on failure."""

        try:
            category, schema = load_category_schema(category_id, registry_path=self._cfg.registry_path)
        except ValueError:
            alias = self._resolve_category_alias(doc, category_id)
            if not alias:
                raise
            category_id = alias
            category, schema = load_category_schema(category_id, registry_path=self._cfg.registry_path)
        return category_id, category, schema

    def _validate_payload(self, category_id: str, properties_payload: Dict[str, Dict[str, Any]]):
        """Validate fused properties and attach eventual issues to the payload."""

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
        return validation

    def _finalise_document(
        self,
        original_doc: Dict[str, Any],
        text_id: Optional[str],
        category_id: str,
        properties_payload: Dict[str, Dict[str, Any]],
        validation,
    ) -> Dict[str, Any]:
        """Compose the final document payload with validation metadata."""

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
            **original_doc,
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

    # ------------------------------------------------------------------
    # Candidate helpers
    # ------------------------------------------------------------------
    def _deterministic_candidates(
        self,
        category: str,
        prop_id: str,
        prop_spec: Optional[PropertySpec],
        text: str,
    ) -> Tuple[Sequence[str], List[Candidate]]:
        """Return allowed sources and deterministic candidates."""

        allowed_sources = self._determine_sources(prop_spec)
        candidates: List[Candidate] = []

        if "parser" in allowed_sources:
            candidates.extend(self._parser_candidates(prop_id, prop_spec, text))

        if self._cfg.enable_matcher and "matcher" in allowed_sources:
            candidates.extend(self._matcher_candidates(category, prop_id, text))

        return allowed_sources, candidates

    def _build_property_payload(
        self,
        category: str,
        prop: str,
        candidates: List[Candidate],
        fused: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalise fused data into the canonical payload structure."""

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
                "category": category,
                "property": prop,
                "candidates": candidates,
                "selected": result,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Shared deterministic logic
    # ------------------------------------------------------------------
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
        from .parsers import dimensions, numbers
        from .parsers.colors import parse_ral_colors
        from .parsers.standards import parse_standards
        from .parsers.thickness import parse_thickness
        from .parsers.thermal import parse_thermal_transmittance
        from .parsers.sound_insulation import parse_sound_insulation

        results: List[Candidate] = []
        lowered = prop_id.lower()

        if "formato" in lowered:
            if re.search(r"zoccol|battiscop", text, re.IGNORECASE):
                candidate = self._skirting_format_candidate(text)
                if candidate:
                    results.append(candidate)
            range_candidate = self._format_range_candidate(text)
            if range_candidate:
                results.append(range_candidate)

        if any(token in lowered for token in ("dimension", "formato")):
            for match in dimensions.parse_dimensions(text):
                values = list(match.values_mm)
                if not values:
                    continue

                max_dimension = max(values)
                start, end = match.span
                context_window = text[max(0, start - 40): min(len(text), end + 40)].lower()
                if max_dimension <= 150 and any(keyword in context_window for keyword in _SECTION_PROFILE_KEYWORDS):
                    continue

                raw_lower = match.raw.lower()
                has_height_marker = bool(re.search(r"\bh\s*\d", raw_lower))

                if "lunghezza" in lowered or "length" in lowered:
                    if len(values) >= 2:
                        selected: Any = max(values[:2])
                    else:
                        selected = values[0]
                elif "larghezza" in lowered or "width" in lowered:
                    if len(values) >= 2:
                        selected = min(values[:2])
                    else:
                        selected = values[0]
                elif "altezza" in lowered or "height" in lowered:
                    if len(values) >= 3:
                        selected = values[2]
                    elif has_height_marker and len(values) >= 2:
                        selected = values[-1]
                    else:
                        continue
                elif "profond" in lowered or "depth" in lowered:
                    if len(values) >= 3:
                        selected = values[2]
                    elif len(values) == 2 and not has_height_marker:
                        selected = values[-1]
                    else:
                        continue
                else:
                    if "formato" in lowered:
                        if re.search(r"zoccol|battiscop", text, re.IGNORECASE) and values:
                            height_mm = max(values)
                            if height_mm >= 100:
                                selected = f"{int(height_mm / 10)} cm"
                            else:
                                selected = f"{int(height_mm)} mm"
                        elif all(v >= 100 for v in values):
                            cm_values = [int(v / 10) for v in values]
                            selected = "x".join(str(v) for v in cm_values) + " cm"
                        else:
                            selected = "x".join(str(int(v)) for v in values) + " mm"
                    else:
                        keys = ["width_mm", "height_mm", "depth_mm"]
                        selected = {key: values[idx] for idx, key in enumerate(keys) if idx < len(values)}

                if isinstance(selected, (int, float)):
                    if lowered == "dimensione_larghezza" and selected < 150:
                        continue
                    if lowered == "dimensione_altezza" and selected < 150:
                        continue
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
            labeled_found = False
            for match in parse_thickness(text):
                results.append(
                    Candidate(
                        value=match.value_mm,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.92,
                        unit="mm",
                        errors=[],
                    )
                )
                labeled_found = True

            if not labeled_found:
                for match in numbers.extract_numbers(text):
                    before = text[max(0, match.start - 25):match.start].lower()
                    after = text[match.end: min(len(text), match.end + 20)].lower()

                    if not re.search(r"sp(?:\.|ess)", before):
                        continue

                    if re.search(r"\b(?:iso|uni|en)\s*(?:en\s*)?\d", after):
                        continue
                    if re.search(r"\b(?:iso|uni|en)\s*$", before):
                        continue

                    if match.start > 0:
                        prev_chars = text[max(0, match.start - 8):match.start]
                        if prev_chars and prev_chars[-1].isalpha():
                            continue
                        if re.search(r'[A-Z]{1,3}\d*$', prev_chars, re.IGNORECASE):
                            continue
                        if re.search(r'(ral|uni)\s*$', prev_chars, re.IGNORECASE):
                            continue

                    results.append(
                        Candidate(
                            value=match.value,
                            source="parser",
                            raw=match.raw,
                            span=(match.start, match.end),
                            confidence=0.70,
                            unit="mm",
                            errors=[],
                        )
                    )


        if lowered.startswith("dimensione_") and not any(c.get("source") == "parser" and c.get("unit") == "mm" for c in results):
            special = re.search(r'dim\.?\s*(\d+)(?:\([^\)]*\))?\s*x\s*(\d+)(?:\s*x\s*(\d+))?', text, re.IGNORECASE)
            if special:
                def _to_mm(value: str | None) -> float | None:
                    if not value:
                        return None
                    try:
                        return float(value.replace(',', '.')) * 10
                    except ValueError:
                        return None

                width_cm = _to_mm(special.group(1))
                height_cm = _to_mm(special.group(2))
                depth_cm = _to_mm(special.group(3))
                if lowered == "dimensione_larghezza" and width_cm is not None and width_cm >= 150:
                    results.append(
                        Candidate(
                            value=width_cm,
                            source="parser",
                            raw=special.group(0),
                            span=(special.start(), special.end()),
                            confidence=0.85,
                            unit="mm",
                            errors=[],
                        )
                    )
                if lowered == "dimensione_altezza" and height_cm is not None and height_cm >= 150:
                    results.append(
                        Candidate(
                            value=height_cm,
                            source="parser",
                            raw=special.group(0),
                            span=(special.start(), special.end()),
                            confidence=0.85,
                            unit="mm",
                            errors=[],
                        )
                    )
                if lowered == "dimensione_profondita" and depth_cm is not None:
                    results.append(
                        Candidate(
                            value=depth_cm,
                            source="parser",
                            raw=special.group(0),
                            span=(special.start(), special.end()),
                            confidence=0.85,
                            unit="mm",
                            errors=[],
                        )
                    )


        if lowered == "classe_ei":
            allowed = set(spec.enum) if spec and spec.enum else None
            for match in _EI_CLASS_PATTERN.finditer(text):
                value = f"EI{match.group(1)}"
                if allowed and value not in allowed:
                    continue
                results.append(
                    Candidate(
                        value=value,
                        source="parser",
                        raw=match.group(0),
                        span=(match.start(), match.end()),
                        confidence=0.88,
                        unit=None,
                        errors=[],
                    )
                )

        if lowered == "presenza_isolante":
            negative = _ISOLANTE_NEGATIVE_PATTERN.search(text)
            if negative:
                results.append(
                    Candidate(
                        value="no",
                        source="parser",
                        raw=text[negative.start() : negative.end()],
                        span=(negative.start(), negative.end()),
                        confidence=0.85,
                        unit=None,
                        errors=[],
                    )
                )
            else:
                positive = _ISOLANTE_PATTERN.search(text)
                if positive:
                    results.append(
                        Candidate(
                            value="si",
                            source="parser",
                            raw=text[positive.start() : positive.end()],
                            span=(positive.start(), positive.end()),
                            confidence=0.85,
                            unit=None,
                            errors=[],
                        )
                    )


        if "trasmittanza" in lowered or "uw" in lowered or "uf" in lowered or "ug" in lowered:
            for match in parse_thermal_transmittance(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.9,
                        unit="W/m²K",
                        errors=[],
                    )
                )

        if "isolamento_acustico" in lowered or lowered.endswith("_acustico_db") or lowered.endswith("_db"):
            for match in parse_sound_insulation(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.88,
                        unit="dB",
                        errors=[],
                    )
                )

        if "ral" in lowered or "colore" in lowered:
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

        if "norma" in lowered or "standard" in lowered:
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

        if "db" in lowered or "decibel" in lowered:
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
                        unit="dB",
                        errors=[],
                    )
                )

        return results

    def _skirting_format_candidate(self, text: str) -> Optional[Candidate]:
        match = _SKIRTING_PATTERN.search(text)
        if not match:
            return None
        raw_value = match.group("value")
        unit = match.group("unit").lower()
        start, end = match.span()

        value = raw_value.replace(",", ".")
        if unit == "m":
            formatted = f"{float(value) * 100:.0f} cm"
        elif unit == "cm":
            formatted = raw_value.replace(".", ",") + " cm" if "," in raw_value else f"{raw_value} cm"
        else:
            formatted = f"{raw_value} mm"

        return Candidate(
            value=formatted,
            source="parser",
            raw=text[start:end],
            span=[start, end],
            confidence=0.92,
            unit=None,
            errors=[],
        )

    def _format_range_candidate(self, text: str) -> Optional[Candidate]:
        width_match = _WIDTH_RANGE_PATTERN.search(text)
        length_match = _LENGTH_RANGE_PATTERN.search(text)
        if not width_match and not length_match:
            return None

        def _format_part(match_obj):
            if not match_obj:
                return None, None
            v1 = match_obj.group("v1").replace(",", ".")
            v2 = match_obj.group("v2").replace(",", ".")
            unit = match_obj.group("unit") or "cm"
            def _fmt(value: str) -> str:
                return value.rstrip("0").rstrip(".") if "." in value else value
            return f"{_fmt(v1)}-{_fmt(v2)}", unit

        width_str, width_unit = _format_part(width_match)
        length_str, length_unit = _format_part(length_match)
        if not width_str and not length_str:
            return None

        unit = width_unit or length_unit or "cm"
        if width_str and length_str:
            value = f"{width_str}x{length_str} {unit}"
            start = width_match.start()
            end = length_match.end()
        elif width_str:
            value = f"{width_str} {unit}"
            start, end = width_match.span()
        else:
            value = f"{length_str} {unit}"
            start, end = length_match.span()

        return Candidate(
            value=value,
            source="parser",
            raw=text[start:end],
            span=[start, end],
            confidence=0.88,
            unit=None,
            errors=[],
        )

    def _tipologia_candidate(self, text: str) -> Optional[Candidate]:
        for tipologia, patterns in _TIPOLOGIA_KEYWORDS:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    span = [match.start(), match.end()]
                    candidate = Candidate(
                        value=tipologia,
                        source="matcher",
                        raw=text[span[0] : span[1]],
                        span=span,
                        confidence=0.75,
                        unit=None,
                        errors=[],
                    )
                    candidate["start"], candidate["end"] = span
                    return candidate

        match = re.search(r"\b(gkb|standard)\b", text, re.IGNORECASE)
        if match:
            span = [match.start(), match.end()]
            candidate = Candidate(
                value="standard",
                source="matcher",
                raw=text[span[0] : span[1]],
                span=span,
                confidence=0.6,
                unit=None,
                errors=[],
            )
            candidate["start"], candidate["end"] = span
            return candidate

        return None

    def _matcher_candidates(self, category: str, prop_id: str, text: str) -> Iterable[Candidate]:
        results: List[Candidate] = []
        lowered = prop_id.lower()
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
                        source="matcher_fallback",
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
        if lowered == "tipologia_lastra":
            candidate = self._tipologia_candidate(text)
            if candidate:
                results.append(candidate)

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

