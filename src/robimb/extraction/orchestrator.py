"""Orchestrates multi-strategy property extraction with cascading fallbacks."""
from __future__ import annotations

import logging
import re
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..config import get_settings
from .fuse import Candidate, CandidateSource, Fuser
from .fusion_policy import FusionThresholds, fuse_property_candidates
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .matchers.norms import StandardMatcher
from .domain_heuristics import post_process_properties
from .parsers import dimensions, numbers
from .parsers.colors import parse_ral_colors
from .parsers.standards import parse_standards
from .parsers.flow_rate import parse_flow_rate
from .parsers.labeled_dimensions import parse_labeled_dimensions
from .parsers.acoustic import parse_acoustic_coefficient
from .parsers.fire_class import parse_fire_class
from .parsers.thickness import parse_thickness
from .parsers.installation_type import parse_installation_type
from .parsers.sound_insulation import parse_sound_insulation
from .parsers.thermal import parse_thermal_transmittance
from .qa_llm import QALLM
from ..registry.schemas import slugify
from .schema_registry import PropertySpec, load_category_schema, load_registry
from .validators import validate_properties

LOGGER = logging.getLogger(__name__)

__all__ = ["Orchestrator", "OrchestratorConfig"]

PROPERTY_EXTRA_HINTS = {
    "trasmittanza_termica": "Cerca valori di trasmittanza (Uw, Uf, Ug) espressi in W/m²K.",
    "isolamento_acustico_db": "Cerca valori di isolamento acustico (Rw, Ra) espressi in dB.",
}

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

_ISOLANTE_PATTERN = re.compile(r"isolant|lana\s+minerale|naturboard|mineral\s+wool", re.IGNORECASE)
_ISOLANTE_NEGATIVE_PATTERN = re.compile(
    r"\b(?:senza|privo|priva|sprovvist[oa])\s+(?:di\s+)?isolant",
    re.IGNORECASE,
)
_EI_CLASS_PATTERN = re.compile(r"\bEI\s*(\d{2,3})\b", re.IGNORECASE)

_ORDITURA_PATTERN = re.compile(r"orditura[^\n]{0,40}?(\d+(?:[\.,]\d+)?)\s*(mm|cm|m)", re.IGNORECASE)


class OrchestratorConfig(BaseModel):
    """Configuration for the property extraction orchestrator."""

    source_priority: List[str] = Field(default_factory=lambda: ["parser", "matcher", "qa_llm"])
    enable_matcher: bool = True
    enable_llm: bool = True
    registry_path: str = Field(default_factory=lambda: str(get_settings().registry_path))
    use_qa: bool = True
    fusion_mode: str = "fuse"
    qa_null_threshold: float = 0.25
    qa_confident_threshold: float = 0.60


class Orchestrator:
    """Coordinate deterministic parsers, matchers and LLM fallbacks."""

    def __init__(self, fuse: Fuser, llm: Optional[QALLM], cfg: OrchestratorConfig) -> None:
        self._fuse = fuse
        if llm is not None and not cfg.enable_llm:
            LOGGER.info(
                "llm_disabled", extra={"reason": "config_disabled", "llm_type": llm.__class__.__name__}
            )
        self._llm = llm if cfg.enable_llm else None
        self._cfg = cfg
        self._brand_matcher = BrandMatcher()
        self._material_matcher = MaterialMatcher()
        self._standard_matcher = StandardMatcher()
        self._property_profiles = self._load_property_profiles(Path(cfg.registry_path))

    def extract_document(
        self,
        doc: Dict[str, Any],
        qa_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
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

        qa_map = qa_predictions or doc.get("_qa_predictions") or {}

        properties_payload: Dict[str, Dict[str, Any]] = {}
        for prop_id, prop_schema in schema_properties.items():
            spec = property_specs.get(prop_id)
            qa_candidate = qa_map.get(prop_id) if self._cfg.use_qa else None
            result = self._extract_property(
                text,
                category_id,
                prop_id,
                prop_schema,
                spec,
                qa_candidate,
            )
            properties_payload[prop_id] = result

        post_process_properties(text, category_id, properties_payload, logger=LOGGER)

        # WBS relevance boost (non distruttivo)
        self._apply_wbs_relevance(doc, properties_payload)

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

        for prop_id, payload in validation.normalized.items():
            target = properties_payload.get(prop_id)
            if target is not None:
                target["normalized"] = payload.value

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

        base_doc = {k: v for k, v in doc.items() if k != "_qa_predictions"}
        result = {
            **base_doc,  # Include all original fields
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

    def _load_property_profiles(self, registry_path: Path) -> List[Dict[str, Any]]:
        try:
            profiles_path = registry_path.parent / "property_profiles.json"
            if not profiles_path.exists():
                return []
            payload = json.loads(profiles_path.read_text(encoding="utf-8"))
            profiles = payload.get("profiles", [])
            normalized_profiles: List[Dict[str, Any]] = []
            for profile in profiles:
                patterns = [str(pat).lower() for pat in profile.get("patterns", [])]
                props = [str(p) for p in profile.get("properties", [])]
                if not patterns or not props:
                    continue
                normalized_profiles.append(
                    {
                        "id": profile.get("id") or "profile",
                        "patterns": patterns,
                        "properties": props,
                    }
                )
            return normalized_profiles
        except Exception as exc:  # pragma: no cover - robustezza
            LOGGER.warning("property_profiles_load_failed", exc_info=exc)
            return []

    def _resolve_profile_properties(self, doc: Dict[str, Any]) -> List[str]:
        code = str(doc.get("wbs6_code") or doc.get("wbs_code") or "").lower()
        desc = str(doc.get("wbs6_description") or doc.get("wbs_description") or "").lower()
        matched: List[str] = []
        for profile in self._property_profiles:
            for pat in profile.get("patterns", []):
                if pat and (pat in code or pat in desc):
                    matched.extend(profile.get("properties", []))
                    break
        # dedup preserving order
        seen = set()
        unique: List[str] = []
        for prop in matched:
            if prop not in seen:
                unique.append(prop)
                seen.add(prop)
        return unique

    def _apply_wbs_relevance(
        self,
        doc: Dict[str, Any],
        properties_payload: Dict[str, Dict[str, Any]],
    ) -> None:
        if not self._property_profiles:
            return
        relevant_props = set(self._resolve_profile_properties(doc))
        if not relevant_props:
            return
        for prop_id, payload in properties_payload.items():
            if payload.get("value") is None:
                continue
            if prop_id in relevant_props:
                payload["relevance"] = "high"
                if "confidence" in payload and payload["confidence"] is not None:
                    try:
                        payload["confidence"] = min(float(payload["confidence"]) + 0.05, 1.0)
                    except Exception:
                        pass
            else:
                payload.setdefault("relevance", "unspecified")

    def _extract_property(
        self,
        text: str,
        cat: str,
        prop: str,
        prop_schema: Dict[str, Any],
        prop_spec: Optional[PropertySpec] = None,
        qa_prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        allowed_sources = self._determine_sources(prop_spec)
        rule_candidates: List[Candidate] = []

        if "parser" in allowed_sources:
            rule_candidates.extend(self._parser_candidates(prop, prop_spec, text))

        if self._cfg.enable_matcher and "matcher" in allowed_sources:
            rule_candidates.extend(self._matcher_candidates(cat, prop, text))

        if self._llm and "qa_llm" in allowed_sources:
            llm_candidate = self._llm_candidate(prop, text, prop_schema, prop_spec)
            if llm_candidate:
                rule_candidates.append(llm_candidate)

        validator = self._build_validator(prop_spec)
        rules_fused = self._fuse.fuse(rule_candidates, validator)
        rules_candidate = self._candidate_from_rules(rules_fused)
        qa_candidate = self._candidate_from_qa(qa_prediction)

        selected_candidate, reason = fuse_property_candidates(
            rules_candidate,
            qa_candidate,
            fusion_mode=self._cfg.fusion_mode,
            thresholds=FusionThresholds(
                qa_min=self._cfg.qa_null_threshold,
                qa_confident=self._cfg.qa_confident_threshold,
            ),
        )

        if selected_candidate is None:
            selected_candidate = {
                "value": None,
                "source": None,
                "raw": None,
                "span": None,
                "confidence": 0.0,
                "unit": None,
                "errors": ["no_valid_candidate"],
            }
        else:
            span = self._normalize_span(selected_candidate.get("span"))
            selected_candidate["span"] = span
            if span:
                selected_candidate["start"], selected_candidate["end"] = span
            else:
                selected_candidate.setdefault("start", None)
                selected_candidate.setdefault("end", None)
            selected_candidate.setdefault("errors", [])
            is_valid, messages = validator(selected_candidate)
            if not is_valid and messages:
                selected_candidate["errors"].extend(messages)

        selected_candidate.setdefault("start", None)
        selected_candidate.setdefault("end", None)
        selected_candidate.setdefault("normalized", None)
        selected_candidate["confidence"] = float(selected_candidate.get("confidence") or 0.0)
        if selected_candidate.get("source") is None:
            if qa_candidate and selected_candidate is qa_candidate:
                selected_candidate["source"] = "qa"
            elif rules_candidate and selected_candidate is rules_candidate:
                selected_candidate["source"] = rules_candidate.get("source", "rules")
            else:
                selected_candidate["source"] = None

        LOGGER.info(
            "property_fused",
            extra={
                "category": cat,
                "property": prop,
                "reason": reason,
                "qa": qa_prediction,
                "selected": selected_candidate,
            },
        )

        return selected_candidate

    def _candidate_from_rules(self, fused: Candidate) -> Optional[Candidate]:
        value = fused.get("value")
        span = self._normalize_span(fused.get("span")) if fused.get("span") else None
        if value is None and span is None:
            return None
        source = fused.get("source")
        if isinstance(source, CandidateSource):
            source_value = source.value
        elif source == "fallback":
            source_value = CandidateSource.MATCHER_FALLBACK.value
        else:
            source_value = source or "rules"
        candidate = Candidate(
            value=value,
            source=source_value,
            raw=fused.get("raw"),
            span=span,
            confidence=float(fused.get("confidence") or 0.0),
            unit=fused.get("unit"),
            errors=list(fused.get("errors", [])),
        )
        if span:
            candidate["start"], candidate["end"] = span
        else:
            candidate["start"] = candidate["end"] = None
        return candidate

    def _candidate_from_qa(self, qa_prediction: Optional[Dict[str, Any]]) -> Optional[Candidate]:
        if not qa_prediction:
            return None
        start = qa_prediction.get("start")
        end = qa_prediction.get("end")
        if start is None or end is None:
            return None
        value = qa_prediction.get("span")
        candidate = Candidate(
            value=value,
            source="qa",
            raw=value,
            span=[int(start), int(end)],
            confidence=float(qa_prediction.get("score", 0.0)),
            unit=None,
            errors=[],
        )
        candidate["start"], candidate["end"] = int(start), int(end)
        return candidate

    def _skirting_format_candidate(self, text: str) -> Optional[Candidate]:
        match = _SKIRTING_PATTERN.search(text)
        if not match:
            return None
        raw_value = match.group("value")
        unit = match.group("unit").lower()
        start, end = match.span()

        if unit == "m":
            value_cm = float(raw_value.replace(",", ".")) * 100
            formatted = f"{value_cm:.0f} cm"
        elif unit == "cm":
            formatted = f"{raw_value.replace('.', ',')} cm" if "," in raw_value else f"{raw_value} cm"
        else:  # mm
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

        if "formato" in lowered:
            if re.search(r"zoccol|battiscop", text, re.IGNORECASE):
                skirting_candidate = self._skirting_format_candidate(text)
                if skirting_candidate:
                    results.append(skirting_candidate)
            range_candidate = self._format_range_candidate(text)
            if range_candidate:
                results.append(range_candidate)

        if any(token in lowered for token in ("dimension", "formato")):
            for match in dimensions.parse_dimensions(text):
                values = list(match.values_mm)
                if not values:
                    continue

                raw_lower = match.raw.lower()
                has_height_marker = bool(re.search(r"\bh\s*\d", raw_lower))
                selected: Any

                if "lunghezza" in lowered or "length" in lowered:
                    if len(values) >= 2:
                        selected = max(values[:2])
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
            # First try to find explicitly labeled thickness (e.g., "sp. 20 mm")
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

            # Fallback to generic numbers only if no labeled thickness found
            if not labeled_found:
                for match in numbers.extract_numbers(text):
                    before = text[max(0, match.start - 25) : match.start].lower()
                    after = text[match.end : min(len(text), match.end + 20)].lower()

                    if not re.search(r"sp(?:\.|ess)", before):
                        continue

                    if re.search(r"\b(?:iso|uni|en)\s*(?:en\s*)?\d", after):
                        continue
                    if re.search(r"\b(?:iso|uni|en)\s*$", before):
                        continue

                    if match.start > 0:
                        prev_chars = text[max(0, match.start - 8) : match.start]
                        if prev_chars and prev_chars[-1].isalpha():
                            continue
                        if re.search(r"[A-Z]{1,3}\d*$", prev_chars, re.IGNORECASE):
                            continue
                        if re.search(r"(ral|uni)\s*$", prev_chars, re.IGNORECASE):
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

        elif "trasmittanza" in lowered or "uw" in lowered or "uf" in lowered or "ug" in lowered:
            for match in parse_thermal_transmittance(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.90,
                        unit="W/m²K",
                        errors=[],
                    )
                )

        elif (
            "isolamento_acustico" in lowered
            or lowered.endswith("_acustico_db")
            or lowered.endswith("_db")
            or "decibel" in lowered
        ):
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

        elif "portata" in lowered or "flow" in lowered or "l/min" in lowered or "l_min" in lowered:
            for match in parse_flow_rate(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.90,
                        unit=match.unit,
                        errors=[],
                    )
                )

        elif "fonoassorbimento" in lowered or "assorbimento" in lowered or "acoustic" in lowered:
            for match in parse_acoustic_coefficient(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.88,
                        unit=None,
                        errors=[],
                    )
                )

        elif "fuoco" in lowered or "classe" in lowered and "reazione" in lowered:
            for match in parse_fire_class(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.85,
                        unit=None,
                        errors=[],
                    )
                )

        elif "installazione" in lowered or "tipologia" in lowered:
            for match in parse_installation_type(text):
                results.append(
                    Candidate(
                        value=match.value,
                        source="parser",
                        raw=match.raw,
                        span=match.span,
                        confidence=0.88,
                        unit=None,
                        errors=[],
                    )
                )

        elif lowered == "classe_ei":
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

        elif lowered == "presenza_isolante":
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

        # Check for explicitly labeled dimensions (e.g., "lunghezza 60 cm")
        if any(token in lowered for token in ("lunghezza", "larghezza", "altezza", "profondità", "profondita")):
            for match in parse_labeled_dimensions(text):
                # Match the label to the property
                if match.label in lowered:
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
                        source="matcher",
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

    def _llm_candidate(
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
            response = self._llm.ask(text, question, llm_schema)
        except Exception as exc:  # pragma: no cover - defensive
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
            source="qa_llm",
            raw=response.get("raw"),
            span=span if isinstance(span, (list, tuple)) else None,
            confidence=confidence_value,
            unit=unit,
            errors=errors_list,
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
