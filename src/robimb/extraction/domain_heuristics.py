"""Domain-specific heuristics for property extraction.

Questo modulo fornisce regole euristiche basate su conoscenza del dominio BIM
per inferire proprieta' quando rules/matchers/LLM falliscono.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, MutableMapping, Optional

from .cartongesso import CartongessoFeatures, extract_cartongesso_features, summarize_cartongesso_features
from .parsers.numbers import parse_number_it


# Regole per materiali basate su keywords nel testo
MATERIAL_KEYWORDS = {
    # Legno
    r"\bdoghe\b": "legno",
    r"\blegno\b": "legno",
    r"\bteak\b": "legno_teak",
    r"\blaminato\b": "legno_laminato",

    # Vetro (piu' specifici prima)
    r"\b(?:cristallo|vetro)\s+temperato\b": "vetro_temperato",
    r"\bcristallo\b": "vetro",
    r"\bvetro\b": "vetro",
    r"\bmetallo\b": "metallo",
    r"\bmetallica?\b": "metallo",
    r"\bvitreous\s+china\b": "ceramica",
    r"\bmdf\b": "legno",
    r"\blaminato\s+hpl\b": "resina",
    r"\bhpl\b": "resina",
    r"\bduroplast\b": "plastica",
    r"\bnylon\b": "plastica",

    # Metalli
    r"\bAISI\s*304\b": "acciaio_inox",
    r"\bacciaio\s+inox\b": "acciaio_inox",
    r"\binox\b": "acciaio_inox",
    r"\bacciaio\b": "acciaio",
    r"\bcromatx?[oa]\b": "metallo_cromato",  # cromato/cromata
    r"\bottone\b": "metallo_ottone",
    r"\balluminio\b": "alluminio",

    # Plastica/PVC (piu' specifici prima)
    r"\bin\s+pvc\b": "plastica_pvc",
    r"\bpvc\b": "plastica_pvc",
    r"\bplastica\b": "plastica",
    r"\bpolipropilene\b": "plastica_polipropilene",

    # Ceramica/Gres
    r"\bgres\s+porcellanato\b": "ceramica_gres_porcellanato",
    r"\bgres\b": "ceramica_gres",
    r"\bceramica\b": "ceramica",
    r"\bporcellana\b": "ceramica_porcellana",

    # Altri
    r"\bcartongesso\b": "gesso",
    r"\bgesso\b": "gesso",
    r"\bmarmo\b": "pietra_marmo",
    r"\bgranito\b": "pietra_granito",
}

# Default materiali per categoria oggetto
MATERIAL_BY_OBJECT_TYPE = {
    # Rubinetteria
    r"\b(miscelatore|rubinetto|doccetta)\b": "metallo",

    # Sanitari
    r"\b(wc|lavabo|bidet|vasca)\b": "ceramica",

    # Vetro
    r"\b(specchio|box\s+doccia)\b": "vetro",

    # Metallo
    r"\b(griglia|maniglione|sostegno|corrimano)\b": "metallo",

    # Legno
    r"\b(seggiolino|panca|ripiano)\b": "legno",

    # Accessori bagno
    r"\bporta\s+scopino\b": "metallo",
    r"\bcassetta\s+di\s+scarico\b": "plastica",
    r"\bpiatto\s+doccia\b": "resina",
    r"\bpiano\s+lavabo\b": "resina",

    # Plastica
    r"\b(piletta|sifone)\b": "plastica",
}

# Tipologia installazione da keywords
INSTALLATION_TYPE_KEYWORDS = {
    r"\ba\s+parete\b": "a_parete",
    r"\bparete\b": "a_parete",
    r"\bpareti\b": "a_parete",
    r"\bmuro\b": "a_parete",
    r"\bsospeso\b": "sospesa",
    r"\ba\s+pavimento\b": "a_pavimento",
    r"\bpavimento\b": "a_pavimento",
    r"\bda\s+appoggio\b": "a_pavimento",
    r"\bappoggio\b": "a_pavimento",
    r"\bincasso\b": "ad_incasso",
    r"\bfilo\s+parete\b": "filo_parete",
    r"\bangolo\b": "angolare",
}


UNIT_TO_MM = {
    'mm': 1.0,
    'millimetri': 1.0,
    'millimetro': 1.0,
    'cm': 10.0,
    'centimetri': 10.0,
    'centimetro': 10.0,
    'm': 1000.0,
    'metri': 1000.0,
    'metro': 1000.0,
}

DIMENSION_KEYS = (
    'dimensione_lunghezza',
    'dimensione_larghezza',
    'dimensione_altezza',
)

DIMENSION_LABEL_PATTERNS = {
    'dimensione_lunghezza': re.compile(r"\b(?:L\.?|lunghezza)\s*(?:[:=]?)*\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?", re.IGNORECASE),
    'dimensione_larghezza': re.compile(r"\b(?:P\.?|prof(?:ondita'?|\.)|larghezza)\s*(?:[:=]?)*\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?", re.IGNORECASE),
    'dimensione_altezza': re.compile(r"\b(?:H\.?|altezza)\s*(?:[:=]?)*\s*(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?", re.IGNORECASE),
}

DIMENSION_RANGE_PATTERNS = {
    'dimensione_lunghezza': re.compile(r"\b(?:lunghezza|L\.?)\s*(?:variabile\s+)?da\s+(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?\s*(?:a|-)\s*(?P<value_max>\d+(?:[.,]\d+)?)\s*(?P<unit_max>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?", re.IGNORECASE),
    'dimensione_larghezza': re.compile(r"\b(?:prof(?:ondita'?|\.)|P\.?|larghezza)\s*(?:variabile\s+)?da\s+(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?\s*(?:a|-)\s*(?P<value_max>\d+(?:[.,]\d+)?)\s*(?P<unit_max>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?", re.IGNORECASE),
    'dimensione_altezza': re.compile(r"\b(?:H\.?|altezza)\s*(?:variabile\s+)?da\s+(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?\s*(?:a|-)\s*(?P<value_max>\d+(?:[.,]\d+)?)\s*(?P<unit_max>mm|cm|m|millimetri|millimetro|centimetri|centimetro|metri|metro)?", re.IGNORECASE),
}


def infer_material(text: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Inferisci materiale usando euristiche dominio.

    Args:
        text: Testo descrizione
        category: Categoria BIM (opzionale)

    Returns:
        Dict con value, confidence, source se trovato, None altrimenti
    """
    text_lower = text.lower()

    # Check for false positives first (e.g., "cartuccia ceramica" in mixer description)
    # If we find "cartuccia" + "ceramica", it's about the cartridge, not the material
    if re.search(r"cartuccia[^.]{0,30}ceramica|ceramica[^.]{0,30}cartuccia", text_lower):
        # Skip ceramica match, infer from object type instead
        for pattern, material in MATERIAL_BY_OBJECT_TYPE.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                confidence = 0.7  # Higher confidence when we avoid false positive
                return {
                    "value": material,
                    "confidence": confidence,
                    "source": "heuristic_object_type",
                    "raw": pattern
                }

    # 1. Cerca keywords specifiche materiali
    for pattern, material in MATERIAL_KEYWORDS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            # Confidence piu' alta se match esplicito
            confidence = 0.75
            return {
                "value": material,
                "confidence": confidence,
                "source": "heuristic_keyword",
                "raw": pattern
            }

    # 2. Inferisci da tipo oggetto
    for pattern, material in MATERIAL_BY_OBJECT_TYPE.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            # Confidence piu' bassa (inferenza)
            confidence = 0.6
            return {
                "value": material,
                "confidence": confidence,
                "source": "heuristic_object_type",
                "raw": pattern
            }

    return None


def infer_installation_type(text: str) -> Optional[Dict[str, Any]]:
    """Inferisci tipologia installazione da keywords.

    Args:
        text: Testo descrizione

    Returns:
        Dict con value, confidence, source se trovato, None altrimenti
    """
    text_lower = text.lower()

    for pattern, inst_type in INSTALLATION_TYPE_KEYWORDS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            confidence = 0.8 if "a parete" in pattern or "pavimento" in pattern else 0.7
            return {
                "value": inst_type,
                "confidence": confidence,
                "source": "heuristic_keyword",
                "raw": re.search(pattern, text_lower, re.IGNORECASE).group()
            }

    return None


def apply_domain_heuristics(
    text: str,
    category: str,
    existing_properties: Dict[str, Any]
) -> Dict[str, Any]:
    """Applica euristiche dominio per riempire gap nelle proprieta'.

    Questa funzione viene chiamata DOPO rules/matchers e PRIMA di LLM,
    per cercare di riempire proprieta' mancanti con conoscenza dominio.

    Args:
        text: Testo descrizione
        category: Categoria BIM
        existing_properties: Proprieta' gia' estratte

    Returns:
        Dict con proprieta' inferite (solo quelle mancanti/null)
    """
    inferred = {}

    # Skip material/installation heuristics for cartongesso (not relevant)
    if category.lower() in ["opere_da_cartongessista", "controsoffitti"]:
        return inferred

    # Materiale
    if not existing_properties.get("materiale") or \
       existing_properties.get("materiale", {}).get("value") is None:
        material = infer_material(text, category)
        if material:
            inferred["materiale"] = material

    # Tipologia installazione
    if not existing_properties.get("tipologia_installazione") or \
       existing_properties.get("tipologia_installazione", {}).get("value") is None:
        inst_type = infer_installation_type(text)
        if inst_type:
            inferred["tipologia_installazione"] = inst_type

    return inferred


def validate_material_consistency(
    material_value: str,
    text: str,
    category: str
) -> Dict[str, Any]:
    """Valida consistenza del materiale estratto con il contesto.

    Args:
        material_value: Valore materiale estratto
        text: Testo originale
        category: Categoria BIM

    Returns:
        Dict con:
        - is_valid: bool
        - confidence_adjustment: float (-1.0 a +1.0)
        - warnings: list di warning se inconsistente
    """
    result = {
        "is_valid": True,
        "confidence_adjustment": 0.0,
        "warnings": []
    }

    text_lower = text.lower()

    # Check 1: Miscelatori/rubinetti NON sono ceramica
    if re.search(r"\b(miscelatore|rubinetto)\b", text_lower):
        if "ceramica" in material_value.lower():
            # Probabilmente ha confuso "cartuccia ceramica" con materiale
            result["warnings"].append(
                "Materiale 'ceramica' sospetto per miscelatore/rubinetto (probabilmente metallo)"
            )
            result["confidence_adjustment"] = -0.4
            result["is_valid"] = False

    # Check 2: Box doccia "cristallo" dovrebbe essere vetro
    if re.search(r"\bbox\s+doccia\b", text_lower) and "cristallo" in text_lower:
        if material_value.lower() not in ["vetro", "vetro_temperato"]:
            result["warnings"].append(
                "Box doccia in 'cristallo' dovrebbe essere vetro"
            )
            result["confidence_adjustment"] = -0.3

    # Check 3: AISI 304 -> acciaio inox
    if "aisi" in text_lower or "inox" in text_lower:
        if "acciaio" not in material_value.lower():
            result["warnings"].append(
                "Testo menziona AISI/inox ma materiale non e' acciaio"
            )
            result["confidence_adjustment"] = -0.3

    return result

def post_process_properties(
    text: str,
    category: str,
    properties_payload: MutableMapping[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Apply heuristics and material validation to the extraction payload."""

    log = logger or logging.getLogger(__name__)

    heuristic_properties = apply_domain_heuristics(text, category, properties_payload)
    for prop_id, heuristic_result in heuristic_properties.items():
        existing = properties_payload.get(prop_id)
        existing_confidence = float(existing.get("confidence") or 0.0) if existing else 0.0
        if existing is None or existing.get("value") is None or existing_confidence < 0.3:
            properties_payload[prop_id] = {
                "value": heuristic_result["value"],
                "source": heuristic_result["source"],
                "raw": heuristic_result.get("raw"),
                "span": None,
                "confidence": heuristic_result["confidence"],
                "unit": None,
                "errors": [],
            }
            log.info(
                "heuristic_applied",
                extra={
                    "property": prop_id,
                    "value": heuristic_result["value"],
                    "source": heuristic_result["source"],
                    "confidence": heuristic_result["confidence"],
                },
            )

    material_prop = properties_payload.get("materiale")
    if material_prop and material_prop.get("value") is not None:
        validation_result = validate_material_consistency(material_prop["value"], text, category)
        if not validation_result["is_valid"]:
            original_confidence = float(material_prop.get("confidence") or 0.0)
            adjustment = validation_result["confidence_adjustment"]
            new_confidence = max(0.0, min(1.0, original_confidence + adjustment))
            material_prop["confidence"] = new_confidence

            if validation_result["warnings"]:
                material_prop.setdefault("errors", []).extend(validation_result["warnings"])

            log.warning(
                "material_validation_warning",
                extra={
                    "value": material_prop["value"],
                    "original_confidence": original_confidence,
                    "adjusted_confidence": new_confidence,
                    "warnings": validation_result["warnings"],
                },
            )

    def _convert_to_mm(value_str: str | None, unit_str: str | None) -> float | None:
        if not value_str:
            return None
        try:
            value = parse_number_it(value_str)
        except ValueError:
            return None
        unit_key = (unit_str or "").lower()
        multiplier = UNIT_TO_MM.get(unit_key)
        if multiplier is None:
            multiplier = 10.0 if not unit_key else None
        if multiplier is None:
            return None
        return float(value) * multiplier

    def _normalize_dimension_value(raw_value: float | int | None) -> int | None:
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None
        while value > 20000:
            value /= 10.0
        if value <= 0:
            return None
        return int(round(value))

    def _maybe_update_dimension(prop_id: str, candidate_value: float | None) -> None:
        if candidate_value is None:
            return
        normalized_candidate = _normalize_dimension_value(candidate_value)
        if normalized_candidate is None:
            return
        current = properties_payload.get(prop_id)
        replace_current = (
            current is None
            or current.get("value") is None
            or current.get("source") == "qa_llm"
            or float(current.get("confidence") or 0.0) < 0.5
        )
        if not replace_current:
            normalized_existing = _normalize_dimension_value(current.get("value"))
            if normalized_existing is not None:
                current["value"] = normalized_existing
            return
        properties_payload[prop_id] = {
            "value": normalized_candidate,
            "source": "heuristic_dimension",
            "raw": None,
            "span": None,
            "confidence": 0.65,
            "unit": "mm",
            "errors": [],
        }
        log.info(
            "dimension_heuristic_applied",
            extra={"property": prop_id, "value": normalized_candidate},
        )

    for dim_key in DIMENSION_KEYS:
        current = properties_payload.get(dim_key)
        if current and current.get("value") is not None:
            normalized = _normalize_dimension_value(current.get("value"))
            if normalized is not None:
                current["value"] = normalized

    for dim_key in DIMENSION_KEYS:
        candidate = None
        range_pattern = DIMENSION_RANGE_PATTERNS.get(dim_key)
        if range_pattern:
            match = range_pattern.search(text)
            if match:
                candidate = _convert_to_mm(match.group("value"), match.group("unit"))
        if candidate is None:
            label_pattern = DIMENSION_LABEL_PATTERNS.get(dim_key)
            if label_pattern:
                match = label_pattern.search(text)
                if match:
                    candidate = _convert_to_mm(match.group("value"), match.group("unit"))
        if candidate is not None:
            _maybe_update_dimension(dim_key, candidate)



    if category.lower() == "opere_da_cartongessista":
        features = extract_cartongesso_features(text)
        if features:
            _apply_cartongesso_properties(features, text, properties_payload, log)

def _apply_cartongesso_properties(
    features: CartongessoFeatures,
    text: str,
    properties_payload: MutableMapping[str, Dict[str, Any]],
    log: logging.Logger,
) -> None:
    """Populate cartongesso-specific properties from extracted features."""

    def _set_property(
        prop_id: str,
        value: Any,
        *,
        unit: Optional[str] = None,
        raw: Optional[str] = None,
        confidence: float = 0.75,
    ) -> None:
        if value in (None, "", []):
            return
        current = properties_payload.get(prop_id)
        replace = (
            current is None
            or current.get("value") in (None, "")
            or float(current.get("confidence") or 0.0) < 0.6
        )
        if not replace:
            return
        properties_payload[prop_id] = {
            "value": value,
            "source": "parser",
            "raw": raw,
            "span": None,
            "confidence": confidence,
            "unit": unit,
            "errors": [],
        }
        log.info(
            "cartongesso_property_filled",
            extra={"property": prop_id, "value": value, "unit": unit},
        )

    if features.rei_class:
        _set_property("classe_ei", features.rei_class, raw=features.rei_class, confidence=0.8)

    if features.reaction_class:
        _set_property(
            "classe_reazione_al_fuoco",
            features.reaction_class,
            raw=features.reaction_class,
            confidence=0.7,
        )

    presenza_isolante = "si" if features.insulation_thickness_mm else "no"
    insulation_raw_parts: List[str] = []
    if features.insulation_material:
        insulation_raw_parts.append(features.insulation_material)
    if features.insulation_thickness_mm:
        insulation_raw_parts.append(f"{features.insulation_thickness_mm:g} mm")
    if features.insulation_density:
        insulation_raw_parts.append(f"{features.insulation_density:g} kg/m3")
    _set_property(
        "presenza_isolante",
        presenza_isolante,
        raw=" ".join(insulation_raw_parts) or None,
        confidence=0.7,
    )

    summary = summarize_cartongesso_features(features, text=text)
    stratigrafia = summary.get("stratigrafia")
    if stratigrafia:
        import json
        # Convert dict to JSON string for 'raw' field
        raw_str = json.dumps(stratigrafia, ensure_ascii=False)
        _set_property("stratigrafia_lastre", stratigrafia, raw=raw_str, confidence=0.8)

    if not features.reaction_class and "classe a1" in text.lower():
        _set_property("classe_reazione_al_fuoco", "A1", raw="Classe A1", confidence=0.6)

    norma_match = re.search(r"Regolamento[^\n]{0,120}?(?:\([A-Z]+\)|(?=\s+e\s))", text, re.IGNORECASE)
    if norma_match:
        _set_property("normativa_riferimento", norma_match.group(0).strip(), raw=norma_match.group(0).strip())


__all__ = [
    "infer_material",
    "infer_installation_type",
    "apply_domain_heuristics",
    "validate_material_consistency",
    "post_process_properties",
]



