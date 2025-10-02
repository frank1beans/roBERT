"""Domain-specific heuristics for property extraction.

Questo modulo fornisce regole euristiche basate su conoscenza del dominio BIM
per inferire proprietà quando rules/matchers/LLM falliscono.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional


# Regole per materiali basate su keywords nel testo
MATERIAL_KEYWORDS = {
    # Legno
    r"\bdoghe\b": "legno",
    r"\blegno\b": "legno",
    r"\bteak\b": "legno_teak",
    r"\blaminato\b": "legno_laminato",

    # Vetro (più specifici prima)
    r"\b(?:cristallo|vetro)\s+temperato\b": "vetro_temperato",
    r"\bcristallo\b": "vetro",
    r"\bvetro\b": "vetro",

    # Metalli
    r"\bAISI\s*304\b": "acciaio_inox",
    r"\bacciaio\s+inox\b": "acciaio_inox",
    r"\binox\b": "acciaio_inox",
    r"\bacciaio\b": "acciaio",
    r"\bcromatx?[oa]\b": "metallo_cromato",  # cromato/cromata
    r"\bottone\b": "metallo_ottone",
    r"\balluminio\b": "alluminio",

    # Plastica/PVC (più specifici prima)
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

    # Plastica
    r"\b(piletta|sifone)\b": "plastica",
}

# Tipologia installazione da keywords
INSTALLATION_TYPE_KEYWORDS = {
    r"\ba\s+parete\b": "a_parete",
    r"\bparete\b": "a_parete",
    r"\bsospeso\b": "sospesa",
    r"\ba\s+pavimento\b": "a_pavimento",
    r"\bpavimento\b": "a_pavimento",
    r"\bincasso\b": "ad_incasso",
    r"\bfilo\s+parete\b": "filo_parete",
    r"\bangolo\b": "angolare",
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
            # Confidence più alta se match esplicito
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
            # Confidence più bassa (inferenza)
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
    """Applica euristiche dominio per riempire gap nelle proprietà.

    Questa funzione viene chiamata DOPO rules/matchers e PRIMA di LLM,
    per cercare di riempire proprietà mancanti con conoscenza dominio.

    Args:
        text: Testo descrizione
        category: Categoria BIM
        existing_properties: Proprietà già estratte

    Returns:
        Dict con proprietà inferite (solo quelle mancanti/null)
    """
    inferred = {}

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

    # Check 3: AISI 304 → acciaio inox
    if "aisi" in text_lower or "inox" in text_lower:
        if "acciaio" not in material_value.lower():
            result["warnings"].append(
                "Testo menziona AISI/inox ma materiale non è acciaio"
            )
            result["confidence_adjustment"] = -0.3

    return result


__all__ = [
    "infer_material",
    "infer_installation_type",
    "apply_domain_heuristics",
    "validate_material_consistency",
]
