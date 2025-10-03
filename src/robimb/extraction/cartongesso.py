"""Cartongesso feature extraction and catalog matching."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = _PROJECT_ROOT / "resources" / "data" / "catalogs" / "cartongesso_contropareti.csv"

_LAYER_KEYWORDS = {
    "standard": ["standard", "gkb"],
    "idrofuga": ["idrolastra", "idrofuga", "gki", "h2o"],
    "ignifuga": ["ignilastra", "antincendio", "gkf", "a-zero", "fireguard"],
    "fonoisolante": ["fonoisolante", "fono", "acustica"],
    "fibrogesso": ["fibrogesso"],
    "accoppiata_isolante": ["accoppiata", "sandwich"],
}

_INSULATION_KEYWORDS = {
    "lana minerale": ["lana minerale", "mineral wool"],
    "lana di roccia": ["lana di roccia"],
}

_REACTION_CLASSES = [
    "A1",
    "A2-s1,d0",
    "A2-s2,d0",
    "B-s1,d0",
    "B-s2,d0",
]

_UNIT_FACTORS = {
    "mm": 1.0,
    "millimetri": 1.0,
    "cm": 10.0,
    "centimetri": 10.0,
}

_MM_RE = re.compile(r"(?P<value>\d+[.,]?\d*)\s*(?P<unit>mm|cm|millimetri|centimetri)", re.IGNORECASE)
_EI_RE = re.compile(r"\b(REI|EI)\s*-?\s*(\d{2,3})", re.IGNORECASE)
_PASSO_RE = re.compile(r"passo[^\d]*(\d+)\s*cm", re.IGNORECASE)
_DENSITY_RE = re.compile(r"densit[aà]\s*[:\s]*(\d+[.,]?\d*)\s*(?:kg\/m[c3]|kg\s*mc)", re.IGNORECASE)
_MONTANTE_RE = re.compile(r"montant[ei][^\d]*(\d+[.,]?\d*)\s*mm", re.IGNORECASE)
_ORIDITURA_RE = re.compile(r"orditura[^\n]{0,50}?(?:c\s*(\d{2,3})[/x]\d{2,3}|da\s+(\d+[.,]?\d*)\s*mm)", re.IGNORECASE)
_LAYER_LINE_RE = re.compile(r"(?:lastra|idrolastra|ignilastra)[^\n]*sp\.?\s*(\d+[.,]?\d*)\s*(mm|cm)", re.IGNORECASE)


@dataclass(frozen=True)
class CartongessoLayer:
    """Single layer within a cartongesso system."""

    raw: str
    type: str
    thickness_mm: float


@dataclass(frozen=True)
class CartongessoFeatures:
    """Structured representation extracted from a description."""

    layers: List[CartongessoLayer]
    frame_width_mm: Optional[float]
    frame_step_cm: Optional[int]
    insulation_material: Optional[str]
    insulation_thickness_mm: Optional[float]
    insulation_density: Optional[float]
    rei_class: Optional[str]
    reaction_class: Optional[str]
    catalog_matches: Dict[str, List[str]]

    @property
    def total_thickness_mm(self) -> Optional[float]:
        if not self.layers:
            return None
        total = sum(layer.thickness_mm for layer in self.layers)
        if self.frame_width_mm:
            total += self.frame_width_mm
        return total


class CartongessoCatalog:
    """Utility to load and match catalog configurations."""

    def __init__(self, rows: List[Dict[str, str]]):
        self._rows = rows

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> "CartongessoCatalog":
        if not CATALOG_PATH.exists():
            return cls([])
        with CATALOG_PATH.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            rows = list(reader)
        return cls(rows)

    def match_layers(self, layers: Iterable[CartongessoLayer], frame_width: Optional[float]) -> Dict[str, List[str]]:
        matches: Dict[str, List[str]] = {}
        for layer in layers:
            candidates: List[str] = []
            for row in self._rows:
                montante = row.get("Montante", "")
                if frame_width and montante:
                    width_match = re.search(r"(\d{2,3})", montante)
                    if width_match and abs(float(width_match.group(1)) - frame_width) > 5:
                        continue
                tipologia = (row.get("Tipologia_Lastra") or "").lower()
                if layer.type and layer.type not in tipologia:
                    continue
                try:
                    thickness = float(row.get("Spessore_Lastra_mm", "0"))
                except ValueError:
                    continue
                if abs(thickness - layer.thickness_mm) > 1.0:
                    continue
                codice = row.get("Codice")
                if codice:
                    candidates.append(codice)
            matches[layer.raw] = candidates
        return matches


def _normalize_lines(text: str) -> List[str]:
    normalized = re.sub(r"\s*-\s+(?=[A-Za-z])", "\n- ", text)
    lines: List[str] = []
    for raw in normalized.splitlines():
        cleaned = raw.strip(" \\t-;:")
        if cleaned:
            lines.append(cleaned)
    return lines


def _parse_number(value: str) -> float:
    return float(value.replace(",", "."))


def _to_mm(match: Optional[re.Match[str]]) -> Optional[float]:
    if not match:
        return None
    value = _parse_number(match.group("value"))
    unit = match.group("unit").lower()
    factor = _UNIT_FACTORS.get(unit, 1.0)
    return value * factor


def _detect_layer_type(text: str) -> str:
    lower = text.lower()
    for canonical, keywords in _LAYER_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return canonical
    return "standard"


def _extract_layers(lines: List[str]) -> List[CartongessoLayer]:
    layers: List[CartongessoLayer] = []
    for line in lines:
        lower = line.lower()

        # Skip description lines that describe the overall wall (e.g., "Parete in cartongesso...")
        # We want only the actual layer specifications (bullet points)
        if re.match(r'^(?:parete|fornitura|controparete|controsoffitto)', lower):
            continue

        # Must contain lastra/idrolastra/ignilastra
        if not re.search(r'\b(?:lastra|idrolastra|ignilastra)\b', lower):
            continue

        match = _LAYER_LINE_RE.search(line)
        if not match:
            continue
        thickness = _parse_number(match.group(1))
        if match.group(2).lower().startswith("c"):
            thickness *= 10
        layer = CartongessoLayer(raw=line, type=_detect_layer_type(line), thickness_mm=thickness)
        layers.append(layer)
    return layers


def _extract_frames(lines: List[str]) -> List[float]:
    """Extract all frame widths from lines."""
    frames: List[float] = []
    for line in lines:
        lower = line.lower()

        # Skip description lines
        if re.match(r'^(?:parete|fornitura|controparete|controsoffitto)', lower):
            continue

        if "orditura" not in lower:
            continue
        match = _ORIDITURA_RE.search(line)
        if match:
            if match.group(1):
                try:
                    frames.append(float(match.group(1)))
                except ValueError:
                    pass
            elif match.group(2):
                frames.append(_parse_number(match.group(2)))
    return frames


def _extract_frame(lines: List[str], text: str) -> Dict[str, Optional[float]]:
    """Extract first frame width (legacy compatibility)."""
    match = _ORIDITURA_RE.search(text)
    frame_width = None
    frame_code = None
    if match:
        if match.group(1):
            frame_code = f"C {match.group(1)}"
            try:
                frame_width = float(match.group(1))
            except ValueError:
                frame_width = None
        elif match.group(2):
            frame_width = _parse_number(match.group(2))
    if frame_width is None:
        montante = _MONTANTE_RE.search(text)
        if montante:
            frame_width = _parse_number(montante.group(1))
    step_match = _PASSO_RE.search(text)
    frame_step = int(step_match.group(1)) if step_match else None
    return {
        "width_mm": frame_width,
        "code": frame_code,
        "step_cm": frame_step,
    }


@dataclass(frozen=True)
class InsulationLayer:
    """Single insulation layer."""
    material: Optional[str]
    thickness_mm: Optional[float]
    density_kg_m3: Optional[float]


def _extract_insulations(lines: List[str]) -> List[InsulationLayer]:
    """Extract all insulation layers from lines."""
    insulations: List[InsulationLayer] = []
    for line in lines:
        lower = line.lower()

        # Skip description lines
        if re.match(r'^(?:parete|fornitura|controparete|controsoffitto)', lower):
            continue

        if "isolant" not in lower:
            continue
        material = None
        for canonical, keywords in _INSULATION_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                material = canonical
                break
        thickness = _to_mm(_MM_RE.search(line))
        density_match = _DENSITY_RE.search(line)
        density = _parse_number(density_match.group(1)) if density_match else None
        insulations.append(InsulationLayer(
            material=material,
            thickness_mm=thickness,
            density_kg_m3=density
        ))
    return insulations


def _extract_insulation(lines: List[str]) -> Dict[str, Optional[float]]:
    """Extract first insulation (legacy compatibility)."""
    for line in lines:
        lower = line.lower()
        if "isolant" not in lower:
            continue
        material = None
        for canonical, keywords in _INSULATION_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                material = canonical
                break
        thickness = _to_mm(_MM_RE.search(line))
        density_match = _DENSITY_RE.search(line)
        density = _parse_number(density_match.group(1)) if density_match else None
        return {
            "raw": line,
            "material": material,
            "thickness_mm": thickness,
            "density_kg_m3": density,
        }
    return {
        "raw": None,
        "material": None,
        "thickness_mm": None,
        "density_kg_m3": None,
    }


def _extract_rei(text: str) -> Optional[str]:
    match = _EI_RE.search(text)
    if not match:
        return None
    value = match.group(2)
    return f"EI{value}"


def _extract_reaction_class(text: str) -> Optional[str]:
    for reaction in _REACTION_CLASSES:
        if reaction.lower() in text.lower():
            return reaction
    return None


def _build_stratigrafia(
    layers: List[CartongessoLayer],
    frames: List[float],
    insulations: List[InsulationLayer],
) -> Dict[str, any]:
    """Build detailed stratigraphy as structured dictionary.

    Returns:
        Dict with:
        - lastre: List of layer objects with id, spessore_mm, tipologia
        - orditure: List of frame objects with id, larghezza_mm
        - isolanti: List of insulation objects with id, materiale, spessore_mm, densita_kg_m3
        - sequenza: Human-readable sequence string
    """
    lastre_list: List[Dict[str, any]] = []
    orditure_list: List[Dict[str, any]] = []
    isolanti_list: List[Dict[str, any]] = []
    sequenza_parts: List[str] = []

    layer_counter = 0
    frame_counter = 0
    insulation_counter = 0

    # Group layers by position (we assume frame/insulation is in the middle)
    # For walls: layers on one side, then frame+insulation, then layers on other side
    # Simple heuristic: if we have 2+ layers, put frame/insulation in middle

    if len(layers) >= 2:
        # First layer(s)
        mid_point = len(layers) // 2

        # Layers before frame/insulation
        for i in range(mid_point):
            layer_counter += 1
            layer = layers[i]
            layer_id = f"lastra_{layer_counter}"
            lastre_list.append({
                "id": layer_id,
                "spessore_mm": layer.thickness_mm,
                "tipologia": layer.type
            })
            sequenza_parts.append(layer_id)

        # All frames
        for frame_width in frames:
            frame_counter += 1
            frame_id = f"orditura_{frame_counter}"
            orditure_list.append({
                "id": frame_id,
                "larghezza_mm": frame_width
            })
            sequenza_parts.append(frame_id)

        # All insulations
        for insulation in insulations:
            if insulation.thickness_mm:
                insulation_counter += 1
                isolante_id = f"isolante_{insulation_counter}"
                isolante_obj = {
                    "id": isolante_id,
                    "spessore_mm": insulation.thickness_mm
                }
                if insulation.material:
                    isolante_obj["materiale"] = insulation.material
                if insulation.density_kg_m3:
                    isolante_obj["densita_kg_m3"] = insulation.density_kg_m3
                isolanti_list.append(isolante_obj)
                sequenza_parts.append(isolante_id)

        # Remaining layers after frame/insulation
        for i in range(mid_point, len(layers)):
            layer_counter += 1
            layer = layers[i]
            layer_id = f"lastra_{layer_counter}"
            lastre_list.append({
                "id": layer_id,
                "spessore_mm": layer.thickness_mm,
                "tipologia": layer.type
            })
            sequenza_parts.append(layer_id)
    else:
        # Simple case: just list all layers
        for layer in layers:
            layer_counter += 1
            layer_id = f"lastra_{layer_counter}"
            lastre_list.append({
                "id": layer_id,
                "spessore_mm": layer.thickness_mm,
                "tipologia": layer.type
            })
            sequenza_parts.append(layer_id)

        # All frames
        for frame_width in frames:
            frame_counter += 1
            frame_id = f"orditura_{frame_counter}"
            orditure_list.append({
                "id": frame_id,
                "larghezza_mm": frame_width
            })
            sequenza_parts.append(frame_id)

        # All insulations
        for insulation in insulations:
            if insulation.thickness_mm:
                insulation_counter += 1
                isolante_id = f"isolante_{insulation_counter}"
                isolante_obj = {
                    "id": isolante_id,
                    "spessore_mm": insulation.thickness_mm
                }
                if insulation.material:
                    isolante_obj["materiale"] = insulation.material
                if insulation.density_kg_m3:
                    isolante_obj["densita_kg_m3"] = insulation.density_kg_m3
                isolanti_list.append(isolante_obj)
                sequenza_parts.append(isolante_id)

    return {
        "lastre": lastre_list,
        "orditure": orditure_list,
        "isolanti": isolanti_list,
        "sequenza": " + ".join(sequenza_parts)
    }


def extract_cartongesso_features(text: str) -> Optional[CartongessoFeatures]:
    lines = _normalize_lines(text)
    layers = _extract_layers(lines)
    if not layers:
        return None
    frame = _extract_frame(lines, text)
    insulation_info = _extract_insulation(lines)
    rei = _extract_rei(text)
    reaction = _extract_reaction_class(text)
    catalog = CartongessoCatalog.load()
    matches = catalog.match_layers(layers, frame.get("width_mm")) if catalog else {}
    return CartongessoFeatures(
        layers=layers,
        frame_width_mm=frame.get("width_mm"),
        frame_step_cm=frame.get("step_cm"),
        insulation_material=insulation_info["material"],
        insulation_thickness_mm=insulation_info["thickness_mm"],
        insulation_density=insulation_info["density_kg_m3"],
        rei_class=rei,
        reaction_class=reaction,
        catalog_matches=matches,
    )


def summarize_cartongesso_features(features: CartongessoFeatures, text: str = "") -> Dict[str, any]:
    """Summarize cartongesso features including all frames and insulations.

    Args:
        features: Extracted features from extract_cartongesso_features
        text: Original text (needed to extract all frames/insulations)
    """
    summary: Dict[str, any] = {}

    # Extract all frames and insulations from text
    lines = _normalize_lines(text) if text else []
    frames = _extract_frames(lines) if lines else []
    insulations = _extract_insulations(lines) if lines else []

    # Fall back to single values from features if extraction failed
    if not frames and features.frame_width_mm:
        frames = [features.frame_width_mm]
    if not insulations and features.insulation_thickness_mm:
        insulations = [InsulationLayer(
            material=features.insulation_material,
            thickness_mm=features.insulation_thickness_mm,
            density_kg_m3=features.insulation_density
        )]

    summary["stratigrafia"] = _build_stratigrafia(
        features.layers,
        frames,
        insulations,
    )

    if features.insulation_material:
        insulation_parts = [features.insulation_material]
        if features.insulation_thickness_mm:
            insulation_parts.append(f"{features.insulation_thickness_mm:g} mm")
        if features.insulation_density:
            insulation_parts.append(f"{features.insulation_density:g} kg/m3")
        summary["isolante"] = " ".join(insulation_parts)
    if features.catalog_matches:
        matched_codes = [code for codes in features.catalog_matches.values() for code in codes]
        if matched_codes:
            summary["catalogo"] = ", ".join(sorted(set(matched_codes))[:5])
    return summary



