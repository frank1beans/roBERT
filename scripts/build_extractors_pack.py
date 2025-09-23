#!/usr/bin/env python3
"""Generate extractor packs and checklist from the properties registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "properties_registry_extended.json"
OUTPUT_PACK = Path(__file__).resolve().parents[1] / "src" / "robimb" / "extraction" / "resources" / "extractors.json"
OUTPUT_PATTERNS = Path(__file__).resolve().parents[1] / "src" / "robimb" / "extraction" / "resources" / "extractors_patterns.json"
OUTPUT_PACK_CURRENT = Path(__file__).resolve().parents[1] / "pack" / "current" / "pack.json"
OUTPUT_CHECKLIST = Path(__file__).resolve().parents[1] / "docs" / "extraction_checklist.md"

AUGMENT_MAP = {
    "mq": r"(?:mq|m²|m2|metri\\s*quad(?:ri|rati))",
    "m²": r"(?:mq|m²|m2|metri\\s*quad(?:ri|rati))",
    "m3": r"(?:m3|m³|metri\\s*cub(?:i|ici))",
    "m³": r"(?:m3|m³|metri\\s*cub(?:i|ici))",
    "kVA": r"(?:kVA|kva|kilovolt[\\s-]*ampere)",
    "kW": r"(?:kW|kw|kilowatt)",
    "WC": r"(?:WC|wc|water\\s*closet|bagno)",
    "pz": r"(?:pz|pz\\.|pezzi)",
    "gg": r"(?:gg|giorni)",
}

NORMALIZER_DESCRIPTIONS = {
    "comma_to_dot": "replace decimal comma with dot",
    "to_float": "cast to float",
    "to_int": "cast to int",
    "lower": "lowercase string values",
    "strip": "strip leading/trailing whitespace",
    "normalize_unit_symbols": "canonicalise SI unit tokens",
    "split_structured_list": "split textual lists on punctuation and conjunctions",
    "map_yes_no_multilang": "map yes/no multilingual variants to boolean",
    "to_ei_class": "standardise EI fire rating notation",
    "to_strati_count": "extract per-side gypsum board layer count",
}

PRIORITY_CONFIDENCE = 0.85
NON_PRIORITY_CONFIDENCE = 0.6

FALLBACK_PATTERNS = [
    {
        "property_id": "cst.unita_misura",
        "regex": [r"\b(m²|m3|m|kg|pz|cad)\b", r"\b(mq|metri\s*quad(?:ri|rati))\b"],
        "normalizers": ["normalize_unit_symbols", "as_string"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "flr.formato",
        "regex": [r"\b(\d{2,4})\s*[x×]\s*(\d{2,4})\b"],
        "normalizers": ["concat_dims"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "frs.resistenza_fuoco",
        "regex": [r"\b(REI?|EI)\s?(15|30|45|60|90|120|180|240)\b"],
        "normalizers": ["format_EI_from_last_int", "to_ei_class"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "geo.foratura_laterizio",
        "regex": [r"\b(pieno|forato|semi[-\s]?pieno)\b"],
        "normalizers": ["normalize_foratura"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "geo.spessore_elemento",
        "regex": [
            r"\bspessore\s*(\d+(?:[.,]\d+)?)\s*mm\b",
            r"\bspessore\s*(\d+(?:[.,]\d+)?)\s*cm\b",
        ],
        "normalizers": ["comma_to_dot", "to_float", "cm_to_mm?"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "qty.spessore",
        "regex": [r"\bsp\.?\s*(\d+(?:[.,]\d+)?)\s*mm\b", r"\b(\d+(?:[.,]\d+)?)\s*cm\b"],
        "normalizers": ["comma_to_dot", "to_float", "cm_to_mm?"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "aco.rw",
        "regex": [r"\bRw\s*(\d{2})\s*dB\b"],
        "normalizers": ["to_int"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
    {
        "property_id": "opn.trasmittanza_uw",
        "regex": [r"\bUw\s*=?\s*(\d+(?:[.,]\d+)?)\s*W/?m²K\b"],
        "normalizers": ["comma_to_dot", "to_float"],
        "language": "it",
        "confidence": 0.9,
        "tags": ["legacy"],
    },
]


@dataclass
class SlotInfo:
    slot_type: str | None
    is_priority: bool
    regexes: Sequence[str]
    normalizers_override: Sequence[str] | None = None


def load_registry() -> Dict[str, Dict[str, object]]:
    with REGISTRY_PATH.open("r", encoding="utf8") as handle:
        return json.load(handle)


def augment_regexes(regexes: Iterable[str]) -> List[str]:
    augmented: Set[str] = set()
    for rx in regexes:
        augmented.add(rx)
        for needle, replacement in AUGMENT_MAP.items():
            if needle in rx and replacement not in rx:
                augmented.add(rx.replace(needle, replacement))
        if "mm" in rx and "cm" not in rx:
            augmented.add(rx.replace("mm", "cm"))
    return sorted(augmented)


def infer_normalizers(property_id: str, slot: SlotInfo) -> List[str]:
    if slot.normalizers_override is not None:
        return list(slot.normalizers_override)
    normals: List[str] = []
    if slot.slot_type == "float":
        normals.extend(["comma_to_dot", "to_float"])
    elif slot.slot_type == "int":
        normals.append("to_int")
    elif slot.slot_type == "enum":
        normals.append("lower")
    elif slot.slot_type == "bool":
        normals.append("map_yes_no_multilang")
    if any(token in property_id.lower() for token in ("unita", "unità", "unit")):
        normals.append("normalize_unit_symbols")
    if slot.slot_type in {"text", "enum"}:
        if any(sep in rx for rx in slot.regexes for sep in (",", ";", "/")):
            normals.append("split_structured_list")
    if slot.slot_type in {"float", "int"}:
        if any("cm" in rx for rx in slot.regexes):
            property_id_lower = property_id.lower()
            if "_mm" in property_id_lower or property_id_lower.endswith("mm"):
                normals.append("cm_to_mm?")
    if "resistenza_fuoco" in property_id or property_id.lower().startswith("frs.rei"):
        normals.append("format_EI_from_last_int")
        normals.append("to_ei_class")
    # Deduplicate preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for name in normals:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def split_category_tags(category: str) -> List[str]:
    parts = [part.strip() for part in category.split("|") if part.strip()]
    tags: List[str] = []
    if parts:
        tags.append(f"category:{parts[0]}")
    if len(parts) > 1:
        tags.append(f"subcategory:{parts[1]}")
    return tags


def build_patterns(registry: Dict[str, Dict[str, object]]):
    aggregated: Dict[str, Dict[str, object]] = {}
    checklist_sections: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    checklist_sections.append("# Checklist estrazione proprietà prioritari\n")
    checklist_sections.append(
        "Generata automaticamente il {} a partire da `data/properties_registry_extended.json`.\n".format(
            timestamp
        )
    )
    checklist_sections.append(
        "Per ogni categoria sono elencati gli slot prioritari strutturati (numerici, enum, stringhe) con le regex di supporto e i normalizzatori suggeriti.\n"
    )

    for category, payload in sorted(registry.items()):
        if category == "_schema":
            continue
        priority: Sequence[str] = payload.get("priority", [])  # type: ignore[assignment]
        slots: Dict[str, Dict[str, object]] = payload.get("slots", {})  # type: ignore[assignment]
        patterns: Dict[str, Sequence[str]] = payload.get("patterns", {})  # type: ignore[assignment]
        tags = split_category_tags(category)

        checklist_sections.append(f"\n## {category}\n")
        checklist_sections.append("| Proprietà | Tipo | Regex | Normalizzatori |\n")
        checklist_sections.append("| --- | --- | --- | --- |\n")

        for prop in priority:
            slot_spec = slots.get(prop)
            if not slot_spec:
                continue
            slot_type = slot_spec.get("type") if isinstance(slot_spec, dict) else None
            if slot_type not in {"float", "int", "enum", "bool", "text"}:
                continue
            regexes = augment_regexes(patterns.get(prop, []))
            if not regexes:
                continue
            override = slot_spec.get("normalizers") if isinstance(slot_spec, dict) else None
            normals_override = list(override) if isinstance(override, (list, tuple)) else None
            slot = SlotInfo(
                slot_type=slot_type,
                is_priority=True,
                regexes=regexes,
                normalizers_override=normals_override,
            )
            normalizers = infer_normalizers(prop, slot)
            regex_repr = "<br>".join(f"`{rx}`" for rx in regexes)
            norm_repr = ", ".join(f"`{name}`" for name in normalizers) if normalizers else "—"
            checklist_sections.append(
                f"| `{prop}` | {slot_type or '—'} | {regex_repr} | {norm_repr} |\n"
            )

            entry = aggregated.setdefault(
                prop,
                {
                    "regex": set(),
                    "normalizers": [],
                    "tags": set(),
                    "confidence_scores": [],
                    "slot_types": set(),
                    "language": "it",
                    "priority_categories": set(),
                    "categories": set(),
                },
            )
            entry["regex"].update(regexes)  # type: ignore[index]
            entry["slot_types"].add(slot_type)  # type: ignore[index]
            entry["categories"].add(category)  # type: ignore[index]
            if slot.is_priority:
                entry["priority_categories"].add(category)  # type: ignore[index]
            entry["confidence_scores"].append(PRIORITY_CONFIDENCE if slot.is_priority else NON_PRIORITY_CONFIDENCE)  # type: ignore[index]
            normals = infer_normalizers(prop, slot)
            current_normals: List[str] = entry.setdefault("normalizers", [])  # type: ignore[index]
            for name in normals:
                if name not in current_normals:
                    current_normals.append(name)
            entry["tags"].update(tags)  # type: ignore[index]
            entry["tags"].add(f"slot_type:{slot_type}")  # type: ignore[index]
            entry["tags"].add("priority")  # type: ignore[index]

        # Add non-priority structured slots if regex present
        for prop, regexes in patterns.items():
            if prop in priority:
                continue
            slot_spec = slots.get(prop)
            if not slot_spec:
                continue
            slot_type = slot_spec.get("type") if isinstance(slot_spec, dict) else None
            if slot_type not in {"float", "int", "enum", "bool", "text"}:
                continue
            augmented_regexes = augment_regexes(regexes)
            if not augmented_regexes:
                continue
            override = slot_spec.get("normalizers") if isinstance(slot_spec, dict) else None
            normals_override = list(override) if isinstance(override, (list, tuple)) else None
            slot = SlotInfo(
                slot_type=slot_type,
                is_priority=False,
                regexes=augmented_regexes,
                normalizers_override=normals_override,
            )
            entry = aggregated.setdefault(
                prop,
                {
                    "regex": set(),
                    "normalizers": [],
                    "tags": set(),
                    "confidence_scores": [],
                    "slot_types": set(),
                    "language": "it",
                    "priority_categories": set(),
                    "categories": set(),
                },
            )
            entry["regex"].update(augmented_regexes)  # type: ignore[index]
            entry["slot_types"].add(slot_type)  # type: ignore[index]
            entry["categories"].add(category)  # type: ignore[index]
            entry["confidence_scores"].append(NON_PRIORITY_CONFIDENCE)  # type: ignore[index]
            normals = infer_normalizers(prop, slot)
            current_normals: List[str] = entry.setdefault("normalizers", [])  # type: ignore[index]
            for name in normals:
                if name not in current_normals:
                    current_normals.append(name)
            entry["tags"].update(tags)  # type: ignore[index]
            entry["tags"].add(f"slot_type:{slot_type}")  # type: ignore[index]

    return aggregated, "".join(checklist_sections)


def serialise_patterns(aggregated: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    serialised: List[Dict[str, object]] = []
    for property_id, info in aggregated.items():
        regexes = sorted(info["regex"])  # type: ignore[index]
        normalizers = info.get("normalizers", [])
        tags = sorted(info.get("tags", []))  # type: ignore[arg-type]
        scores = info.get("confidence_scores", [])
        confidence = max(scores) if scores else NON_PRIORITY_CONFIDENCE
        serialised.append(
            {
                "property_id": property_id,
                "regex": regexes,
                "normalizers": normalizers,
                "language": info.get("language", "it"),
                "confidence": round(confidence, 2),
                "tags": tags,
            }
        )
    serialised.sort(key=lambda item: item["property_id"])
    return serialised


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_checklist(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        handle.write(content)


def main() -> None:
    registry = load_registry()
    aggregated, checklist = build_patterns(registry)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    patterns = serialise_patterns(aggregated)
    by_id = {item["property_id"]: item for item in patterns}
    for fallback in FALLBACK_PATTERNS:
        current = by_id.get(fallback["property_id"])
        if current is None:
            patterns.append(dict(fallback))
            by_id[fallback["property_id"]] = patterns[-1]
            continue
        current_regex = set(current.get("regex", []))
        current_regex.update(fallback.get("regex", []))
        current["regex"] = sorted(current_regex)
        merged_normalizers = list(dict.fromkeys(current.get("normalizers", []) + fallback.get("normalizers", [])))
        current["normalizers"] = merged_normalizers
        current["tags"] = sorted(set(current.get("tags", [])) | set(fallback.get("tags", [])))
        current["confidence"] = round(max(current.get("confidence", NON_PRIORITY_CONFIDENCE), fallback.get("confidence", NON_PRIORITY_CONFIDENCE)), 2)
    
    patterns.sort(key=lambda item: item["property_id"])

    pack_payload = {
        "version": "0.2.0",
        "generated_at": timestamp,
        "patterns": patterns,
        "normalizers": NORMALIZER_DESCRIPTIONS,
    }

    write_json(OUTPUT_PACK, pack_payload)
    write_json(OUTPUT_PATTERNS, pack_payload)
    write_json(OUTPUT_PACK_CURRENT, pack_payload)
    write_checklist(OUTPUT_CHECKLIST, checklist)


if __name__ == "__main__":
    main()
