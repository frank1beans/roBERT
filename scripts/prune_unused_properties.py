#!/usr/bin/env python3
"""Prune property extractors and registry slots for unused properties."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set

TARGET_SUPERS = {
    "massetti_sottofondi_drenaggi_vespai",
    "tetti_manti_di_copertura_e_opere_accessorie",
    "opere_di_impermeabilizzazione",
    "sistemi_oscuranti_per_facciate",
    "controsoffitti",
}


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def load_unused_properties(path: Path) -> List[str]:
    with path.open() as f:
        data = json.load(f)
    return data.get("unused_property_ids", [])


def build_super_directory_map(modular_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for super_dir in modular_root.iterdir():
        if not super_dir.is_dir() or super_dir.name == "metadata":
            continue
        mapping[slugify(super_dir.name)] = super_dir
    return mapping


def prune_patterns(path: Path, property_ids: Set[str]) -> bool:
    if not path.exists():
        return False
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        original_len = len(data)
        data = [entry for entry in data if entry.get("property_id") not in property_ids]
        if len(data) == original_len:
            return False
        with path.open("w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return True
    if isinstance(data, dict):
        patterns = data.get("patterns")
        if not isinstance(patterns, list):
            return False
        original_len = len(patterns)
        patterns = [
            entry for entry in patterns if entry.get("property_id") not in property_ids
        ]
        if len(patterns) == original_len:
            return False
        data["patterns"] = patterns
        with path.open("w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        return True
    return False


def prune_registry_slots(path: Path, slots_to_remove: Set[str]) -> bool:
    if not path.exists() or not slots_to_remove:
        return False
    with path.open() as f:
        data = json.load(f)
    slots = data.get("slots")
    if not isinstance(slots, dict):
        return False
    changed = False
    for slot in list(slots.keys()):
        if slot in slots_to_remove:
            del slots[slot]
            changed = True
    if not changed:
        return False
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return True


def main() -> int:
    report_path = Path("data/properties/unused_properties_report.json")
    unused_properties = load_unused_properties(report_path)

    targets = [
        pid
        for pid in unused_properties
        if pid.split(".", 1)[0] in TARGET_SUPERS
    ]
    if not targets:
        print("No unused properties found for target supers.")
        return 0

    to_remove = set(targets)

    extractors_path = Path("data/properties/extractors.json")
    changed_main = prune_patterns(extractors_path, to_remove)
    print(f"Main extractors updated: {changed_main}")

    modular_root = Path("data/properties/modular")
    super_map = build_super_directory_map(modular_root)

    registry_changes = 0
    extractor_changes = 0

    for pid in targets:
        super_id, rest = pid.split(".", 1)
        category_id, slot = rest.split(".", 1)
        slot_name = slot.rsplit(".", 1)[-1] if "." in slot else slot
        slot_name = slot_name
        super_dir = super_map.get(super_id)
        if super_dir is None:
            raise SystemExit(f"Unable to map super id {super_id} to directory")
        if category_id == "__global__":
            category_dir = super_dir / "_global"
        else:
            category_dir = None
            for path in super_dir.iterdir():
                if not path.is_dir() or path.name in {"_global", "_orphans"}:
                    continue
                if slugify(path.name) == category_id:
                    category_dir = path
                    break
            if category_dir is None:
                raise SystemExit(
                    f"Unable to find category directory for {pid} (super_dir={super_dir})"
                )
        extractor_path = category_dir / "extractors.json"
        if prune_patterns(extractor_path, {pid}):
            extractor_changes += 1
        registry_path = category_dir / "registry.json"
        if prune_registry_slots(registry_path, {slot_name}):
            registry_changes += 1

    print(f"Updated {extractor_changes} modular extractor files and {registry_changes} registry files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
