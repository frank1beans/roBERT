from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

def _read_json_if_exists(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8-sig"))
    return None

def pack_folders_to_monolith(properties_root: Path, out_registry: Path, out_extractors: Path) -> None:
    registry: Dict[str, Any] = {}
    extractors_all: List[Dict[str, Any]] = []

    for sc_dir in sorted([p for p in properties_root.iterdir() if p.is_dir()]):
        super_name = sc_dir.name
        reg_sc: Dict[str, Any] = {"_global": {"slots": {}}, "categories": {}}

        # _global registry
        g = _read_json_if_exists(sc_dir / "_global" / "registry.json")
        if isinstance(g, dict):
            reg_sc["_global"]["slots"] = g.get("slots", {})

        # categorie
        for cat_dir in sorted([p for p in sc_dir.iterdir() if p.is_dir() and p.name != "_global"]):
            cat_name = cat_dir.name
            rj = _read_json_if_exists(cat_dir / "registry.json")
            if isinstance(rj, dict):
                reg_sc["categories"][cat_name] = rj

            ej = _read_json_if_exists(cat_dir / "extractors.json")
            if isinstance(ej, list):
                extractors_all.extend(ej)

        # anche gli extractors globali
        eg = _read_json_if_exists(sc_dir / "_global" / "extractors.json")
        if isinstance(eg, list):
            extractors_all.extend(eg)

        registry[super_name] = reg_sc

    out_registry.parent.mkdir(parents=True, exist_ok=True)
    out_extractors.parent.mkdir(parents=True, exist_ok=True)
    out_registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    out_extractors.write_text(json.dumps(extractors_all, ensure_ascii=False, indent=2), encoding="utf-8")