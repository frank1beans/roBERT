from __future__ import annotations
import json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, DefaultDict, Iterable, Optional
from collections import defaultdict

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")

def _read_json(path: Path):
    return json.loads(_read_text(path))

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _slugify(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s_norm = s_norm.lower()
    s_norm = re.sub(r"[^\w\s\-\/]+", "", s_norm)
    s_norm = re.sub(r"[\s\/\-]+", "_", s_norm).strip("_")
    return s_norm

def _looks_like_extractors_list(lst: Any) -> bool:
    if not isinstance(lst, list) or not lst:
        return False
    # Consideriamo "lista di estrattori" se almeno metà degli elementi sono dict con property_id
    hits = sum(1 for x in lst if isinstance(x, dict) and "property_id" in x)
    return hits >= max(1, len(lst) // 2)

def _extractors_from_mapping_like(d: dict) -> Optional[List[dict]]:
    """Se d è un dict i cui valori sono estrattori con property_id, restituisci la lista dei valori."""
    if d and all(isinstance(v, dict) for v in d.values()):
        vals = list(d.values())
        if vals and all("property_id" in v for v in vals):
            return vals
    return None

def _find_extractors_rec(obj: Any) -> Optional[List[dict]]:
    """
    Scansione ricorsiva:
    - se è una lista che 'sembra' estrattori → filtra e ritorna
    - se è un dict:
        * se è mapping-like di estrattori → ritorna values()
        * altrimenti scendi sui valori finché trovi una lista valida
    """
    if isinstance(obj, list):
        if _looks_like_extractors_list(obj):
            return [x for x in obj if isinstance(x, dict) and "property_id" in x]
        # prova a cercare più in profondità (liste annidate)
        for el in obj:
            res = _find_extractors_rec(el)
            if res:
                return res
        return None

    if isinstance(obj, dict):
        # mapping -> lista
        mapped = _extractors_from_mapping_like(obj)
        if mapped:
            return mapped
        # casi noti: "extractors", "patterns", "definitions", "property_extractors"
        for key in ("extractors", "patterns", "definitions", "property_extractors"):
            if key in obj and isinstance(obj[key], list) and _looks_like_extractors_list(obj[key]):
                return [x for x in obj[key] if isinstance(x, dict) and "property_id" in x]
        # altrimenti scendi ricorsivamente
        for v in obj.values():
            res = _find_extractors_rec(v)
            if res:
                return res
        return None

    return None

def _coerce_extractors_list(obj: Any, origin: Path) -> List[dict]:
    """
    Normalizza vari formati in una lista di estrattori (dict con property_id):
      - list pura
      - dict con chiavi 'extractors'/'patterns'/... o mapping per property_id
      - pack annidati (ricerca ricorsiva)
      - JSONL (fallback)
    """
    # 1) lista pura
    if isinstance(obj, list):
        if not _looks_like_extractors_list(obj):
            # filtra comunque solo quelli con property_id
            obj = [x for x in obj if isinstance(x, dict) and "property_id" in x]
        return obj

    # 2) qualunque dict/pack → ricerca ricorsiva
    if isinstance(obj, dict):
        found = _find_extractors_rec(obj)
        if found:
            return found

    # 3) fallback JSONL
    try:
        text = _read_text(origin)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        objs: List[dict] = []
        hits = 0
        for ln in lines:
            if not ln.startswith("{"):
                continue
            try:
                row = json.loads(ln)
                if isinstance(row, dict):
                    objs.append(row)
                    if "property_id" in row:
                        hits += 1
            except Exception:
                pass
        if hits > 0:
            return [x for x in objs if isinstance(x, dict) and "property_id" in x]
    except Exception:
        pass

    raise TypeError(
        "Impossibile individuare la lista di estrattori in "
        f"{origin}. Attesi: lista di dict con 'property_id', oppure pack annidato che la contenga."
    )

def convert_monolith_to_folders(registry_path: Path, extractors_path: Path, out_dir: Path) -> None:
    registry = _read_json(registry_path)
    if not isinstance(registry, dict):
        raise TypeError("registry.json deve essere un dizionario al top-level.")

    raw_extractors = _read_json(extractors_path)
    extractors = _coerce_extractors_list(raw_extractors, extractors_path)

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    # Pre-smistamento extractors: { (super_slug, cat_slug) : [extractor, ...] }
    by_super_cat: DefaultDict[tuple, List[dict]] = defaultdict(list)
    for ext in extractors:
        pid = str(ext.get("property_id", "")).strip()
        parts = pid.split(".")
        if len(parts) < 3:
            by_super_cat[("__orphans__", "__orphans__")].append(ext)
            continue
        super_slug, cat_slug = parts[0], parts[1]
        by_super_cat[(super_slug, cat_slug)].append(ext)

    # Per ogni supercategoria nel registry
    for super_name, sc_obj in registry.items():
        super_dir = root / super_name
        super_dir.mkdir(parents=True, exist_ok=True)

        if not isinstance(sc_obj, dict):
            continue

        # _global registry
        global_slots = (sc_obj.get("_global") or {}).get("slots", {})
        _write_json(super_dir / "_global" / "registry.json", {"slots": global_slots})

        # mappa slug categoria → nome cartella
        categories: Dict[str, Any] = (sc_obj.get("categories") or {})
        slug2catname: Dict[str, str] = {}
        for cat_name, cat_payload in categories.items():
            cat_dir = super_dir / cat_name
            cat_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(cat_payload, dict):
                if "slots" in cat_payload:
                    _write_json(cat_dir / "registry.json", {"slots": cat_payload["slots"]})
                else:
                    _write_json(cat_dir / "registry.json", {"slots": cat_payload})
            else:
                _write_json(cat_dir / "registry.json", {"slots": {}})

            slug2catname[_slugify(cat_name)] = cat_name

        # Slug della super come nei property_id
        super_slug = _slugify(super_name)

        # _global extractors per questa super
        g_exts = by_super_cat.get((super_slug, "__global__"), [])
        if g_exts:
            g_path = super_dir / "_global" / "extractors.json"
            _write_json(g_path, g_exts)

        # extractors di categoria
        for (sup_slug, cat_slug), exts in list(by_super_cat.items()):
            if sup_slug != super_slug or cat_slug in ("__global__", "__orphans__"):
                continue
            cat_name = slug2catname.get(cat_slug)
            if cat_name is None:
                orphans_dir = super_dir / "_orphans"
                orphans_dir.mkdir(parents=True, exist_ok=True)
                o_path = orphans_dir / "extractors.json"
                existing = []
                if o_path.exists():
                    try:
                        existing = json.loads(o_path.read_text(encoding="utf-8-sig"))
                        if not isinstance(existing, list):
                            existing = []
                    except Exception:
                        existing = []
                existing.extend(exts)
                _write_json(o_path, existing)
                continue
            e_path = super_dir / cat_name / "extractors.json"
            _write_json(e_path, exts)

    # Orfani globali
    if ("__orphans__", "__orphans__") in by_super_cat:
        orphans_root = root / "__orphans__"
        _write_json(orphans_root / "extractors.json", by_super_cat[("__orphans__", "__orphans__")])
