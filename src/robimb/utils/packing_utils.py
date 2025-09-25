import re, json
from pathlib import Path

def slugify_label(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("__", "")
    return s

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj):
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def split_property_id(full_property_id: str):
    # es. "opere_da_cartongessista.global.modello" oppure "opere_da_cartongessista.categoria.slot"
    parts = full_property_id.split(".")
    if len(parts) < 3:
        return None, None, parts[-1]
    return parts[0], parts[1], parts[-1]

def trim_property_id_for_global(prop_id: str):
    _, cog, slot = split_property_id(prop_id)
    return f"global.{slot}" if cog == "global" else slot

def trim_property_id_for_category(prop_id: str):
    return split_property_id(prop_id)[-1]