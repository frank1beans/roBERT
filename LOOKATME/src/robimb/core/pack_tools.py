
from __future__ import annotations
import os, json, shutil, hashlib
from pathlib import Path
from typing import Dict

PACK_KEYS = [
    "registry","catmap","categories","extractors","validators",
    "formulas","templates","views","profiles","contexts",
    "schema_keynote","manifest"
]

def _sha256(path: str) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_manifest(src_dir: str, out_dir: str|None=None) -> Dict:
    src = Path(src_dir)
    out = Path(out_dir) if out_dir else src
    out.mkdir(parents=True, exist_ok=True)

    files = {}
    for key in PACK_KEYS:
        # map key to filename
        name = f"{key}.json" if key != "schema_keynote" else "keynote.schema.json"
        p = src / name
        if p.exists():
            if out != src:
                shutil.copy2(p, out / name)
            files[key] = name

    manifest = {"generated_at": __import__("datetime").datetime.utcnow().isoformat()+"Z", "files":[]}
    for key, name in files.items():
        p = out / name
        manifest["files"].append({
            "key": key, "path": name, "size": p.stat().st_size, "sha256": _sha256(str(p))
        })
    with open(out/"manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    files["manifest"] = "manifest.json"

    return {"out_dir": str(out), "files": files, "manifest": manifest}

def update_current(pack_root: str, version_dir: str):
    root = Path(pack_root)
    current = root/"current"/"pack.json"
    rel = f"../{Path(version_dir).name}/"
    # assemble mapping
    mapping = {
        "version": str(Path(version_dir).name),
        "files": {
            "registry":        rel+"registry.json",
            "catmap":          rel+"catmap.json",
            "categories":      rel+"categories.json",
            "extractors":      rel+"extractors.json",
            "validators":      rel+"validators.json",
            "formulas":        rel+"formulas.json",
            "templates":       rel+"templates.json",
            "views":           rel+"views.json",
            "profiles":        rel+"profiles.json",
            "contexts":        rel+"contexts.json",
            "schema_keynote":  rel+"keynote.schema.json",
            "manifest":        rel+"manifest.json"
        }
    }
    current.parent.mkdir(parents=True, exist_ok=True)
    with open(current,"w",encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return str(current)
