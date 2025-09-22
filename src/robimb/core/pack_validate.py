
from __future__ import annotations
import json, os, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from jsonschema import Draft202012Validator

SCHEMA_MAP = {
  "registry": "registry.schema.json",
  "categories": "categories.schema.json",
  "catmap": "catmap.schema.json",
  "extractors": "extractors.schema.json",
  "validators": "validators.schema.json",
  "formulas": "formulas.schema.json",
  "templates": "templates.schema.json",
  "views": "views.schema.json",
  "profiles": "profiles.schema.json",
  "contexts": "contexts.schema.json",
  "manifest": "manifest.schema.json"
}

@dataclass
class FileReport:
  name: str
  path: str
  ok: bool
  errors: List[str] = field(default_factory=list)
  size: int = 0
  sha256: str = ""

@dataclass
class PackReport:
  ok: bool
  files: List[FileReport]

def _sha256(path: str) -> str:
  h=hashlib.sha256()
  with open(path,"rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)
  return h.hexdigest()

def validate_json(obj: dict, schema: dict) -> List[str]:
  v = Draft202012Validator(schema)
  errs = []
  for e in v.iter_errors(obj):
    errs.append(f"{'/'.join([str(p) for p in e.path])}: {e.message}")
  return errs

def validate_pack(pack_json_path: str) -> PackReport:
  # Load pack index
  with open(pack_json_path,"r",encoding="utf-8") as f:
    idx = json.load(f)
  base = Path(pack_json_path).parent
  schemas_dir = Path(__file__).parent / "schemas"

  reports: List[FileReport] = []
  ok_all = True

  for key, rel in idx.get("files", {}).items():
    path = str((base / rel).resolve())
    name = key
    fr = FileReport(name=name, path=path, ok=True)
    if not os.path.exists(path):
      fr.ok = False; fr.errors.append("file not found"); reports.append(fr); ok_all=False; continue
    fr.size = os.path.getsize(path)
    fr.sha256 = _sha256(path)

    # Load JSON
    try:
      with open(path,"r",encoding="utf-8") as f:
        obj = json.load(f)
    except Exception as e:
      fr.ok=False; fr.errors.append(f"json load error: {e}"); reports.append(fr); ok_all=False; continue

    # Validate against schema if available
    schema_name = SCHEMA_MAP.get(key)
    if schema_name:
      with open(schemas_dir / schema_name,"r",encoding="utf-8") as s:
        schema = json.load(s)
      errs = validate_json(obj, schema)
      if errs:
        fr.ok=False; fr.errors += errs
    reports.append(fr)
    ok_all = ok_all and fr.ok

  return PackReport(ok=ok_all, files=reports)
