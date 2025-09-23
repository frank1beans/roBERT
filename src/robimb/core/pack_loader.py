
"""Loader e contratti per il Knowledge Pack (registry, catmap, ...)."""
from dataclasses import dataclass
import json, pathlib

from robimb.extraction import resources as extraction_resources
@dataclass
class KnowledgePack:
    registry: dict
    catmap: dict
    categories: dict
    extractors: dict
    validators: dict
    formulas: dict
    templates: dict
    views: dict
    profiles: dict
    contexts: dict
    schema_keynote: dict | None = None
    manifest: dict | None = None

def load_pack(pack_json_path: str) -> KnowledgePack:
    with open(pack_json_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    base = pathlib.Path(pack_json_path).parent
    def load(key):
        p = base / idx["files"][key]
        with open(p, "r", encoding="utf-8") as fh: return json.load(fh)

    def load_extractors():
        path = pathlib.Path(idx["files"].get("extractors", ""))
        candidate = (base / path).resolve()
        default_resource = extraction_resources.default_path().resolve()
        if candidate == default_resource:
            return extraction_resources.load_default()
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return extraction_resources.load_default()
    return KnowledgePack(
        registry=load("registry"),
        catmap=load("catmap"),
        categories=load("categories"),
        extractors=load_extractors(),
        validators=load("validators"),
        formulas=load("formulas"),
        templates=load("templates"),
        views=load("views"),
        profiles=load("profiles"),
        contexts=load("contexts"),
        schema_keynote=load("schema_keynote") if "schema_keynote" in idx["files"] else None,
        manifest=load("manifest") if "manifest" in idx["files"] else None,
    )
