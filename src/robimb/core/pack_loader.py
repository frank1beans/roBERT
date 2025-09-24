"""Loader e contratti per il Knowledge Pack (registry, catmap, ...)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
from typing import Any, Dict

from robimb.extraction import resources as extraction_resources


@dataclass
class KnowledgePack:
    version: str
    generated_at: str | None
    registry: Dict[str, Any]
    catmap: Dict[str, Any]
    categories: Dict[str, Any]
    extractors: Dict[str, Any]
    validators: Dict[str, Any]
    formulas: Dict[str, Any]
    templates: Dict[str, Any]
    views: Dict[str, Any]
    profiles: Dict[str, Any]
    contexts: Dict[str, Any]
    schema_keynote: Dict[str, Any] | None = None
    manifest: Dict[str, Any] | None = None


def _load_old_style(idx: Dict[str, Any], base: pathlib.Path) -> KnowledgePack:
    def load(key: str) -> Dict[str, Any]:
        path = base / idx["files"][key]
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_extractors() -> Dict[str, Any]:
        path = pathlib.Path(idx["files"].get("extractors", ""))
        candidate = (base / path).resolve()
        default_resource = extraction_resources.default_path().resolve()
        if candidate == default_resource:
            return extraction_resources.load_default()
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return extraction_resources.load_default()

    return KnowledgePack(
        version=idx.get("version", ""),
        generated_at=idx.get("generated_at"),
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
        schema_keynote=load("schema_keynote") if "schema_keynote" in idx.get("files", {}) else None,
        manifest=load("manifest") if "manifest" in idx.get("files", {}) else None,
    )


def _as_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _load_inline(payload: Dict[str, Any]) -> KnowledgePack:
    extractors: Dict[str, Any]
    if isinstance(payload.get("extractors"), dict):
        extractors = payload["extractors"]
    elif "patterns" in payload:
        extractors = {"patterns": payload.get("patterns", []), "normalizers": payload.get("normalizers", {})}
    else:
        extractors = {}

    return KnowledgePack(
        version=str(payload.get("version", "")),
        generated_at=payload.get("generated_at"),
        registry=_as_dict(payload, "registry"),
        catmap=_as_dict(payload, "catmap"),
        categories=_as_dict(payload, "categories"),
        extractors=extractors,
        validators=_as_dict(payload, "validators"),
        formulas=_as_dict(payload, "formulas"),
        templates=_as_dict(payload, "templates"),
        views=_as_dict(payload, "views"),
        profiles=_as_dict(payload, "profiles"),
        contexts=_as_dict(payload, "contexts"),
        schema_keynote=_as_dict(payload, "schema_keynote") or None,
        manifest=_as_dict(payload, "manifest") or None,
    )


def load_pack(pack_json_path: str) -> KnowledgePack:
    path = pathlib.Path(pack_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "files" in payload:
        return _load_old_style(payload, path.parent)
    return _load_inline(payload)
