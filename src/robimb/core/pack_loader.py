"""Loader e contratti per il Knowledge Pack (registry, catmap, ...)."""
from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
from typing import Any, Dict, Iterable, Mapping, Tuple

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


def _merge_mapping(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(base, Mapping):
        for key, value in base.items():
            if isinstance(value, Mapping):
                result[key] = dict(value)
            elif isinstance(value, list):
                result[key] = list(value)
            else:
                result[key] = value
    if isinstance(override, Mapping):
        for key, value in override.items():
            if key == "slots" and isinstance(value, Mapping):
                slots = dict(result.get("slots", {}))
                slots.update(value)
                result["slots"] = slots
            elif isinstance(value, Mapping):
                base_value = result.get(key)
                if isinstance(base_value, dict):
                    result[key] = {**base_value, **value}
                else:
                    result[key] = dict(value)
            elif isinstance(value, list):
                base_list = result.get(key)
                if isinstance(base_list, list):
                    result[key] = base_list + [item for item in value if item not in base_list]
                else:
                    result[key] = list(value)
            else:
                result[key] = value
    return result


def _flatten_v4_registry(payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str, str | None, Dict[str, Any]]:
    metadata = payload.get("metadata")
    version = ""
    generated_at: str | None = None
    if isinstance(metadata, Mapping):
        version = str(metadata.get("version", ""))
        generated_at = metadata.get("generated_at") or metadata.get("timestamp")

    registry: Dict[str, Any] = {}
    categories: Dict[str, Any] = {}
    catmap: Dict[str, Any] = {"mappings": []}

    for super_name, super_payload in payload.items():
        if super_name.startswith("_") or super_name == "metadata":
            continue
        if not isinstance(super_payload, Mapping):
            continue

        global_payload = super_payload.get("_global")
        cat_payloads = super_payload.get("categories")
        if not isinstance(cat_payloads, Mapping):
            continue

        super_categories: Dict[str, Any] = {}
        for cat_name, cat_payload in cat_payloads.items():
            if not isinstance(cat_payload, Mapping):
                continue
            merged = _merge_mapping(global_payload if isinstance(global_payload, Mapping) else {}, cat_payload)
            merged.pop("categories", None)
            merged.pop("_global", None)
            key = f"{super_name}|{cat_name}"
            registry[key] = merged
            super_categories[cat_name] = merged
            catmap["mappings"].append(
                {
                    "super_label": super_name,
                    "cat_label": cat_name,
                    "key": key,
                }
            )
        if super_categories:
            categories[super_name] = {"categories": super_categories}

    manifest = {
        "schema": payload.get("_schema"),
        "metadata": metadata,
    }

    return registry, categories, catmap, version, generated_at, manifest


def _load_optional(base: pathlib.Path, name: str) -> Dict[str, Any]:
    candidate = base / name
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_v4_bundle(base: pathlib.Path, registry_path: pathlib.Path) -> KnowledgePack:
    with registry_path.open("r", encoding="utf-8") as handle:
        registry_payload = json.load(handle)

    registry, categories, catmap, version, generated_at, manifest = _flatten_v4_registry(registry_payload)

    extractors_path = base / "extractors.json"
    if extractors_path.exists():
        with extractors_path.open("r", encoding="utf-8") as handle:
            extractors_payload = json.load(handle)
    else:
        extractors_payload = {}

    validators = _load_optional(base, "validators.json")
    formulas = _load_optional(base, "formulas.json")
    templates = _load_optional(base, "templates.json")
    views = _load_optional(base, "views.json")
    profiles = _load_optional(base, "profiles.json")
    contexts = _load_optional(base, "contexts.json")

    manifest_sources = {}
    if registry_path.exists():
        manifest_sources["registry"] = str(registry_path)
    if extractors_path.exists():
        manifest_sources["extractors"] = str(extractors_path)
    if manifest_sources:
        manifest["sources"] = manifest_sources

    if isinstance(extractors_payload, Mapping) and "metadata" in extractors_payload:
        manifest.setdefault("extractors_metadata", extractors_payload.get("metadata"))

    return KnowledgePack(
        version=version,
        generated_at=generated_at,
        registry=registry,
        catmap=catmap,
        categories=categories,
        extractors=extractors_payload if isinstance(extractors_payload, dict) else {},
        validators=validators,
        formulas=formulas,
        templates=templates,
        views=views,
        profiles=profiles,
        contexts=contexts,
        schema_keynote=None,
        manifest=manifest,
    )


def _maybe_load_v4_from_directory(path: pathlib.Path) -> KnowledgePack | None:
    candidates: Iterable[pathlib.Path] = (
        path,
        path / "properties",
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        registry_path = candidate / "registry.json"
        if registry_path.exists():
            return _load_v4_bundle(candidate, registry_path)
    return None


def _maybe_load_v4_from_file(path: pathlib.Path) -> KnowledgePack | None:
    if path.name != "registry.json":
        return None
    base = path.parent
    return _load_v4_bundle(base, path)


def load_pack(pack_json_path: str) -> KnowledgePack:
    path = pathlib.Path(pack_json_path)
    if path.is_dir():
        modern = _maybe_load_v4_from_directory(path)
        if modern is not None:
            return modern
        candidate = path / "pack.json"
        if candidate.exists():
            path = candidate
    elif path.name == "registry.json":
        modern = _maybe_load_v4_from_file(path)
        if modern is not None:
            return modern
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "files" in payload:
        return _load_old_style(payload, path.parent)
    if isinstance(payload, dict) and payload.get("_schema") == "v4":
        return _load_v4_bundle(path.parent, path)
    return _load_inline(payload)
