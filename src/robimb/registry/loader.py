"""Registry loader capable of resolving the official knowledge pack."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import pathlib
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from ..config import get_settings

from .normalizers import register_plugins
from .schemas import (
    CategoryDefinition,
    build_category_key,
    merge_inherited_structures,
)

__all__ = [
    "RegistryBundle",
    "load_pack",
    "RegistryLoader",
    "load_registry",
    "load_category",
    "json_schema_for",
]


@dataclass
class RegistryBundle:
    version: str
    generated_at: Optional[str]
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
    schema_keynote: Optional[Dict[str, Any]] = None
    manifest: Optional[Dict[str, Any]] = None
    category_models: MutableMapping[str, CategoryDefinition] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Legacy pack loading logic (migrated from robimb.core.pack_loader)
# ---------------------------------------------------------------------------


def _load_old_style(idx: Dict[str, Any], base: pathlib.Path) -> RegistryBundle:
    def load(key: str) -> Dict[str, Any]:
        path = base / idx["files"][key]
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_optional(key: str) -> Dict[str, Any]:
        filename = idx["files"].get(key)
        if not filename:
            return {}
        candidate = base / filename
        if not candidate.exists():
            return {}
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    bundle = RegistryBundle(
        version=idx.get("version", ""),
        generated_at=idx.get("generated_at"),
        registry=load("registry"),
        catmap=load("catmap"),
        categories=load("categories"),
        extractors=load_optional("extractors"),
        validators=load("validators"),
        formulas=load_optional("formulas"),
        templates=load_optional("templates"),
        views=load_optional("views"),
        profiles=load_optional("profiles"),
        contexts=load_optional("contexts"),
        schema_keynote=load_optional("schema_keynote") or None,
        manifest=load_optional("manifest") or None,
    )
    return bundle


def _as_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _load_inline(payload: Dict[str, Any]) -> RegistryBundle:
    extractors: Dict[str, Any]
    if isinstance(payload.get("extractors"), dict):
        extractors = payload["extractors"]
    elif "patterns" in payload:
        extractors = {
            "patterns": payload.get("patterns", []),
            "normalizers": payload.get("normalizers", {}),
        }
    else:
        extractors = {}

    bundle = RegistryBundle(
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
    return bundle


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


def _flatten_v4_registry(
    payload: Mapping[str, Any]
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    str,
    Optional[str],
    Dict[str, Any],
    Dict[str, CategoryDefinition],
]:
    metadata = payload.get("metadata")
    version = ""
    generated_at: Optional[str] = None
    if isinstance(metadata, Mapping):
        version = str(metadata.get("version", ""))
        generated_at = metadata.get("generated_at") or metadata.get("timestamp")

    registry: Dict[str, Any] = {}
    categories: Dict[str, Any] = {}
    catmap: Dict[str, Any] = {"mappings": []}
    category_models: Dict[str, CategoryDefinition] = {}

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
            key = build_category_key(super_name, cat_name)
            registry[key] = merged
            super_categories[cat_name] = merged
            catmap["mappings"].append(
                {"super_label": super_name, "cat_label": cat_name, "key": key}
            )
            category_models[key] = merge_inherited_structures(
                base=global_payload if isinstance(global_payload, Mapping) else {},
                override=cat_payload,
                super_label=super_name,
                category_label=cat_name,
            )
        if super_categories:
            categories[super_name] = {"categories": super_categories}

    manifest = {"schema": payload.get("_schema"), "metadata": metadata}
    return registry, categories, catmap, version, generated_at, manifest, category_models


def _load_optional(base: pathlib.Path, name: str) -> Dict[str, Any]:
    candidate = base / name
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _load_v4_bundle(base: pathlib.Path, registry_path: pathlib.Path) -> RegistryBundle:
    with registry_path.open("r", encoding="utf-8") as handle:
        registry_payload = json.load(handle)

    (
        registry,
        categories,
        catmap,
        version,
        generated_at,
        manifest,
        category_models,
    ) = _flatten_v4_registry(registry_payload)

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
        manifest.setdefault("sources", {}).update(manifest_sources)

    if isinstance(extractors_payload, Mapping) and "metadata" in extractors_payload:
        manifest.setdefault("extractors_metadata", extractors_payload.get("metadata"))

    bundle = RegistryBundle(
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
        category_models=category_models,
    )
    return bundle


def _maybe_load_v4_from_directory(path: pathlib.Path) -> Optional[RegistryBundle]:
    for candidate in (path, path / "properties"):
        if not candidate.exists():
            continue
        registry_path = candidate / "registry.json"
        if registry_path.exists():
            return _load_v4_bundle(candidate, registry_path)
    return None


def _maybe_load_v4_from_file(path: pathlib.Path) -> Optional[RegistryBundle]:
    if path.name != "registry.json":
        return None
    base = path.parent
    return _load_v4_bundle(base, path)


def load_pack(pack_json_path: str | pathlib.Path) -> RegistryBundle:
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
    bundle = _load_inline(payload)
    return bundle


# ---------------------------------------------------------------------------
# Public loader API
# ---------------------------------------------------------------------------


def _discover_default_pack() -> pathlib.Path:
    env_current = os.getenv("ROBIMB_PACK_CURRENT")
    if env_current:
        candidate = pathlib.Path(env_current)
        if candidate.exists():
            return candidate

    pack_dir = get_settings().pack_dir

    current = pack_dir / "current"
    if current.exists():
        return current

    if not pack_dir.exists():
        raise FileNotFoundError(
            "Directory 'pack' non trovata: verificare la configurazione ROBIMB_PACK_DIR."
        )
    candidates = sorted(pack_dir.glob("v*/registry.json"), reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "Nessun registry trovato sotto la directory del pack configurata; atteso un file 'v*/registry.json'."
    )


class RegistryLoader:
    """High level interface used across the codebase."""

    def __init__(self, source: str | pathlib.Path | None = None) -> None:
        self._source = pathlib.Path(source) if source is not None else _discover_default_pack()
        self._bundle: Optional[RegistryBundle] = None
        self._categories: Optional[Dict[str, CategoryDefinition]] = None

    def bundle(self) -> RegistryBundle:
        if self._bundle is None:
            self._bundle = load_pack(self._source)
        return self._bundle

    def load_registry(self) -> Dict[str, CategoryDefinition]:
        if self._categories is None:
            bundle = self.bundle()
            if bundle.category_models:
                categories = dict(bundle.category_models)
            else:
                categories = self._build_from_bundle(bundle)
            self._register_plugins(categories)
            self._categories = categories
        return self._categories

    def _build_from_bundle(self, bundle: RegistryBundle) -> Dict[str, CategoryDefinition]:
        categories: Dict[str, CategoryDefinition] = {}
        catmap = bundle.catmap or {}
        for mapping in catmap.get("mappings", []):
            if not isinstance(mapping, Mapping):
                continue
            key = str(mapping.get("key"))
            super_label = str(mapping.get("super_label"))
            cat_label = str(mapping.get("cat_label"))
            payload = bundle.registry.get(key, {})
            if not isinstance(payload, Mapping):
                payload = {}
            categories[key] = merge_inherited_structures(
                base={},
                override=payload,
                super_label=super_label,
                category_label=cat_label,
            )
        if not categories and isinstance(bundle.registry, Mapping):
            for key, payload in bundle.registry.items():
                if not isinstance(payload, Mapping):
                    continue
                if "|" in key:
                    super_label, cat_label = key.split("|", 1)
                else:
                    super_label, cat_label = key, key
                categories[key] = merge_inherited_structures(
                    base={},
                    override=payload,
                    super_label=super_label,
                    category_label=cat_label,
                )
        return categories

    def _register_plugins(self, categories: Mapping[str, CategoryDefinition]) -> None:
        for category in categories.values():
            for kind, specs in category.plugins.items():
                register_plugins(kind, specs)

    def load_category(self, key: str) -> Optional[CategoryDefinition]:
        return self.load_registry().get(key)

    def json_schema_for(self, key: str) -> Dict[str, Any]:
        category = self.load_category(key)
        if category is None:
            raise KeyError(f"Categoria non trovata: {key}")
        return category.json_schema()


# Convenience module-level wrappers -------------------------------------------------


def load_registry(source: str | pathlib.Path | None = None) -> Dict[str, CategoryDefinition]:
    return RegistryLoader(source).load_registry()


def load_category(key: str, *, source: str | pathlib.Path | None = None) -> Optional[CategoryDefinition]:
    return RegistryLoader(source).load_category(key)


def json_schema_for(key: str, *, source: str | pathlib.Path | None = None) -> Dict[str, Any]:
    return RegistryLoader(source).json_schema_for(key)
