"""Utilities dealing with registry-driven normalisers and plugins."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

__all__ = [
    "PluginRegistry",
    "register_plugins",
    "get_registered_plugins",
    "pack_folders_to_monolith",
]


class PluginRegistry:
    """Keep track of callables declared inside registry payloads."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, kind: str, name: str, obj: Any) -> None:
        self._registry.setdefault(kind, {})[name] = obj

    def get(self, kind: str) -> Mapping[str, Any]:
        return self._registry.get(kind, {})

    def as_mapping(self) -> Mapping[str, Mapping[str, Any]]:
        return {kind: dict(values) for kind, values in self._registry.items()}


_GLOBAL_REGISTRY = PluginRegistry()


def _load_plugin(spec: str) -> Any:
    if "=" in spec:
        alias, path = spec.split("=", 1)
        alias = alias.strip()
    else:
        alias = None
        path = spec
    if ":" not in path:
        raise ValueError(f"Plugin specification '{spec}' must use the form 'module:attribute'.")
    module_name, attr = path.split(":", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    name = alias or attr.split(".")[-1]
    return name, obj


def register_plugins(kind: str, specs: Iterable[str]) -> None:
    for spec in specs:
        try:
            name, obj = _load_plugin(spec)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Impossibile importare il plugin '{spec}': {exc}") from exc
        _GLOBAL_REGISTRY.register(kind, name, obj)


def get_registered_plugins(kind: str) -> Mapping[str, Any]:
    return _GLOBAL_REGISTRY.get(kind)


# ---------------------------------------------------------------------------
# Legacy helpers migrated from props.pack
# ---------------------------------------------------------------------------

import json


def _read_json_if_exists(path: Path) -> Optional[Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return None


def pack_folders_to_monolith(properties_root: Path, out_registry: Path, out_extractors: Path) -> None:
    """Re-create the legacy monolith (registry + extractors) from the folder layout."""

    registry: Dict[str, Any] = {}
    extractors_all: List[Dict[str, Any]] = []

    for super_dir in sorted([p for p in properties_root.iterdir() if p.is_dir()]):
        super_name = super_dir.name
        super_registry: Dict[str, Any] = {"_global": {"slots": {}}, "categories": {}}

        global_registry = _read_json_if_exists(super_dir / "_global" / "registry.json")
        if isinstance(global_registry, Mapping):
            super_registry["_global"]["slots"] = global_registry.get("slots", {})

        for cat_dir in sorted([p for p in super_dir.iterdir() if p.is_dir() and p.name != "_global"]):
            cat_name = cat_dir.name
            registry_json = _read_json_if_exists(cat_dir / "registry.json")
            if isinstance(registry_json, Mapping):
                super_registry["categories"][cat_name] = registry_json

            extractors_json = _read_json_if_exists(cat_dir / "extractors.json")
            if isinstance(extractors_json, list):
                extractors_all.extend(extractors_json)

        global_extractors = _read_json_if_exists(super_dir / "_global" / "extractors.json")
        if isinstance(global_extractors, list):
            extractors_all.extend(global_extractors)

        registry[super_name] = super_registry

    out_registry.parent.mkdir(parents=True, exist_ok=True)
    out_extractors.parent.mkdir(parents=True, exist_ok=True)
    out_registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    out_extractors.write_text(json.dumps(extractors_all, ensure_ascii=False, indent=2), encoding="utf-8")
