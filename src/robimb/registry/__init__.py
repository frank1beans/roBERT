"""Registry management utilities."""
from __future__ import annotations

from .loader import RegistryBundle, RegistryLoader, json_schema_for, load_category, load_pack, load_registry
from .normalizers import PluginRegistry, get_registered_plugins, pack_folders_to_monolith, register_plugins
from .schemas import CategoryDefinition, PropertySlot
from .validators import Issue, validate

__all__ = [
    "RegistryBundle",
    "RegistryLoader",
    "load_pack",
    "load_registry",
    "load_category",
    "json_schema_for",
    "CategoryDefinition",
    "PropertySlot",
    "Issue",
    "validate",
    "PluginRegistry",
    "register_plugins",
    "get_registered_plugins",
    "pack_folders_to_monolith",
]
