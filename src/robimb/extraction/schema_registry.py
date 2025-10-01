"""Schema registry utilities for property extraction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..config import get_settings

__all__ = [
    "PropertySpec",
    "CategorySchema",
    "SchemaRegistry",
    "load_registry",
    "load_category_schema",
]


@dataclass(frozen=True)
class PropertySpec:
    """Metadata describing a property expected in a category schema."""

    id: str
    title: str
    type: str
    unit: Optional[str]
    required: bool
    description: Optional[str] = None
    enum: Sequence[str] | None = None
    sources: Sequence[str] | None = None
    aliases: Sequence[str] | None = None


@dataclass(frozen=True)
class CategorySchema:
    """Logical schema describing a category."""

    id: str
    name: str
    schema_path: Path
    properties: Sequence[PropertySpec]
    required: Sequence[str]

    def property_ids(self) -> List[str]:
        return [spec.id for spec in self.properties]


class SchemaRegistry:
    """Load and query category schemas."""

    def __init__(self, registry_path: Path):
        self._path = Path(registry_path)
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        categories_payload = payload.get("categories", [])
        self.metadata: Dict[str, object] = payload.get("metadata", {})
        self.categories: Dict[str, CategorySchema] = {}
        for entry in categories_payload:
            schema_path = self._resolve_schema_path(entry["schema"])
            properties = [
                PropertySpec(
                    id=prop["id"],
                    title=prop.get("title", prop["id"]),
                    type=prop.get("type", "string"),
                    unit=prop.get("unit"),
                    required=prop.get("id") in set(entry.get("required", [])),
                    description=prop.get("description"),
                    enum=tuple(prop.get("enum", [])) or None,
                    sources=tuple(prop.get("sources", [])) or None,
                    aliases=tuple(prop.get("aliases", [])) or None,
                )
                for prop in entry.get("properties", [])
            ]
            category = CategorySchema(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                schema_path=schema_path,
                properties=tuple(properties),
                required=tuple(entry.get("required", [])),
            )
            self.categories[category.id] = category

    def _resolve_schema_path(self, schema_reference: str) -> Path:
        schema_path = Path(schema_reference)
        if not schema_path.is_absolute():
            schema_path = (self._path.parent / schema_path).resolve()
        return schema_path

    def list(self) -> Iterable[CategorySchema]:
        return self.categories.values()

    def get(self, category: str) -> Optional[CategorySchema]:
        if category in self.categories:
            return self.categories[category]
        lowered = category.lower()
        for schema in self.categories.values():
            if schema.name.lower() == lowered:
                return schema
        return None


@lru_cache(maxsize=8)
def load_registry(registry_path: Path | str | None = None) -> SchemaRegistry:
    """Load and cache the registry located at ``registry_path``."""

    settings = get_settings()
    path = Path(registry_path) if registry_path is not None else settings.registry_path
    return SchemaRegistry(Path(path))


def load_category_schema(
    category_id: str,
    *,
    registry_path: Path | str | None = None,
) -> tuple[CategorySchema, Dict[str, Any]]:
    """Return the :class:`CategorySchema` metadata and JSON schema body.

    Parameters
    ----------
    category_id:
        Identifier of the category to load.
    registry_path:
        Path to the registry JSON file. Defaults to the project-wide registry.

    Returns
    -------
    tuple[CategorySchema, Dict[str, Any]]
        The dataclass describing the category together with the parsed JSON
        schema document.

    Raises
    ------
    ValueError
        If the requested category is not present in the registry or the schema
        file cannot be read.
    """

    registry = load_registry(registry_path)
    category = registry.get(category_id)
    if category is None:
        raise ValueError(f"Categoria '{category_id}' non presente nel registry")
    schema_path = category.schema_path
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return category, payload
