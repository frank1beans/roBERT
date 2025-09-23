from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Set

from ..extraction import extract_properties


def _normalize_category_labels(category: Any) -> Sequence[str]:
    """Return a flat list of category labels from various inputs."""

    if category is None:
        return []
    if isinstance(category, str):
        return [category]
    if isinstance(category, dict):
        value = category.get("label")
        return [value] if value else []

    labels: list[str] = []
    try:
        iterator = iter(category)
    except TypeError:
        return labels

    for item in iterator:
        labels.extend(_normalize_category_labels(item))
    return labels


def _extractor_property_ids(pack) -> Set[str]:
    """Return the set of property ids defined in the extractors pack."""

    cache = getattr(pack, "_extractor_property_ids", None)
    if cache is None:
        cache = {
            item.get("property_id")
            for item in pack.extractors.get("patterns", [])
            if item.get("property_id")
        }
        setattr(pack, "_extractor_property_ids", cache)
    return cache


def _category_property_index(pack) -> Dict[str, Set[str]]:
    """Build an index mapping category identifiers to allowed property ids."""

    cache = getattr(pack, "_category_property_index", None)
    if cache is not None:
        return cache

    index: Dict[str, Set[str]] = {}
    groups = pack.registry.get("groups", {}) if getattr(pack, "registry", None) else {}

    def collect_from_groups(group_ids: Iterable[str]) -> Set[str]:
        props: Set[str] = set()
        for gid in group_ids or []:
            group = groups.get(gid)
            if group:
                props.update(group.get("properties", []))
        return props

    for mapping in pack.catmap.get("mappings", []):
        allowed: Set[str] = set()
        for key in ("props_required", "props_recommended"):
            allowed.update(mapping.get(key, []))
        allowed.update(collect_from_groups(mapping.get("groups_required", [])))
        allowed.update(collect_from_groups(mapping.get("groups_recommended", [])))
        for target in mapping.get("keynote_mapping", {}).values():
            if isinstance(target, str) and target:
                allowed.add(target)

        cat_label = str(mapping.get("cat_label", "")).lower()
        cat_id = str(mapping.get("cat_id", "")).lower()
        frozen = set(allowed)
        if cat_label:
            index[cat_label] = frozen
        if cat_id:
            index[cat_id] = frozen

    setattr(pack, "_category_property_index", index)
    return index


def predict_properties(text: str, pack, categories: Any) -> Dict[str, Any]:
    """
    Extract properties from ``text`` limited to those allowed for ``categories``.

    ``categories`` can be expressed as a string, dict, iterable or nested
    combination of the previous forms. Only the properties whose identifiers are
    both declared in the extractors pack and allowed by the category mappings
    are returned.
    """

    labels = {label.strip() for label in _normalize_category_labels(categories) if label}
    allowed_properties: Optional[Set[str]] = None
    if labels:
        index = _category_property_index(pack)
        extractor_ids = _extractor_property_ids(pack)
        collected: Set[str] = set()
        for label in labels:
            key = label.lower()
            collected.update(index.get(key, set()))
        collected.intersection_update(extractor_ids)
        if collected:
            allowed_properties = collected

    return extract_properties(text, pack.extractors, allowed_properties=allowed_properties)


__all__ = [
    "predict_properties",
    "_normalize_category_labels",
    "_extractor_property_ids",
    "_category_property_index",
]
