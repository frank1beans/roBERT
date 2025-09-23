"""Utilities for working with ontology and label mappings."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

FALLBACK_LABEL = "#N/D"

__all__ = [
    "Ontology",
    "load_ontology",
    "build_mask_from_ontology",
    "load_label_maps",
    "save_label_maps",
]


@dataclass(frozen=True)
class Ontology:
    super_to_cat: Mapping[str, List[str]]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Iterable[str]]) -> "Ontology":
        cleaned: Dict[str, List[str]] = {
            str(k): [str(v) for v in values] for k, values in mapping.items()
        }
        return cls(super_to_cat=cleaned)

    def super_labels(self) -> List[str]:
        return list(self.super_to_cat.keys())

    def cat_labels(self) -> List[str]:
        cats: List[str] = []
        for values in self.super_to_cat.values():
            cats.extend(values)
        seen = set()
        unique: List[str] = []
        for cat in cats:
            if cat not in seen:
                seen.add(cat)
                unique.append(cat)
        return unique


def load_ontology(path: str | Path) -> Ontology:
    raw = json.load(open(path, "r", encoding="utf-8"))
    if "super_to_cats" in raw:
        mapping = raw["super_to_cats"]
    elif isinstance(raw, MutableMapping):
        mapping = raw
    else:
        raise ValueError(f"Unsupported ontology format: {type(raw)!r}")
    return Ontology.from_mapping(mapping)


def _invert(mapping: Mapping[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def _normalise_name(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _ensure_fallback_label(mapping: Mapping[str, int]) -> Dict[str, int]:
    """Return a copy of *mapping* that contains the fallback label at index 0."""

    ordered = sorted(((str(name), int(idx)) for name, idx in mapping.items()), key=lambda item: item[1])
    labels = [name for name, _ in ordered if name != FALLBACK_LABEL]
    labels.insert(0, FALLBACK_LABEL)
    return {label: idx for idx, label in enumerate(labels)}


def load_label_maps(
    path: str | Path,
    *,
    ontology: Optional[Ontology] = None,
    create_if_missing: bool = False,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    path = Path(path)
    if path.exists():
        raw = json.load(open(path, "r", encoding="utf-8"))
        if "super2id" in raw and "cat2id" in raw:
            super_name_to_id = {str(k): int(v) for k, v in raw["super2id"].items()}
            cat_name_to_id = {str(k): int(v) for k, v in raw["cat2id"].items()}
        elif "id2super" in raw and "id2cat" in raw:
            super_name_to_id = {str(v): int(k) for k, v in raw["id2super"].items()}
            cat_name_to_id = {str(v): int(k) for k, v in raw["id2cat"].items()}
        else:
            raise ValueError(f"Unsupported label map schema in {path}")
        normalised_super = _ensure_fallback_label(super_name_to_id)
        normalised_cat = _ensure_fallback_label(cat_name_to_id)
        if normalised_super != super_name_to_id or normalised_cat != cat_name_to_id:
            super_name_to_id = normalised_super
            cat_name_to_id = normalised_cat
            save_label_maps(
                path,
                super_name_to_id=super_name_to_id,
                cat_name_to_id=cat_name_to_id,
                super_id_to_name=_invert(super_name_to_id),
                cat_id_to_name=_invert(cat_name_to_id),
            )
        super_id_to_name = _invert(super_name_to_id)
        cat_id_to_name = _invert(cat_name_to_id)
        return super_name_to_id, cat_name_to_id, super_id_to_name, cat_id_to_name

    if not create_if_missing:
        raise FileNotFoundError(f"Label map file {path} does not exist")
    if ontology is None:
        raise ValueError("An ontology is required to build label maps from scratch")

    super_labels = [label for label in ontology.super_labels() if label != FALLBACK_LABEL]
    cat_labels = [label for label in ontology.cat_labels() if label != FALLBACK_LABEL]
    super_name_to_id = {FALLBACK_LABEL: 0}
    super_name_to_id.update({name: idx for idx, name in enumerate(super_labels, start=1)})
    cat_name_to_id = {FALLBACK_LABEL: 0}
    cat_name_to_id.update({name: idx for idx, name in enumerate(cat_labels, start=1)})
    super_id_to_name = _invert(super_name_to_id)
    cat_id_to_name = _invert(cat_name_to_id)

    save_label_maps(
        path,
        super_name_to_id=super_name_to_id,
        cat_name_to_id=cat_name_to_id,
        super_id_to_name=super_id_to_name,
        cat_id_to_name=cat_id_to_name,
    )
    return super_name_to_id, cat_name_to_id, super_id_to_name, cat_id_to_name


def save_label_maps(
    path: str | Path,
    *,
    super_name_to_id: Mapping[str, int],
    cat_name_to_id: Mapping[str, int],
    super_id_to_name: Mapping[int, str],
    cat_id_to_name: Mapping[int, str],
) -> None:
    payload = {
        "super2id": {str(k): int(v) for k, v in super_name_to_id.items()},
        "cat2id": {str(k): int(v) for k, v in cat_name_to_id.items()},
        "id2super": {int(k): str(v) for k, v in super_id_to_name.items()},
        "id2cat": {int(k): str(v) for k, v in cat_id_to_name.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_mask_from_ontology(
    ontology_path: str | Path,
    super_name_to_id: Mapping[str, int],
    cat_name_to_id: Mapping[str, int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    ontology = load_ontology(ontology_path)
    num_super = max(super_name_to_id.values()) + 1
    num_cat = max(cat_name_to_id.values()) + 1
    mask = np.zeros((num_super, num_cat), dtype=np.float32)
    missing_super = 0
    missing_cat = 0
    for super_name, cat_list in ontology.super_to_cat.items():
        super_idx = super_name_to_id.get(super_name)
        if super_idx is None:
            super_idx = super_name_to_id.get(_normalise_name(super_name))
        if super_idx is None:
            missing_super += 1
            continue
        for cat in cat_list:
            cat_idx = cat_name_to_id.get(cat)
            if cat_idx is None:
                cat_idx = cat_name_to_id.get(_normalise_name(cat))
            if cat_idx is None:
                missing_cat += 1
                continue
            mask[int(super_idx), int(cat_idx)] = 1.0
    report = {
        "note": "mask built from ontology",
        "missing_super": int(missing_super),
        "missing_cat": int(missing_cat),
        "coverage": float(mask.sum()),
    }
    return mask, report
