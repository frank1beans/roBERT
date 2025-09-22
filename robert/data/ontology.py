"""Ontology helpers shared between training and inference pipelines."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np


__all__ = [
    "Ontology",
    "load_ontology",
    "load_label_maps",
    "build_name_maps",
    "build_mask",
]


@dataclass(frozen=True)
class Ontology:
    super_to_cat: Mapping[str, List[str]]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Iterable[str]]) -> "Ontology":
        cleaned = {str(k): [str(v) for v in values] for k, values in mapping.items()}
        return cls(super_to_cat=cleaned)

    def super_labels(self) -> List[str]:
        return list(self.super_to_cat.keys())

    def cat_labels(self) -> List[str]:
        cats: List[str] = []
        for values in self.super_to_cat.values():
            cats.extend(values)
        # preserva ordine ma rimuove duplicati
        seen = set()
        unique: List[str] = []
        for cat in cats:
            if cat not in seen:
                unique.append(cat)
                seen.add(cat)
        return unique


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_ontology(path: str | Path) -> Ontology:
    """Load an ontology file compatible with the historical training scripts."""

    raw = json.load(open(path, "r", encoding="utf-8"))
    if "super_to_cats" in raw:
        mapping = raw["super_to_cats"]
    elif isinstance(raw, dict):
        mapping = raw
    else:
        raise ValueError(f"Unsupported ontology format: {type(raw)!r}")
    if not isinstance(mapping, MutableMapping):
        raise ValueError("Ontology mapping must be a dictionary-like structure")
    return Ontology.from_mapping(mapping)


def load_label_maps(path: str | Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Load label <-> id maps from a JSON file.

    The function is tolerant with legacy schemas that were used across
    different experiments. It always returns normalized string keys and
    integer identifiers.
    """

    raw = json.load(open(path, "r", encoding="utf-8"))

    def _norm_name2id(data: Mapping) -> Dict[str, int]:
        return {str(k): int(v) for k, v in data.items()}

    def _invert(mapping: Mapping[str, int]) -> Dict[int, str]:
        return {int(v): str(k) for k, v in mapping.items()}

    super_name_to_id: Dict[str, int] | None = None
    cat_name_to_id: Dict[str, int] | None = None

    if "id2super" in raw and "id2cat" in raw:
        super_id_to_name = {int(k): str(v) for k, v in raw["id2super"].items()}
        cat_id_to_name = {int(k): str(v) for k, v in raw["id2cat"].items()}
        super_name_to_id = {v: k for k, v in super_id_to_name.items()}
        cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}
    elif "super2id" in raw and "cat2id" in raw:
        super_name_to_id = _norm_name2id(raw["super2id"])
        cat_name_to_id = _norm_name2id(raw["cat2id"])
    elif "supercats" in raw and "cats" in raw:
        super_name_to_id = _norm_name2id(raw["supercats"])
        cat_name_to_id = _norm_name2id(raw["cats"])
    else:
        raise ValueError(f"Unsupported label map schema, keys={list(raw.keys())}")

    super_id_to_name = _invert(super_name_to_id)
    cat_id_to_name = _invert(cat_name_to_id)
    return super_name_to_id, cat_name_to_id, super_id_to_name, cat_id_to_name


def build_name_maps(ontology: Ontology) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    """Build deterministic id maps from an :class:`Ontology`."""

    super_labels = ontology.super_labels()
    cat_labels = ontology.cat_labels()
    super_name_to_id = {name: idx for idx, name in enumerate(super_labels)}
    cat_name_to_id = {name: idx for idx, name in enumerate(cat_labels)}
    super_id_to_name = {idx: name for name, idx in super_name_to_id.items()}
    cat_id_to_name = {idx: name for name, idx in cat_name_to_id.items()}
    return super_name_to_id, cat_name_to_id, super_id_to_name, cat_id_to_name


def build_mask(
    ontology: Ontology,
    super_name_to_id: Mapping[str, int],
    cat_name_to_id: Mapping[str, int],
    *,
    return_report: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, object]]:
    """Construct a boolean mask describing the hierarchy relations.

    Parameters
    ----------
    ontology:
        Ontology instance describing the hierarchy.
    super_name_to_id / cat_name_to_id:
        Mapping from label names to ids.
    return_report:
        If ``True`` the function returns a tuple ``(mask, report)`` where
        ``report`` contains diagnostic information about missing labels.
    """

    if not super_name_to_id or not cat_name_to_id:
        raise ValueError("Name to id dictionaries must not be empty")

    num_super = max(super_name_to_id.values()) + 1
    num_cat = max(cat_name_to_id.values()) + 1
    mask = np.zeros((num_super, num_cat), dtype=np.float32)

    missing_super = 0
    missing_cat = 0
    for super_name, cat_list in ontology.super_to_cat.items():
        super_idx = super_name_to_id.get(super_name)
        if super_idx is None:
            super_idx = super_name_to_id.get(_normalize(super_name))
        if super_idx is None:
            missing_super += 1
            continue
        for cat in cat_list:
            cat_idx = cat_name_to_id.get(cat)
            if cat_idx is None:
                cat_idx = cat_name_to_id.get(_normalize(cat))
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

    if return_report:
        return mask, report
    return mask
