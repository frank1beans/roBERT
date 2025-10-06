"""Dataset preparation helpers for the BIM NLP pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..extraction.legacy import extract_properties
from ..registry.schemas import CategoryDefinition, build_category_key
from .ontology_utils import (
    build_mask_from_ontology,
    load_label_maps,
    load_ontology,
)
from .registry_io import (
    ExtractorsPack,
    build_registry_extractors,
    load_extractors_pack,
    load_property_registry,
    merge_extractors_pack,
)
from .sampling import load_jsonl_to_df

__all__ = [
    "LabelMaps",
    "create_or_load_label_maps",
    "build_mask_and_report",
    "prepare_classification_dataset",
    "prepare_dataset_simple",
    "save_datasets",
    "prepare_mlm_corpus",
]


@dataclass(frozen=True)
class LabelMaps:
    """Typed wrapper around the four label map dictionaries used by training."""

    super_name_to_id: Mapping[str, int]
    cat_name_to_id: Mapping[str, int]
    super_id_to_name: Mapping[int, str]
    cat_id_to_name: Mapping[int, str]

    def as_tuple(self) -> Tuple[
        Mapping[str, int],
        Mapping[str, int],
        Mapping[int, str],
        Mapping[int, str],
    ]:
        return (
            self.super_name_to_id,
            self.cat_name_to_id,
            self.super_id_to_name,
            self.cat_id_to_name,
        )


def create_or_load_label_maps(
    label_maps_path: str | Path,
    *,
    ontology_path: Optional[str | Path] = None,
) -> LabelMaps:
    label_maps_path = Path(label_maps_path)
    if label_maps_path.exists():
        maps = load_label_maps(label_maps_path)
    else:
        if not ontology_path:
            raise ValueError("Cannot create label maps without an ontology file")
        ontology = load_ontology(ontology_path)
        maps = load_label_maps(
            label_maps_path, ontology=ontology, create_if_missing=True
        )
    return LabelMaps(*maps)


def build_mask_and_report(
    ontology_path: Optional[str | Path],
    label_maps: LabelMaps,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    if ontology_path is None:
        mask = np.ones(
            (
                len(label_maps.super_name_to_id),
                len(label_maps.cat_name_to_id),
            ),
            dtype=np.float32,
        )
        report = {
            "note": "no ontology provided, using full mask",
            "missing_super": 0,
            "missing_cat": 0,
            "coverage": float(mask.sum()),
        }
        return mask, report
    return build_mask_from_ontology(
        ontology_path, label_maps.super_name_to_id, label_maps.cat_name_to_id
    )


def _infer_category(
    property_registry: Optional[Mapping[str, CategoryDefinition]],
    super_name: str,
    cat_name: str,
) -> Optional[CategoryDefinition]:
    if property_registry is None:
        return None
    key = build_category_key(super_name, cat_name)
    category = property_registry.get(key)
    if category is not None:
        return category
    lowered = key.lower()
    for candidate_key, candidate in property_registry.items():
        if isinstance(candidate_key, str) and candidate_key.lower() == lowered:
            return candidate
    return None


def _build_target_tags(super_name: str, cat_name: str) -> Tuple[str, ...]:
    tags = []
    if super_name:
        tags.append(f"category:{super_name}")
    if cat_name:
        tags.append(f"subcategory:{cat_name}")
    return tuple(tags)


def prepare_classification_dataset(
    train_path: str | Path,
    val_path: Optional[str | Path],
    *,
    label_maps_path: str | Path,
    ontology_path: Optional[str | Path] = None,
    done_uids_path: Optional[str | Path] = None,
    val_split: float = 0.2,
    random_state: int = 42,
    properties_registry_path: Optional[str | Path] = None,
    extractors_pack_path: Optional[str | Path] = None,
    text_field: str = "text",
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelMaps]:
    label_maps = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    train_df = load_jsonl_to_df(train_path)

    property_registry: Optional[Dict[str, CategoryDefinition]] = None
    if properties_registry_path is not None:
        property_registry = load_property_registry(Path(properties_registry_path))

    registry_extractors: Optional[ExtractorsPack] = None
    if property_registry:
        registry_extractors = build_registry_extractors(property_registry)

    extractors_pack: Optional[ExtractorsPack] = None
    use_registry_tags = False
    if extractors_pack_path is not None:
        extractors_pack = load_extractors_pack(Path(extractors_pack_path))
        if extractors_pack is None:
            extractors_pack = registry_extractors
            use_registry_tags = registry_extractors is not None
        elif registry_extractors is not None:
            extractors_pack = merge_extractors_pack(extractors_pack, registry_extractors)
    else:
        extractors_pack = registry_extractors
        use_registry_tags = registry_extractors is not None

    pack_payload = extractors_pack.to_mapping() if extractors_pack else None

    if done_uids_path and Path(done_uids_path).exists():
        done_uids = {
            line.strip()
            for line in open(done_uids_path, "r", encoding="utf-8")
            if line.strip()
        }
        if "uid" in train_df.columns:
            train_df = train_df[~train_df["uid"].isin(done_uids)]

    def _map_row(row: pd.Series) -> Optional[Tuple[int, int, str, str]]:
        super_name = row.get("super_id") or row.get("super") or row.get("super_name")
        cat_name = row.get("cat_id") or row.get("cat") or row.get("cat_name")
        if isinstance(cat_name, str) and "::" in cat_name:
            _, cat_name = [part.strip() for part in cat_name.split("::", 1)]
        if super_name is None or cat_name is None:
            return None
        super_idx = label_maps.super_name_to_id.get(str(super_name))
        if super_idx is None:
            super_idx = label_maps.super_name_to_id.get(str(super_name).lower())
        cat_idx = label_maps.cat_name_to_id.get(str(cat_name))
        if cat_idx is None:
            cat_idx = label_maps.cat_name_to_id.get(str(cat_name).lower())
        if super_idx is None or cat_idx is None:
            return None
        return int(super_idx), int(cat_idx), str(super_name), str(cat_name)

    mapped_super: List[int] = []
    mapped_cat: List[int] = []
    property_schemas: List[Dict[str, object]] = []
    extracted_properties: List[Dict[str, object]] = []
    kept_rows = []
    for _, row in train_df.iterrows():
        mapping = _map_row(row)
        if mapping is None:
            continue
        s_idx, c_idx, super_name, cat_name = mapping
        mapped_super.append(s_idx)
        mapped_cat.append(c_idx)
        category_schema = _infer_category(property_registry, super_name, cat_name)
        schema = (
            category_schema.json_schema() if category_schema is not None else {}
        )
        property_schemas.append(schema)
        text_value = (
            str(row.get(text_field, "")) if text_field in row else str(row.get("text", ""))
        )
        allowed = (
            tuple(category_schema.property_ids())
            if category_schema is not None
            else None
        )
        target_tags: Optional[Tuple[str, ...]] = None
        if use_registry_tags:
            tags = _build_target_tags(super_name, cat_name)
            target_tags = tags or None
        elif not schema:
            tags = _build_target_tags(super_name, cat_name)
            target_tags = tags or None
        extracted = (
            extract_properties(
                text_value,
                pack_payload,
                allowed_properties=allowed,
                target_tags=target_tags,
            )
            if pack_payload
            else {}
        )
        if allowed is None:
            properties_payload = extracted
        else:
            allowed_set = set(allowed)
            properties_payload = {
                key: val for key, val in extracted.items() if key in allowed_set
            }
        extracted_properties.append(properties_payload)
        kept_rows.append(row)
    processed = pd.DataFrame(kept_rows)
    processed = processed.assign(
        super_label=mapped_super,
        cat_label=mapped_cat,
        property_schema=property_schemas,
        properties=extracted_properties,
    )

    if val_path:
        val_df = load_jsonl_to_df(val_path)
        val_rows = []
        mapped_super_val: List[int] = []
        mapped_cat_val: List[int] = []
        property_schemas_val: List[Dict[str, object]] = []
        extracted_properties_val: List[Dict[str, object]] = []
        for _, row in val_df.iterrows():
            mapping = _map_row(row)
            if mapping is None:
                continue
            s_idx, c_idx, super_name, cat_name = mapping
            mapped_super_val.append(s_idx)
            mapped_cat_val.append(c_idx)
            category_schema = _infer_category(property_registry, super_name, cat_name)
            schema = (
                category_schema.json_schema() if category_schema is not None else {}
            )
            property_schemas_val.append(schema)
            text_value = (
                str(row.get(text_field, "")) if text_field in row else str(row.get("text", ""))
            )
            allowed = (
                tuple(category_schema.property_ids())
                if category_schema is not None
                else None
            )
            target_tags: Optional[Tuple[str, ...]] = None
            if use_registry_tags:
                tags = _build_target_tags(super_name, cat_name)
                target_tags = tags or None
            elif not schema:
                tags = _build_target_tags(super_name, cat_name)
                target_tags = tags or None
            extracted = (
                extract_properties(
                    text_value,
                    pack_payload,
                    allowed_properties=allowed,
                    target_tags=target_tags,
                )
                if pack_payload
                else {}
            )
            if allowed is None:
                properties_payload = extracted
            else:
                allowed_set = set(allowed)
                properties_payload = {
                    key: val for key, val in extracted.items() if key in allowed_set
                }
            extracted_properties_val.append(properties_payload)
            val_rows.append(row)
        val_processed = pd.DataFrame(val_rows).assign(
            super_label=mapped_super_val,
            cat_label=mapped_cat_val,
            property_schema=property_schemas_val,
            properties=extracted_properties_val,
        )
    else:
        processed = processed.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )
        split_idx = int(len(processed) * (1.0 - val_split))
        val_processed = processed.iloc[split_idx:].reset_index(drop=True)
        processed = processed.iloc[:split_idx].reset_index(drop=True)

    return processed, val_processed, label_maps


def prepare_dataset_simple(
    train_path: str | Path,
    val_path: Optional[str | Path],
    *,
    label_maps_path: str | Path,
    ontology_path: Optional[str | Path] = None,
    done_uids_path: Optional[str | Path] = None,
    val_split: float = 0.2,
    random_state: int = 42,
    text_field: str = "text",
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelMaps]:
    """Prepare dataset without property extraction - just normalize labels and split.

    This is a simplified version that:
    - Loads data from JSONL/CSV/Excel/TXT
    - Maps category labels to IDs
    - Splits into train/val
    - Does NOT extract properties
    """
    label_maps = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    train_df = load_jsonl_to_df(train_path)

    if done_uids_path and Path(done_uids_path).exists():
        done_uids = {
            line.strip()
            for line in open(done_uids_path, "r", encoding="utf-8")
            if line.strip()
        }
        if "uid" in train_df.columns:
            train_df = train_df[~train_df["uid"].isin(done_uids)]

    def _map_row(row: pd.Series) -> Optional[Tuple[int, int]]:
        super_name = row.get("super_id") or row.get("super") or row.get("super_name")
        cat_name = row.get("cat_id") or row.get("cat") or row.get("cat_name")
        if isinstance(cat_name, str) and "::" in cat_name:
            _, cat_name = [part.strip() for part in cat_name.split("::", 1)]
        if super_name is None or cat_name is None:
            return None
        super_idx = label_maps.super_name_to_id.get(str(super_name))
        if super_idx is None:
            super_idx = label_maps.super_name_to_id.get(str(super_name).lower())
        cat_idx = label_maps.cat_name_to_id.get(str(cat_name))
        if cat_idx is None:
            cat_idx = label_maps.cat_name_to_id.get(str(cat_name).lower())
        if super_idx is None or cat_idx is None:
            return None
        return int(super_idx), int(cat_idx)

    mapped_super: List[int] = []
    mapped_cat: List[int] = []
    kept_rows = []

    for _, row in train_df.iterrows():
        mapping = _map_row(row)
        if mapping is None:
            continue
        s_idx, c_idx = mapping
        mapped_super.append(s_idx)
        mapped_cat.append(c_idx)
        kept_rows.append(row)

    processed = pd.DataFrame(kept_rows)
    processed = processed.assign(
        super_label=mapped_super,
        cat_label=mapped_cat,
    )

    # Handle validation split
    if val_path:
        val_df = load_jsonl_to_df(val_path)
        val_rows = []
        mapped_super_val: List[int] = []
        mapped_cat_val: List[int] = []

        for _, row in val_df.iterrows():
            mapping = _map_row(row)
            if mapping is None:
                continue
            s_idx, c_idx = mapping
            mapped_super_val.append(s_idx)
            mapped_cat_val.append(c_idx)
            val_rows.append(row)

        val_processed = pd.DataFrame(val_rows).assign(
            super_label=mapped_super_val,
            cat_label=mapped_cat_val,
        )
    else:
        # Random split
        processed = processed.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )
        split_idx = int(len(processed) * (1.0 - val_split))
        val_processed = processed.iloc[split_idx:].reset_index(drop=True)
        processed = processed.iloc[:split_idx].reset_index(drop=True)

    return processed, val_processed, label_maps


def save_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_processed.jsonl"
    val_path = out_dir / "val_processed.jsonl"
    for path, df in ((train_path, train_df), (val_path, val_df)):
        with open(path, "w", encoding="utf-8") as handle:
            for _, row in df.iterrows():
                record = row.to_dict()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_mlm_corpus(
    jsonl_files: Iterable[str | Path],
    out_txt_path: str | Path,
    *,
    text_field: str = "text",
    min_len: int = 5,
    dedup: bool = True,
) -> int:
    seen = set()
    texts: List[str] = []
    for path in jsonl_files:
        df = load_jsonl_to_df(path)
        for value in df.get(text_field, []):
            text = str(value).strip()
            if len(text) < min_len:
                continue
            if dedup:
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
            texts.append(text)
    out_txt_path = Path(out_txt_path)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(text + "\n")
    return len(texts)
