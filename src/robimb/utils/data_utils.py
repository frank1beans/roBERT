"""Data preparation helpers for the BIM NLP pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .ontology_utils import Ontology, build_mask_from_ontology, load_label_maps, load_ontology, save_label_maps

__all__ = [
    "load_jsonl_to_df",
    "prepare_classification_dataset",
    "save_datasets",
    "prepare_mlm_corpus",
    "create_or_load_label_maps",
    "build_mask_and_report",
]


def load_jsonl_to_df(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def create_or_load_label_maps(
    label_maps_path: str | Path,
    *,
    ontology_path: Optional[str | Path] = None,
) -> Tuple[Mapping[str, int], Mapping[str, int], Mapping[int, str], Mapping[int, str]]:
    label_maps_path = Path(label_maps_path)
    if label_maps_path.exists():
        return load_label_maps(label_maps_path)
    if not ontology_path:
        raise ValueError("Cannot create label maps without an ontology file")
    ontology = load_ontology(ontology_path)
    return load_label_maps(label_maps_path, ontology=ontology, create_if_missing=True)


def build_mask_and_report(
    ontology_path: Optional[str | Path],
    super_name_to_id: Mapping[str, int],
    cat_name_to_id: Mapping[str, int],
) -> Tuple[np.ndarray, Mapping[str, object]]:
    if ontology_path is None:
        mask = np.ones((len(super_name_to_id), len(cat_name_to_id)), dtype=np.float32)
        report = {
            "note": "no ontology provided, using full mask",
            "missing_super": 0,
            "missing_cat": 0,
            "coverage": float(mask.sum()),
        }
        return mask, report
    return build_mask_from_ontology(ontology_path, super_name_to_id, cat_name_to_id)


def prepare_classification_dataset(
    train_path: str | Path,
    val_path: Optional[str | Path],
    *,
    label_maps_path: str | Path,
    ontology_path: Optional[str | Path] = None,
    done_uids_path: Optional[str | Path] = None,
    val_split: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, int], Mapping[str, int]]:
    super_name_to_id, cat_name_to_id, _, _ = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    train_df = load_jsonl_to_df(train_path)

    if done_uids_path and Path(done_uids_path).exists():
        done_uids = {line.strip() for line in open(done_uids_path, "r", encoding="utf-8") if line.strip()}
        if "uid" in train_df.columns:
            train_df = train_df[~train_df["uid"].isin(done_uids)]

    def _map_row(row: pd.Series) -> Optional[Tuple[int, int]]:
        super_name = row.get("super_id") or row.get("super") or row.get("super_name")
        cat_name = row.get("cat_id") or row.get("cat") or row.get("cat_name")
        if isinstance(cat_name, str) and "::" in cat_name:
            _, cat_name = [part.strip() for part in cat_name.split("::", 1)]
        if super_name is None or cat_name is None:
            return None
        super_idx = super_name_to_id.get(str(super_name))
        if super_idx is None:
            super_idx = super_name_to_id.get(str(super_name).lower())
        cat_idx = cat_name_to_id.get(str(cat_name))
        if cat_idx is None:
            cat_idx = cat_name_to_id.get(str(cat_name).lower())
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
    processed = processed.assign(super_label=mapped_super, cat_label=mapped_cat)

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
        val_processed = pd.DataFrame(val_rows).assign(super_label=mapped_super_val, cat_label=mapped_cat_val)
    else:
        processed = processed.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(processed) * (1.0 - val_split))
        val_processed = processed.iloc[split_idx:].reset_index(drop=True)
        processed = processed.iloc[:split_idx].reset_index(drop=True)

    return processed, val_processed, super_name_to_id, cat_name_to_id


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
