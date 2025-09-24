"""Data preparation helpers for the BIM NLP pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ontology_utils import Ontology, build_mask_from_ontology, load_label_maps, load_ontology, save_label_maps
from ..extraction import extract_properties

from ..core.pack_loader import load_pack


def _resolve_pack_json(path: Path) -> Path:
    """Return the JSON file to read when ``path`` points to a pack folder."""

    path = Path(path)
    if path.is_dir():
        candidate = path / "pack.json"
        if candidate.exists():
            return candidate
    return path


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    """Return a stable slug suitable for property identifiers."""

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower()
    slug = _SLUG_RE.sub("_", lowered).strip("_")
    if slug:
        return slug
    # Fall back to a deterministic placeholder derived from the original text
    digest = unicodedata.normalize("NFKD", str(value)).encode("utf-8")
    return "slot_" + re.sub(r"[^a-f0-9]", "", digest.hex())[:8]


def _flatten_registry_v4(payload: Mapping[str, Any]) -> Dict[str, Dict[str, object]]:
    """Flatten the hierarchical registry layout introduced in v4 packs."""

    result: Dict[str, Dict[str, object]] = {}

    for super_name, block in payload.items():
        if not isinstance(block, Mapping):
            continue
        if not isinstance(super_name, str):
            continue
        if super_name.startswith("_") or super_name == "metadata":
            continue

        categories = block.get("categories") if isinstance(block.get("categories"), Mapping) else None
        global_block = block.get("_global") if isinstance(block.get("_global"), Mapping) else None

        if not categories:
            continue

        super_slug = _slugify(super_name)
        global_slots = {}
        global_patterns = {}
        global_meta = {}
        if isinstance(global_block, Mapping):
            raw_slots = global_block.get("slots")
            if isinstance(raw_slots, Mapping):
                global_slots = {
                    f"{super_slug}.__global__.{_slugify(slot_name)}": dict(slot_schema)
                    for slot_name, slot_schema in raw_slots.items()
                    if isinstance(slot_name, str) and isinstance(slot_schema, Mapping)
                }
            raw_patterns = global_block.get("patterns")
            if isinstance(raw_patterns, Mapping):
                global_patterns = {
                    f"{super_slug}.__global__.{_slugify(slot_name)}": list(patterns)
                    for slot_name, patterns in raw_patterns.items()
                    if isinstance(slot_name, str)
                }
            global_meta = {
                key: value
                for key, value in global_block.items()
                if key not in {"slots", "patterns"}
            }

        for cat_name, cat_block in categories.items():
            if not isinstance(cat_name, str) or not isinstance(cat_block, Mapping):
                continue

            cat_slug = _slugify(cat_name)
            schema: Dict[str, Any] = {}

            cat_slots = {}
            raw_cat_slots = cat_block.get("slots")
            if isinstance(raw_cat_slots, Mapping):
                cat_slots = {
                    f"{super_slug}.{cat_slug}.{_slugify(slot_name)}": dict(slot_schema)
                    for slot_name, slot_schema in raw_cat_slots.items()
                    if isinstance(slot_name, str) and isinstance(slot_schema, Mapping)
                }

            cat_patterns = {}
            raw_cat_patterns = cat_block.get("patterns")
            if isinstance(raw_cat_patterns, Mapping):
                cat_patterns = {
                    f"{super_slug}.{cat_slug}.{_slugify(slot_name)}": list(patterns)
                    for slot_name, patterns in raw_cat_patterns.items()
                    if isinstance(slot_name, str)
                }

            # Merge inherited structures first, then the category-specific ones.
            slots: Dict[str, Any] = {}
            slots.update(global_slots)
            slots.update(cat_slots)
            schema["slots"] = slots

            if global_patterns or cat_patterns:
                patterns: Dict[str, Any] = {}
                patterns.update(global_patterns)
                patterns.update(cat_patterns)
                schema["patterns"] = patterns

            for key, value in global_meta.items():
                schema[key] = value

            for key, value in cat_block.items():
                if key in {"slots", "patterns"}:
                    continue
                schema[key] = value

            metadata = dict(schema.get("metadata", {}))
            metadata.setdefault("super_name", super_name)
            metadata.setdefault("category_name", cat_name)
            metadata.setdefault("super_id", super_slug)
            metadata.setdefault("category_id", cat_slug)
            schema["metadata"] = metadata

            result[f"{super_name}|{cat_name}"] = schema

    return result


def _normalize_registry_payload(payload: Any) -> Optional[Dict[str, Dict[str, object]]]:
    """Translate various registry layouts into a flat mapping."""

    def _build_from_entries(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            key = entry.get("key")
            if isinstance(key, str) and key:
                result[key] = dict(entry)
        return result

    if isinstance(payload, Mapping):
        if "registry" in payload:
            candidate = payload["registry"]
            normalized = _normalize_registry_payload(candidate)
            if normalized is not None:
                return normalized
        if "mappings" in payload and isinstance(payload["mappings"], Iterable):
            return _build_from_entries(payload["mappings"])
        if "files" in payload:
            return None
        if any(
            isinstance(value, Mapping) and ("categories" in value or "_global" in value)
            for key, value in payload.items()
            if isinstance(key, str) and not key.startswith("_") and key != "metadata"
        ):
            flattened = _flatten_registry_v4(payload)
            if flattened:
                return flattened
        if all(isinstance(key, str) for key in payload.keys()):
            return {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in payload.items()}
    elif isinstance(payload, list):
        return _build_from_entries(payload)
    return None


def _load_property_registry(path: Path) -> Optional[Dict[str, Dict[str, object]]]:
    """Load a property registry from either a raw JSON or a knowledge pack."""

    if path.is_dir():
        for name in (
            "registry.json",
            "properties_registry_extended.json",
            "properties_registry.json",
        ):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break

    path = _resolve_pack_json(path)
    try:
        payload = _load_json(path)
    except OSError:
        return None

    registry = _normalize_registry_payload(payload)
    if registry is not None:
        return registry

    if isinstance(payload, Mapping) and "files" in payload:
        files = payload.get("files", {})
        if isinstance(files, Mapping):
            registry_ref = files.get("registry")
            if isinstance(registry_ref, str):
                registry_path = (path.parent / registry_ref).resolve()
                if registry_path.exists():
                    nested_payload = _load_json(registry_path)
                    normalized = _normalize_registry_payload(nested_payload)
                    if normalized is not None:
                        return normalized

    try:
        pack = load_pack(str(path))
    except Exception:  # pragma: no cover - defensive fallback
        return None
    return pack.registry or None


def _normalize_extractors_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, Mapping):
        if "extractors" in payload and isinstance(payload["extractors"], Mapping):
            return _normalize_extractors_payload(payload["extractors"])
        patterns = payload.get("patterns") if "patterns" in payload else None
        if isinstance(patterns, list):
            pack: Dict[str, Any] = {"patterns": patterns}
            if "normalizers" in payload and isinstance(payload["normalizers"], Mapping):
                pack["normalizers"] = dict(payload["normalizers"])
            if "defaults" in payload and isinstance(payload["defaults"], Mapping):
                pack["defaults"] = dict(payload["defaults"])
            return pack
    elif isinstance(payload, list):
        return {"patterns": list(payload)}
    return None


def _load_extractors_pack(path: Path) -> Optional[Dict[str, Any]]:
    """Load an extractors pack from raw JSON or a knowledge pack."""

    if path.is_dir():
        for name in (
            "extractors_extended.json",
            "extractors.json",
        ):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break

    path = _resolve_pack_json(path)
    try:
        payload = _load_json(path)
    except OSError:
        return None

    pack = _normalize_extractors_payload(payload)
    if pack is not None:
        return pack

    if isinstance(payload, Mapping) and "files" in payload:
        files = payload.get("files", {})
        if isinstance(files, Mapping):
            extractors_ref = files.get("extractors")
            if isinstance(extractors_ref, str):
                extractors_path = (path.parent / extractors_ref).resolve()
                if extractors_path.exists():
                    nested_payload = _load_json(extractors_path)
                    normalized = _normalize_extractors_payload(nested_payload)
                    if normalized is not None:
                        return normalized

    try:
        pack_obj = load_pack(str(path))
    except Exception:  # pragma: no cover - defensive fallback
        return None
    return pack_obj.extractors or None


def _infer_slot_normalizers(slot: Mapping[str, Any]) -> List[str]:
    slot_type = str(slot.get("type", "")).strip().lower()
    if slot_type in {"float", "number", "numeric", "ratio"}:
        return ["to_number"]
    if slot_type in {"int", "integer"}:
        return ["to_number"]
    if slot_type in {"bool", "boolean"}:
        return ["to_bool_strict"]
    if slot_type in {"enum", "text"}:
        return ["strip"]
    return []


def _build_registry_extractors(
    registry: Mapping[str, Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    for key, schema in registry.items():
        if not isinstance(schema, Mapping):
            continue
        schema_patterns = schema.get("patterns")
        if not isinstance(schema_patterns, Mapping):
            continue
        slots = schema.get("slots") if isinstance(schema.get("slots"), Mapping) else {}
        tags: List[str] = []
        if isinstance(key, str) and "|" in key:
            super_name, cat_name = [part.strip() for part in key.split("|", 1)]
            if super_name:
                tags.append(f"category:{super_name}")
            if cat_name:
                tags.append(f"subcategory:{cat_name}")
        for prop_id, regexes in schema_patterns.items():
            if not isinstance(prop_id, str):
                continue
            if not isinstance(regexes, (list, tuple)):
                continue
            cleaned = [str(rx) for rx in regexes if isinstance(rx, str) and rx]
            if not cleaned:
                continue
            pattern_spec: Dict[str, Any] = {"property_id": prop_id, "regex": cleaned}
            if tags:
                pattern_spec["tags"] = list(tags)
            slot_info = slots.get(prop_id) if isinstance(slots, Mapping) else None
            if isinstance(slot_info, Mapping):
                normals = _infer_slot_normalizers(slot_info)
                if normals:
                    pattern_spec["normalizers"] = normals
            patterns.append(pattern_spec)
    if not patterns:
        return None
    return {"patterns": patterns}


def _merge_extractors_pack(
    primary: Optional[Mapping[str, Any]],
    secondary: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if primary is None:
        return dict(secondary) if isinstance(secondary, Mapping) else None
    if secondary is None:
        return dict(primary)

    merged: Dict[str, Any] = {}
    if isinstance(primary, Mapping):
        for key, value in primary.items():
            if key == "patterns":
                continue
            if key == "normalizers" and isinstance(value, Mapping):
                merged["normalizers"] = dict(value)
            else:
                merged[key] = value
    if isinstance(secondary, Mapping):
        for key, value in secondary.items():
            if key == "patterns":
                continue
            if key == "normalizers" and isinstance(value, Mapping):
                base_norms = merged.get("normalizers")
                if not isinstance(base_norms, dict):
                    base_norms = {}
                merged["normalizers"] = {**base_norms, **value}
            elif key not in merged:
                merged[key] = value

    patterns: List[Any] = []
    if isinstance(primary, Mapping):
        patterns.extend(list(primary.get("patterns", [])))
    if isinstance(secondary, Mapping):
        patterns.extend(list(secondary.get("patterns", [])))
    merged["patterns"] = patterns
    return merged

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
    properties_registry_path: Optional[str | Path] = None,
    extractors_pack_path: Optional[str | Path] = None,
    text_field: str = "text",
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, int], Mapping[str, int]]:
    super_name_to_id, cat_name_to_id, _, _ = create_or_load_label_maps(
        label_maps_path, ontology_path=ontology_path
    )

    train_df = load_jsonl_to_df(train_path)

    property_registry: Optional[Dict[str, Dict[str, object]]] = None
    if properties_registry_path is not None:
        property_registry = _load_property_registry(Path(properties_registry_path))

    registry_extractors: Optional[Dict[str, Any]] = None
    if property_registry:
        registry_extractors = _build_registry_extractors(property_registry)

    extractors_pack: Optional[Dict[str, Any]] = None
    use_registry_tags = False
    if extractors_pack_path is not None:
        extractors_pack = _load_extractors_pack(Path(extractors_pack_path))
        if extractors_pack is None:
            extractors_pack = registry_extractors
            use_registry_tags = registry_extractors is not None
        elif registry_extractors is not None:
            extractors_pack = _merge_extractors_pack(extractors_pack, registry_extractors)
    else:
        extractors_pack = registry_extractors
        use_registry_tags = registry_extractors is not None

    def _resolve_schema(super_name: str, cat_name: str) -> Dict[str, object]:
        if property_registry is None:
            return {}
        key = f"{super_name}|{cat_name}"
        if key in property_registry and isinstance(property_registry[key], dict):
            return property_registry[key]
        lowered = key.lower()
        for candidate, value in property_registry.items():
            if (
                isinstance(candidate, str)
                and "|" in candidate
                and candidate.lower() == lowered
                and isinstance(value, dict)
            ):
                return value
        return {}

    def _extract_props(
        text_value: str,
        allowed: Optional[Sequence[str]],
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        if not text_value or extractors_pack is None:
            return {}
        extracted = extract_properties(
            text_value,
            extractors_pack,
            allowed_properties=allowed,
            target_tags=tags,
        )
        if allowed is None:
            return extracted
        allowed_set = set(allowed)
        return {key: val for key, val in extracted.items() if key in allowed_set}

    if done_uids_path and Path(done_uids_path).exists():
        done_uids = {line.strip() for line in open(done_uids_path, "r", encoding="utf-8") if line.strip()}
        if "uid" in train_df.columns:
            train_df = train_df[~train_df["uid"].isin(done_uids)]

    def _map_row(row: pd.Series) -> Optional[Tuple[int, int, str, str]]:
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
        schema = _resolve_schema(super_name, cat_name)
        property_schemas.append(schema)
        text_value = str(row.get(text_field, "")) if text_field in row else str(row.get("text", ""))
        allowed = tuple((schema or {}).get("slots", {}).keys()) if schema else None
        target_tags: Optional[Tuple[str, ...]] = None
        if use_registry_tags:
            tags = []
            if super_name:
                tags.append(f"category:{super_name}")
            if cat_name:
                tags.append(f"subcategory:{cat_name}")
            target_tags = tuple(tags) if tags else None
        elif not schema:
            tags = []
            if super_name:
                tags.append(f"category:{super_name}")
            if cat_name:
                tags.append(f"subcategory:{cat_name}")
            target_tags = tuple(tags) if tags else None
        extracted_properties.append(_extract_props(text_value, allowed, target_tags))
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
            schema = _resolve_schema(super_name, cat_name)
            property_schemas_val.append(schema)
            text_value = str(row.get(text_field, "")) if text_field in row else str(row.get("text", ""))
            allowed = tuple((schema or {}).get("slots", {}).keys()) if schema else None
            target_tags: Optional[Tuple[str, ...]] = None
            if use_registry_tags:
                tags = []
                if super_name:
                    tags.append(f"category:{super_name}")
                if cat_name:
                    tags.append(f"subcategory:{cat_name}")
                target_tags = tuple(tags) if tags else None
            elif not schema:
                tags = []
                if super_name:
                    tags.append(f"category:{super_name}")
                if cat_name:
                    tags.append(f"subcategory:{cat_name}")
                target_tags = tuple(tags) if tags else None
            extracted_properties_val.append(_extract_props(text_value, allowed, target_tags))
            val_rows.append(row)
        val_processed = pd.DataFrame(val_rows).assign(
            super_label=mapped_super_val,
            cat_label=mapped_cat_val,
            property_schema=property_schemas_val,
            properties=extracted_properties_val,
        )
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
