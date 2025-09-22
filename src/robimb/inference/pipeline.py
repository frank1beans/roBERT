
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional, Sequence, Set
import json, os
from ..inference.predict_category import load_classifier, predict_topk, _load_id2label
from ..inference.calibration import TemperatureCalibrator
from ..features.extractors import extract_properties
from ..validators.engine import validate
from ..templates.render import render

def find_cat_entry(pack, cat_label: str):
    # look into catmap mappings
    for m in pack.catmap.get("mappings", []):
        if m.get("cat_label","").lower() == (cat_label or "").lower():
            return m
    return None


def _extractor_property_ids(pack) -> Set[str]:
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


def _normalize_category_labels(category: Any) -> Sequence[str]:
    if category is None:
        return []
    if isinstance(category, str):
        return [category]
    if isinstance(category, dict):
        val = category.get("label")
        return [val] if val else []
    labels: list[str] = []
    try:
        iterator = iter(category)
    except TypeError:
        return labels
    for item in iterator:
        labels.extend(_normalize_category_labels(item))
    return labels


def predict_properties(text: str, pack, categories: Any) -> Dict[str, Any]:
    labels = {lbl.strip() for lbl in _normalize_category_labels(categories) if lbl}
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

def run_pipeline(text: str, pack, model_name_or_path: str, label_index_path: str, topk: int = 5, calibrator_path: Optional[str]=None) -> Dict[str, Any]:
    id2label = _load_id2label(label_index_path)
    tokenizer, model = load_classifier(model_name_or_path)

    calibrator = None
    if calibrator_path and os.path.exists(calibrator_path):
        with open(calibrator_path,"r",encoding="utf-8") as f: sd=json.load(f)
        calibrator = TemperatureCalibrator.from_state_dict(sd)

    # 1) Category
    top, topk_list, probs, logits = predict_topk(text, model, tokenizer, id2label, topk=topk, calibrator=calibrator)

    # 2) Properties (regex extractors from pack)
    props = predict_properties(text, pack, top["label"])

    # 3) Validation (rules from pack), pass cat entry to rules if needed
    cat_entry = find_cat_entry(pack, top["label"])
    issues = validate(top["label"], props, context={}, rules_pack=pack.validators, cat_entry=cat_entry)

    # 4) Description render (templates from pack)
    descr = render(top["label"], props, pack.templates)

    return {
        "input_text": text,
        "category": top,
        "topk": topk_list,
        "properties": props,
        "issues": issues,
        "description": descr
    }
