
from __future__ import annotations
from typing import Dict, Any, Optional
import json, os
from ..inference.predict_category import load_classifier, predict_topk, _load_id2label
from ..inference.calibration import TemperatureCalibrator
from ..registry import validate
from ..templates.render import render
from . import predict_properties as _properties_module

def find_cat_entry(pack, cat_label: str):
    # look into catmap mappings
    for m in pack.catmap.get("mappings", []):
        if m.get("cat_label","").lower() == (cat_label or "").lower():
            return m
    return None


def predict_properties(text: str, pack, categories: Any) -> Dict[str, Any]:
    return _properties_module.predict_properties(text, pack, categories)

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
