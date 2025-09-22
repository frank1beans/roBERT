
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _load_id2label(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    id2label = {int(k): v for k, v in js["id2label"].items()}
    return id2label

def load_classifier(model_path_or_name: str):
    tok = AutoTokenizer.from_pretrained(model_path_or_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path_or_name)
    mdl.eval()
    return tok, mdl

def predict_topk(text: str, model, tokenizer, id2label: Dict[int, str], topk: int = 5, calibrator=None):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logits = model(**inputs).logits  # [1, C]
        if calibrator is not None:
            from .calibration import TemperatureCalibrator
            logits = calibrator.apply(logits)
        probs = F.softmax(logits, dim=-1)[0]  # [C]
        values, indices = torch.topk(probs, k=min(topk, probs.shape[-1]))
        results = []
        for p, idx in zip(values.tolist(), indices.tolist()):
            results.append({"id": int(idx), "label": id2label.get(int(idx), str(idx)), "score": float(p)})
        top = results[0]
        return top, results, probs.tolist(), logits[0].tolist()
