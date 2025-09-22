
from __future__ import annotations
from dataclasses import dataclass
import random, numpy as np, torch
from typing import Dict, Any
from sklearn.metrics import f1_score, accuracy_score

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@dataclass
class ClsMetrics:
    accuracy: float
    f1_macro: float
    f1_micro: float

def classification_metrics(preds, labels) -> Dict[str, float]:
    y_true = labels
    y_pred = preds
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1mi = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return {"accuracy": acc, "f1_macro": f1m, "f1_micro": f1mi}

def build_llrd_optimizer(model, base_lr: float=3e-5, lr_decay: float=0.95, weight_decay: float=0.01):
    # Create parameter groups with Layer-wise LR Decay (works for BERT/RoBERTa/XLM-R)
    # newer layers (higher index) get higher LR
    no_decay = ["bias", "LayerNorm.weight"]
    named_params = list(model.named_parameters())
    # group layers by encoder.layer.X or roberta.encoder.layer.X
    import re
    layer_map = {}
    for n,p in named_params:
        m = re.search(r"encoder\.layer\.(\d+)", n)
        layer = int(m.group(1)) if m else -1  # embeddings / pooler etc.
        layer_map.setdefault(layer, []).append((n,p))
    max_layer = max([k for k in layer_map.keys() if k >= 0], default=-1)
    param_groups = []
    for layer, items in layer_map.items():
        if layer == -1:
            lr = base_lr * (lr_decay ** (max_layer+1))
        else:
            lr = base_lr * (lr_decay ** (max_layer - layer))
        decay_params = [p for n,p in items if not any(nd in n for nd in no_decay)]
        nodecay_params = [p for n,p in items if any(nd in n for nd in no_decay)]
        if decay_params:
            param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
        if nodecay_params:
            param_groups.append({"params": nodecay_params, "lr": lr, "weight_decay": 0.0})
    # add heads if present (often under classifier or heads.*)
    head_params = [(n,p) for n,p in named_params if any(k in n for k in ["classifier","heads.","score.","lm_head"])]
    if head_params:
        decay = [p for n,p in head_params if not any(nd in n for nd in no_decay)]
        nodec = [p for n,p in head_params if any(nd in n for nd in no_decay)]
        if decay: param_groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay})
        if nodec: param_groups.append({"params": nodec, "lr": base_lr, "weight_decay": 0.0})
    from torch.optim import AdamW
    return AdamW(param_groups)
