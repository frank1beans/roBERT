"""Metric helpers shared across trainers and the CLI."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

__all__ = ["make_compute_metrics"]


def make_compute_metrics(num_super: int, num_cat: int):
    def _compute(eval_pred) -> Dict[str, float]:
        preds = getattr(eval_pred, "predictions", eval_pred)
        labels = getattr(eval_pred, "label_ids", None)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        logits = np.asarray(preds)
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if isinstance(labels, dict):
            y_super = np.asarray(labels["super_labels"])
            y_cat = np.asarray(labels["cat_labels"])
        elif isinstance(labels, (tuple, list)):
            y_super = np.asarray(labels[0])
            y_cat = np.asarray(labels[1])
        else:
            arr = np.asarray(labels)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"Label IDs formato inatteso: {arr.shape}")
            y_super, y_cat = arr[:, 0], arr[:, 1]

        S = num_super
        C = num_cat
        if logits.shape[1] != S + 2 * C:
            raise ValueError(f"atteso (N, {S + 2 * C}), trovato {logits.shape}")

        logits_super = logits[:, :S]
        logits_cat_pred = logits[:, S : S + C]
        logits_cat_gold = logits[:, S + C : S + 2 * C]

        pred_super = logits_super.argmax(-1)
        pred_cat_pred_super = logits_cat_pred.argmax(-1)
        pred_cat_gold_super = logits_cat_gold.argmax(-1)

        mask = y_cat != -100
        metrics = {
            "acc_super": accuracy_score(y_super, pred_super),
            "macro_f1_super": f1_score(y_super, pred_super, average="macro"),
            "acc_cat_pred_super": accuracy_score(y_cat[mask], pred_cat_pred_super[mask]) if mask.any() else float("nan"),
            "macro_f1_cat_pred_super": f1_score(y_cat[mask], pred_cat_pred_super[mask], average="macro")
            if mask.any()
            else float("nan"),
            "acc_cat_gold_super": accuracy_score(y_cat[mask], pred_cat_gold_super[mask]) if mask.any() else float("nan"),
            "macro_f1_cat_gold_super": f1_score(y_cat[mask], pred_cat_gold_super[mask], average="macro")
            if mask.any()
            else float("nan"),
        }
        return metrics

    return _compute
