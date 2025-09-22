"""Metric helpers shared across trainers and the CLI."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

__all__ = ["make_compute_metrics"]


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def make_compute_metrics(num_super: int, num_cat: int, property_meta=None):
    def _compute(eval_pred) -> Dict[str, float]:
        preds = getattr(eval_pred, "predictions", eval_pred)
        labels = getattr(eval_pred, "label_ids", None)
        if isinstance(preds, (tuple, list)):
            arrays = [_to_numpy(item) for item in preds]
            if len(arrays) < 3:
                raise ValueError("Predictions tuple must contain at least three arrays")
            logits_super = arrays[0]
            logits_cat_pred = arrays[1]
            logits_cat_gold = arrays[2]
            prop_presence_logits = arrays[3] if len(arrays) > 3 else None
            prop_regression = arrays[4] if len(arrays) > 4 else None
        else:
            logits = _to_numpy(preds)
            if logits is None:
                raise ValueError("Predictions are empty")
            if logits.ndim != 2 or logits.shape[1] != num_super + 2 * num_cat:
                raise ValueError(f"atteso (N, {num_super + 2 * num_cat}), trovato {logits.shape}")
            logits_super = logits[:, :num_super]
            logits_cat_pred = logits[:, num_super : num_super + num_cat]
            logits_cat_gold = logits[:, num_super + num_cat : num_super + 2 * num_cat]
            prop_presence_logits = None
            prop_regression = None

        logits_super = np.nan_to_num(logits_super, nan=0.0, posinf=1e4, neginf=-1e4)
        logits_cat_pred = np.nan_to_num(logits_cat_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        logits_cat_gold = np.nan_to_num(logits_cat_gold, nan=0.0, posinf=1e4, neginf=-1e4)

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

        if property_meta is not None and getattr(property_meta, "has_properties", lambda: False)():
            slot_mask = None
            presence_labels = None
            regression_targets = None
            regression_mask = None
            if isinstance(labels, dict):
                slot_mask = labels.get("property_slot_mask")
                presence_labels = labels.get("property_presence_labels")
                regression_targets = labels.get("property_regression_targets")
                regression_mask = labels.get("property_regression_mask")

            if (
                prop_presence_logits is not None
                and presence_labels is not None
                and slot_mask is not None
            ):
                slot_mask_np = np.asarray(slot_mask, dtype=np.float32)
                if slot_mask_np.ndim == 1:
                    slot_mask_np = slot_mask_np[:, None]
                slot_mask_np = slot_mask_np > 0.5

                presence_true = np.asarray(presence_labels, dtype=np.float32)
                if presence_true.ndim == 1:
                    presence_true = presence_true[:, None]
                presence_true = presence_true > 0.5

                probs = np.asarray(prop_presence_logits, dtype=np.float32)
                if probs.ndim == 1:
                    probs = probs[:, None]
                probs = 1.0 / (1.0 + np.exp(-probs))
                preds_presence = probs >= 0.5
                valid = slot_mask_np.reshape(slot_mask_np.shape[0], -1)
                if valid.any():
                    y_true = presence_true[valid]
                    y_pred = preds_presence[valid]
                    if y_true.size:
                        metrics["prop_presence_accuracy"] = accuracy_score(y_true, y_pred)
                        metrics["prop_presence_f1"] = f1_score(y_true, y_pred, average="binary")

            if (
                prop_regression is not None
                and regression_targets is not None
                and regression_mask is not None
            ):
                reg_mask_np = np.asarray(regression_mask, dtype=np.float32)
                if reg_mask_np.ndim == 1:
                    reg_mask_np = reg_mask_np[:, None]
                reg_mask_np = reg_mask_np > 0.5
                if reg_mask_np.any():
                    preds_reg = np.asarray(prop_regression, dtype=np.float32)
                    if preds_reg.ndim == 1:
                        preds_reg = preds_reg[:, None]
                    target_reg = np.asarray(regression_targets, dtype=np.float32)
                    if target_reg.ndim == 1:
                        target_reg = target_reg[:, None]
                    diff = preds_reg[reg_mask_np] - target_reg[reg_mask_np]
                    if diff.size:
                        metrics["prop_reg_rmse"] = float(np.sqrt(np.mean(diff**2)))
                        metrics["prop_reg_mae"] = float(np.mean(np.abs(diff)))
        return metrics

    return _compute
