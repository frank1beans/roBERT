
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, roc_curve, auc, log_loss, brier_score_loss

@dataclass
class EvalResults:
    accuracy: float
    f1_macro: float
    f1_micro: float
    per_class_f1: Dict[int, float]
    confusion: np.ndarray
    topk_accuracy: Dict[int, float]
    nll: float
    brier: float
    ece: float
    mce: float
    pr_auc_macro: float
    roc_auc_macro: float

def topk_accuracies(probs: np.ndarray, labels: np.ndarray, ks=(1,3,5)) -> Dict[int, float]:
    # probs: [N, C], labels: [N]
    res = {}
    sorted_idx = np.argsort(-probs, axis=1)
    for k in ks:
        topk = sorted_idx[:, :min(k, probs.shape[1])]
        hits = np.any(topk == labels.reshape(-1,1), axis=1).astype(np.float32)
        res[k] = float(hits.mean())
    return res

def calibration_errors(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, float, dict]:
    # ECE/MCE (Expected/Maximum Calibration Error), confidence = max prob
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    mce = 0.0
    details = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i>0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            acc = 0.0; avg_conf = (lo+hi)/2.0; frac = 0.0
        else:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            frac = float(mask.mean())
        gap = abs(avg_conf - acc)
        ece += gap * frac
        mce = max(mce, gap)
        details.append({"bin": i+1, "lo": lo, "hi": hi, "avg_conf": avg_conf, "acc": acc, "frac": frac})
    return float(ece), float(mce), {"bins": details}

def macro_curves(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    # one-vs-rest macro PR AUC and ROC AUC
    from sklearn.preprocessing import label_binarize
    C = probs.shape[1]
    Y = label_binarize(labels, classes=list(range(C)))
    pr_aucs = []; roc_aucs = []
    for c in range(C):
        y_true = Y[:, c]
        y_score = probs[:, c]
        # PR
        p, r, _ = precision_recall_curve(y_true, y_score)
        pr_aucs.append(auc(r, p) if len(p)>1 and len(r)>1 else 0.0)
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_aucs.append(auc(fpr, tpr) if len(fpr)>1 and len(tpr)>1 else 0.0)
    return float(np.mean(pr_aucs)), float(np.mean(roc_aucs))

def evaluate_all(probs: np.ndarray, labels: np.ndarray) -> EvalResults:
    preds = probs.argmax(axis=1)
    acc = float(accuracy_score(labels, preds))
    f1m = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1mi = float(f1_score(labels, preds, average="micro", zero_division=0))
    # per-class f1
    C = probs.shape[1]
    per_class = {}
    for c in range(C):
        per_class[c] = float(f1_score((labels==c).astype(int), (preds==c).astype(int), zero_division=0))
    # confusion
    cm = confusion_matrix(labels, preds, labels=list(range(C)))
    # top-k
    tk = topk_accuracies(probs, labels, ks=(1,3,5))
    # calibration metrics
    ece, mce, _ = calibration_errors(probs, labels, n_bins=10)
    # nll/log-loss
    nll = float(log_loss(labels, probs, labels=list(range(C)), eps=1e-15))
    # brier (multi-class: average one-vs-rest)
    Y = np.eye(C)[labels]
    brier = float(((probs - Y) ** 2).sum(axis=1).mean())
    # macro PR/ROC
    pr_auc, roc_auc = macro_curves(probs, labels)
    return EvalResults(acc, f1m, f1mi, per_class, cm, tk, nll, brier, ece, mce, pr_auc, roc_auc)
