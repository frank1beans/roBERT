
from __future__ import annotations
import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def plot_confusion(cm: np.ndarray, labels: List[str], out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)  # default colormap; do not set colors
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_reliability(bins_detail: Dict[str, Any], out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    accs = [b["acc"] for b in bins_detail["bins"]]
    confs = [b["avg_conf"] for b in bins_detail["bins"]]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Expected line
    ax.plot([0,1], [0,1])
    # Reliability
    ax.plot(confs, accs, marker="o")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_confidence_hist(conf: np.ndarray, out_path: str, bins: int = 20):
    _ensure_dir(os.path.dirname(out_path))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(conf, bins=bins)
    ax.set_xlabel("Confidence (max prob)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Histogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_topk_curve(topk_acc: Dict[int, float], out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    ks = sorted(topk_acc.keys())
    vals = [topk_acc[k] for k in ks]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ks, vals, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy@k")
    ax.set_title("Top-k Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_risk_coverage(probs: np.ndarray, labels: np.ndarray, out_path: str, steps: int = 50):
    # Risk-Coverage (selective prediction): threshold on confidence
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    ths = np.linspace(0, 1, steps)
    cover = []
    accs = []
    for t in ths:
        mask = conf >= t
        if mask.sum() == 0:
            cover.append(0.0); accs.append(0.0)
        else:
            cover.append(float(mask.mean()))
            accs.append(float(correct[mask].mean()))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cover, accs, marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy")
    ax.set_title("Risk-Coverage Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_pr_macro(pr_auc_macro: float, out_path: str):
    # Simple bar with macro PR AUC value (keep single plot)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar([0], [pr_auc_macro])
    ax.set_xticks([0]); ax.set_xticklabels(["PR AUC (macro)"])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_roc_macro(roc_auc_macro: float, out_path: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar([0], [roc_auc_macro])
    ax.set_xticks([0]); ax.set_xticklabels(["ROC AUC (macro)"])
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
