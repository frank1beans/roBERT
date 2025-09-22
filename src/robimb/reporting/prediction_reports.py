"""Visualization helpers for evaluation and prediction artefacts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

DEFAULT_TOP_N = 25

__all__ = ["generate_prediction_reports"]


def _top_indices_by_support(support: np.ndarray, limit: int) -> Iterable[int]:
    order = np.argsort(support)[::-1]
    return [idx for idx in order if support[idx] > 0][:limit]


def _plot_confusion(
    matrix: np.ndarray,
    labels: Iterable[str],
    *,
    title: str,
    output_path: Path,
) -> Path:
    labels_list = list(labels)
    fig, ax = plt.subplots(figsize=(max(6.0, len(labels_list) * 0.4), 6.0))
    sns.heatmap(matrix, annot=False, cmap="mako", square=True, cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_xticks(np.arange(len(matrix)) + 0.5, labels_list, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix)) + 0.5, labels_list, rotation=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    return normalised


def _compute_top_confusions(
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    id_to_name: Mapping[int, str],
    *,
    limit: int = 10,
) -> Iterable[Dict[str, object]]:
    errors = Counter()
    for true_idx, pred_idx in zip(true_ids, pred_ids):
        if int(true_idx) == int(pred_idx):
            continue
        errors[(int(true_idx), int(pred_idx))] += 1
    results = []
    for (true_idx, pred_idx), count in errors.most_common(limit):
        results.append(
            {
                "true": id_to_name[int(true_idx)],
                "pred": id_to_name[int(pred_idx)],
                "count": int(count),
            }
        )
    return results


def generate_prediction_reports(
    *,
    pred_super: np.ndarray,
    pred_cat: np.ndarray,
    gold_super: np.ndarray,
    gold_cat: np.ndarray,
    super_id_to_name: Mapping[int, str],
    cat_id_to_name: Mapping[int, str],
    output_dir: Path,
    prefix: str = "eval",
) -> Mapping[str, Path]:
    """Create evaluation diagnostics and confusion matrix plots."""

    sns.set_theme(style="white")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artefacts: Dict[str, Path] = {}

    if gold_super.size == 0 or gold_cat.size == 0:
        return artefacts

    super_support = np.bincount(gold_super, minlength=len(super_id_to_name))
    top_super = list(_top_indices_by_support(super_support, DEFAULT_TOP_N))
    if not top_super:
        top_super = list(range(min(len(super_id_to_name), DEFAULT_TOP_N)))
    super_cm = confusion_matrix(gold_super, pred_super, labels=top_super)
    super_cm_norm = _normalise_rows(super_cm)
    super_labels = [super_id_to_name[int(idx)] for idx in top_super]
    super_plot = output_dir / f"{prefix}_super_confusion.png"
    artefacts["super_confusion_plot"] = _plot_confusion(
        super_cm_norm,
        super_labels,
        title="Matrice di confusione (super)",
        output_path=super_plot,
    )

    cat_support = np.bincount(gold_cat, minlength=len(cat_id_to_name))
    top_cat = list(_top_indices_by_support(cat_support, DEFAULT_TOP_N))
    if not top_cat:
        top_cat = list(range(min(len(cat_id_to_name), DEFAULT_TOP_N)))
    cat_cm = confusion_matrix(gold_cat, pred_cat, labels=top_cat)
    cat_cm_norm = _normalise_rows(cat_cm)
    cat_labels = [cat_id_to_name[int(idx)] for idx in top_cat]
    cat_plot = output_dir / f"{prefix}_cat_confusion.png"
    artefacts["cat_confusion_plot"] = _plot_confusion(
        cat_cm_norm,
        cat_labels,
        title="Matrice di confusione (cat)",
        output_path=cat_plot,
    )

    reports: Dict[str, object] = {
        "super_classification_report": classification_report(
            gold_super,
            pred_super,
            labels=list(range(len(super_id_to_name))),
            target_names=[super_id_to_name[idx] for idx in range(len(super_id_to_name))],
            output_dict=True,
            zero_division=0,
        ),
        "cat_classification_report": classification_report(
            gold_cat,
            pred_cat,
            labels=list(range(len(cat_id_to_name))),
            target_names=[cat_id_to_name[idx] for idx in range(len(cat_id_to_name))],
            output_dict=True,
            zero_division=0,
        ),
        "top_super_confusions": list(
            _compute_top_confusions(gold_super, pred_super, super_id_to_name)
        ),
        "top_cat_confusions": list(
            _compute_top_confusions(gold_cat, pred_cat, cat_id_to_name)
        ),
    }

    summary_path = output_dir / f"{prefix}_prediction_report.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False)
    artefacts["prediction_report"] = summary_path

    return artefacts

