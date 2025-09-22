"""Generate visual analytics for training and validation datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402  -- ensure headless environments work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_TOP_N = 30

__all__ = ["generate_dataset_reports"]


def _render_bar_plot(
    counts: MutableMapping[int, int],
    *,
    id_to_name: Mapping[int, str],
    title: str,
    output_path: Path,
    top_n: int = DEFAULT_TOP_N,
) -> Path:
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if top_n:
        ordered = ordered[:top_n]
    labels = [id_to_name[int(idx)] for idx, _ in ordered]
    values = [int(value) for _, value in ordered]

    height = max(4.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(10, height))
    sns.barplot(x=values, y=labels, color="#1f77b4", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frequenza")
    ax.set_ylabel("Label")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _render_histogram(lengths: pd.Series, title: str, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(lengths, bins=min(50, max(5, lengths.nunique())), ax=ax, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Lunghezza del testo")
    ax.set_ylabel("Conteggio")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _compute_basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    lengths = df["text"].astype(str).str.len()
    return {
        "num_records": int(len(df)),
        "avg_text_length": float(lengths.mean()),
        "median_text_length": float(lengths.median()),
        "p95_text_length": float(lengths.quantile(0.95)),
    }


def generate_dataset_reports(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    *,
    super_id_to_name: Mapping[int, str],
    cat_id_to_name: Mapping[int, str],
    output_dir: Path,
) -> Mapping[str, Path]:
    """Create dataset visual reports and summary statistics."""

    sns.set_theme(style="whitegrid")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assets: Dict[str, Path] = {}

    summary: Dict[str, Dict[str, float]] = {}
    for split_name, df in ("train", train_df), ("val", val_df):
        if df is None or df.empty:
            continue
        df = df.copy()
        summary[split_name] = _compute_basic_stats(df)
        length_path = output_dir / f"{split_name}_text_length.png"
        assets[f"{split_name}_text_length_plot"] = _render_histogram(
            df["text"].astype(str).str.len(),
            f"Distribuzione lunghezze testo ({split_name})",
            length_path,
        )

        super_counts = df["super_label"].value_counts().to_dict()
        super_plot_path = output_dir / f"{split_name}_super_distribution.png"
        assets[f"{split_name}_super_distribution_plot"] = _render_bar_plot(
            super_counts,
            id_to_name=super_id_to_name,
            title=f"Distribuzione classi super ({split_name})",
            output_path=super_plot_path,
        )

        cat_counts = df["cat_label"].value_counts().to_dict()
        cat_plot_path = output_dir / f"{split_name}_cat_distribution.png"
        assets[f"{split_name}_cat_distribution_plot"] = _render_bar_plot(
            cat_counts,
            id_to_name=cat_id_to_name,
            title=f"Distribuzione classi cat ({split_name})",
            output_path=cat_plot_path,
        )

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    assets["dataset_summary"] = summary_path

    return assets

