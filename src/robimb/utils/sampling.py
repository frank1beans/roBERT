"""Sampling utilities for dataset inspection and fixtures."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

__all__ = ["load_jsonl_to_df", "sample_one_record_per_category"]


def load_jsonl_to_df(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def sample_one_record_per_category(
    path: str | Path,
    *,
    category_field: str = "cat",
) -> List[Dict[str, Any]]:
    """Return the first occurrence for each category found in ``path``."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_path}")

    samples: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Riga {line_no}: JSON non valido") from exc

            category = row.get(category_field)
            if isinstance(category, str) and category and category not in samples:
                samples[category] = row

    if not samples:
        raise ValueError(
            f"Nessuna categoria trovata usando il campo '{category_field}' in {dataset_path}"
        )

    return list(samples.values())
