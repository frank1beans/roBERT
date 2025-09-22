"""Validate dataset and prediction reporting helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robimb.reporting import generate_dataset_reports, generate_prediction_reports


def test_generate_dataset_reports(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "text": ["a b c", "d e", "f g h i"],
            "super_label": [0, 0, 1],
            "cat_label": [0, 1, 2],
        }
    )
    val_df = pd.DataFrame(
        {
            "text": ["l m n"],
            "super_label": [1],
            "cat_label": [2],
        }
    )
    super_map = {0: "Super A", 1: "Super B"}
    cat_map = {0: "Cat A", 1: "Cat B", 2: "Cat C"}

    artefacts = generate_dataset_reports(
        train_df,
        val_df,
        super_id_to_name=super_map,
        cat_id_to_name=cat_map,
        output_dir=tmp_path,
    )

    expected_keys = {
        "train_text_length_plot",
        "train_super_distribution_plot",
        "train_cat_distribution_plot",
        "val_text_length_plot",
        "val_super_distribution_plot",
        "val_cat_distribution_plot",
        "dataset_summary",
    }
    assert expected_keys.issubset(artefacts.keys())
    for path in artefacts.values():
        assert Path(path).exists()


def test_generate_prediction_reports(tmp_path: Path) -> None:
    super_map = {0: "Super A", 1: "Super B"}
    cat_map = {0: "Cat A", 1: "Cat B"}
    gold_super = np.array([0, 0, 1, 1])
    pred_super = np.array([0, 1, 1, 1])
    gold_cat = np.array([0, 1, 0, 1])
    pred_cat = np.array([0, 1, 1, 1])

    artefacts = generate_prediction_reports(
        pred_super=pred_super,
        pred_cat=pred_cat,
        gold_super=gold_super,
        gold_cat=gold_cat,
        super_id_to_name=super_map,
        cat_id_to_name=cat_map,
        output_dir=tmp_path,
        prefix="unit",
    )

    expected = {
        "super_confusion_plot",
        "cat_confusion_plot",
        "prediction_report",
    }
    assert expected.issubset(artefacts.keys())
    for path in artefacts.values():
        assert Path(path).exists()
