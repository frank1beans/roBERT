import json
from pathlib import Path

import numpy as np

from robimb.utils.dataset_prep import (
    LabelMaps,
    build_mask_and_report,
    create_or_load_label_maps,
    prepare_classification_dataset,
    prepare_mlm_corpus,
    save_datasets,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_label_maps_creation_and_mask(tmp_path: Path) -> None:
    ontology_payload = {"super_to_cats": {"Super": ["Cat"]}}
    ontology_path = tmp_path / "ontology.json"
    ontology_path.write_text(json.dumps(ontology_payload), encoding="utf-8")

    label_maps_path = tmp_path / "label_maps.json"
    label_maps = create_or_load_label_maps(label_maps_path, ontology_path=ontology_path)
    assert isinstance(label_maps, LabelMaps)
    assert label_maps_path.exists()
    assert label_maps.super_name_to_id["Super"] == 1
    mask, report = build_mask_and_report(ontology_path, label_maps)
    assert isinstance(mask, np.ndarray)
    assert mask.shape[0] >= 2  # includes fallback row
    assert report["coverage"] >= 1.0


def test_prepare_dataset_and_serialization(tmp_path: Path) -> None:
    ontology_payload = {"super_to_cats": {"Alpha": ["CatA"]}}
    ontology_path = tmp_path / "ontology.json"
    ontology_path.write_text(json.dumps(ontology_payload), encoding="utf-8")

    label_maps_path = tmp_path / "label_maps.json"
    label_maps = create_or_load_label_maps(label_maps_path, ontology_path=ontology_path)

    train_rows = [
        {"text": "Elemento uno", "super": "Alpha", "cat": "CatA"},
        {"text": "Elemento due", "super": "Alpha", "cat": "CatA"},
    ]
    val_rows = [{"text": "Elemento tre", "super": "Alpha", "cat": "CatA"}]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    train_df, val_df, resulting_maps = prepare_classification_dataset(
        train_path,
        val_path,
        label_maps_path=label_maps_path,
        ontology_path=ontology_path,
    )
    assert len(train_df) == 2
    assert len(val_df) == 1
    expected_columns = {"super_label", "cat_label", "property_schema", "properties"}
    assert expected_columns.issubset(set(train_df.columns))
    assert resulting_maps.super_name_to_id == label_maps.super_name_to_id

    out_dir = tmp_path / "out"
    save_datasets(train_df, val_df, out_dir)
    assert (out_dir / "train_processed.jsonl").exists()
    assert (out_dir / "val_processed.jsonl").exists()

    mlm_path = tmp_path / "mlm.txt"
    num_lines = prepare_mlm_corpus([train_path, val_path], mlm_path, min_len=1)
    assert num_lines == 3
    assert mlm_path.read_text(encoding="utf-8").count("\n") == 3

    mask, _ = build_mask_and_report(None, label_maps)
    assert np.all(mask == 1.0)
