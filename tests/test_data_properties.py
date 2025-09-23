import json
from pathlib import Path

import pytest

from robimb.extraction.resources import load_default
from robimb.features.extractors import extract_properties
from robimb.utils.data_utils import prepare_classification_dataset


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_classification_dataset_enriches_properties(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    rows = [
        {"text": "Parete sp 20 cm", "super": "Strutture", "cat": "Pareti"},
    ]
    _write_jsonl(train_path, rows)
    _write_jsonl(val_path, rows)

    label_maps_path = tmp_path / "labels.json"
    label_maps = {
        "super2id": {"Strutture": 0},
        "cat2id": {"Pareti": 0},
    }
    label_maps_path.write_text(json.dumps(label_maps), encoding="utf-8")

    registry_path = tmp_path / "registry.json"
    registry = {
        "Strutture|Pareti": {
            "slots": {"geo.spessore": {"type": "float"}},
        }
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    extractors_path = tmp_path / "extractors.json"
    extractors = {
        "patterns": [
            {
                "property_id": "geo.spessore",
                "regex": [r"sp\s*(\d+)"],
                "normalizers": ["to_float"],
            }
        ]
    }
    extractors_path.write_text(json.dumps(extractors), encoding="utf-8")

    train_df, val_df, _, _ = prepare_classification_dataset(
        train_path,
        val_path,
        label_maps_path=label_maps_path,
        ontology_path=None,
        properties_registry_path=registry_path,
        extractors_pack_path=extractors_path,
    )

    assert "properties" in train_df.columns
    first_props = train_df.iloc[0]["properties"]
    assert pytest.approx(first_props["geo.spessore"], rel=1e-6) == 20.0

    assert "property_schema" in train_df.columns
    assert train_df.iloc[0]["property_schema"]["slots"] == {"geo.spessore": {"type": "float"}}

    val_props = val_df.iloc[0]["properties"]
    assert pytest.approx(val_props["geo.spessore"], rel=1e-6) == 20.0


def test_pack_extractors_normalize_ei_and_spessore_cm():
    extractors_pack = load_default()

    text = "Parete EI 60 con spessore 12 cm"
    props = extract_properties(text, extractors_pack)

    assert props["frs.resistenza_fuoco"] == "EI60"
    assert pytest.approx(props["geo.spessore_elemento"], rel=1e-6) == 120.0


def test_cm_targets_keep_centimetres():
    extractors_pack = load_default()

    text = "Rivestimento con lastre spessore 1,4 cm"
    props = extract_properties(text, extractors_pack)

    assert pytest.approx(props["spessore_lastre_cm"], rel=1e-6) == 1.4
