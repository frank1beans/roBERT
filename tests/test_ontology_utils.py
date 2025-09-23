"""Tests for ontology and label map utilities."""

from pathlib import Path

import json

from robimb.utils.ontology_utils import FALLBACK_LABEL, load_label_maps, load_ontology


def _write_ontology(path: Path) -> None:
    payload = {"Super": ["CatA", "CatB"]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_label_maps_creates_fallback(tmp_path):
    ontology_path = tmp_path / "ontology.json"
    _write_ontology(ontology_path)
    label_maps_path = tmp_path / "label_maps.json"

    super_map, cat_map, id2super, id2cat = load_label_maps(
        label_maps_path,
        ontology=load_ontology(ontology_path),
        create_if_missing=True,
    )

    assert super_map[FALLBACK_LABEL] == 0
    assert cat_map[FALLBACK_LABEL] == 0
    assert id2super[0] == FALLBACK_LABEL
    assert id2cat[0] == FALLBACK_LABEL


def test_load_label_maps_normalises_existing_file(tmp_path):
    label_maps_path = tmp_path / "label_maps.json"
    label_maps_path.write_text(
        json.dumps(
            {
                "super2id": {"Super": 0},
                "cat2id": {"Cat": 0},
            }
        ),
        encoding="utf-8",
    )

    super_map, cat_map, id2super, id2cat = load_label_maps(label_maps_path)

    assert super_map[FALLBACK_LABEL] == 0
    assert cat_map[FALLBACK_LABEL] == 0
    assert super_map["Super"] == 1
    assert cat_map["Cat"] == 1
    assert id2super[0] == FALLBACK_LABEL
    assert id2cat[0] == FALLBACK_LABEL
