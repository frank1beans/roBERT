import json
from pathlib import Path

import pytest

from robimb.cli import predict as predict_cli
from robimb.inference.category import CategoryInference


def test_extract_numeric_properties_handles_nested():
    data = {
        "length": 120,
        "width": {"value": 60.5, "unit": "cm"},
        "note": "N/A",
        "invalid": {"raw": "foo"},
    }
    expected = {"length": 120.0, "width": 60.5}
    assert predict_cli._extract_numeric_properties(data) == expected


def test_category_inference_coerce_id2label_casts_keys():
    mapping = CategoryInference._coerce_id2label({"0": "A", 1: "B", "x": "ignored"})
    assert mapping == {0: "A", 1: "B"}


def test_category_inference_load_id2label_from_file(tmp_path: Path):
    payload = {"0": "cat_a", "1": "cat_b"}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = CategoryInference._load_id2label_from_file(path)
    assert result == {0: "cat_a", 1: "cat_b"}


def test_category_inference_load_id2label_invalid(tmp_path: Path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")

    with pytest.raises(ValueError):
        CategoryInference._load_id2label_from_file(path)
