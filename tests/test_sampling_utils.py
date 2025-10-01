from pathlib import Path

import pandas as pd

from robimb.utils.sampling import load_jsonl_to_df, sample_one_record_per_category


def test_load_jsonl_to_df(tmp_path: Path) -> None:
    payload = [
        {"text": "uno", "cat": "A"},
        {"text": "due", "cat": "B"},
    ]
    jsonl = tmp_path / "dataset.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in payload:
            handle.write(f"{row}\n".replace("'", '"'))

    df = load_jsonl_to_df(jsonl)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert df.loc[0, "text"] == "uno"


def test_sample_one_record_per_category(tmp_path: Path) -> None:
    rows = [
        {"cat": "alpha", "value": 1},
        {"cat": "beta", "value": 2},
        {"cat": "alpha", "value": 3},
    ]
    jsonl = tmp_path / "records.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n".replace("'", '"'))

    samples = sample_one_record_per_category(jsonl, category_field="cat")
    assert len(samples) == 2
    assert samples[0]["value"] == 1  # first occurrence retained
    assert samples[1]["value"] == 2
