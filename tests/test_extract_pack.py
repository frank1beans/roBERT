import json
from pathlib import Path

import pytest

from robimb.extraction.engine import dry_run


def _load_extractors() -> dict:
    pack_path = Path(__file__).resolve().parents[1] / "data" / "properties" / "extractors.json"
    return json.loads(pack_path.read_text(encoding="utf-8"))


def test_extraction_pack_handles_area_volume_and_fire_resistance():
    extractors_pack = _load_extractors()
    text = (
        "Parete EI 120 con spessore 25 cm, superficie 45 mq e volume calcestruzzo 3.5 m3."
    )

    result = dry_run(text, extractors_pack)
    extracted = result["extracted"]

    classe_values = [value for key, value in extracted.items() if key.endswith("classe_ei")]
    assert "EI120" in classe_values

    spessore_values = [value for key, value in extracted.items() if key.endswith("spessore_mm")]
    assert any(pytest.approx(val, rel=1e-6) == 250.0 for val in spessore_values)

    superficie_values = [value for key, value in extracted.items() if key.endswith("superficie_m2")]
    assert any(pytest.approx(val, rel=1e-6) == 45.0 for val in superficie_values)

    volume_values = [value for key, value in extracted.items() if key.endswith("volume_m3")]
    assert any(pytest.approx(val, rel=1e-6) == 3.5 for val in volume_values)
