"""Tests for schema-based property validation."""

import pytest

from robimb.extraction.validators import validate_properties


@pytest.fixture(scope="module")
def sample_payload() -> dict[str, dict[str, object]]:
    return {
        "tipologia_lastra": {
            "value": "ignifuga",
            "source": "qa_llm",
            "span": [12, 20],
            "confidence": 0.91,
        },
        "spessore_mm": {
            "value": "125",
            "unit": "mm",
            "source": "parser",
            "span": [30, 36],
            "confidence": 0.88,
        },
        "classe_reazione_al_fuoco": {
            "value": "A2-s1,d0",
            "source": "parser",
            "span": [40, 49],
            "confidence": 0.92,
        },
    }


def test_validate_properties_success(sample_payload: dict[str, dict[str, object]]) -> None:
    result = validate_properties("Opere da cartongessista", sample_payload)

    assert result.ok is True
    assert not result.errors
    assert set(result.normalized.keys()) == {"tipologia_lastra", "spessore_mm", "classe_reazione_al_fuoco"}
    assert result.normalized["spessore_mm"].value == pytest.approx(125.0)


def test_missing_required_property(sample_payload: dict[str, dict[str, object]]) -> None:
    payload = {key: value for key, value in sample_payload.items() if key != "spessore_mm"}

    result = validate_properties("Opere da cartongessista", payload)

    assert result.ok is False
    assert any(issue.code == "missing_required" and issue.property_id == "spessore_mm" for issue in result.errors)


def test_enum_mismatch_triggers_error(sample_payload: dict[str, dict[str, object]]) -> None:
    payload = dict(sample_payload)
    payload["classe_reazione_al_fuoco"] = {
        "value": "C-s3,d2",
        "source": "qa_llm",
        "span": [5, 10],
        "confidence": 0.5,
    }

    result = validate_properties("Opere da cartongessista", payload)

    assert result.ok is False
    assert any(issue.code == "enum_mismatch" for issue in result.errors)


def test_type_conversion_error_recorded(sample_payload: dict[str, dict[str, object]]) -> None:
    payload = dict(sample_payload)
    payload["spessore_mm"] = {
        "value": "cento",
        "unit": "mm",
        "source": "parser",
        "span": [1, 5],
        "confidence": 0.7,
    }

    result = validate_properties("Opere da cartongessista", payload)

    assert result.ok is False
    assert any(issue.code == "type_conversion" for issue in result.errors)
