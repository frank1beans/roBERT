import json
from pathlib import Path

from robimb.extraction.engine import dry_run


def _normalize(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def test_extraction_pack_samples():
    base_dir = Path(__file__).resolve().parents[1]
    pack_path = base_dir / "pack" / "current" / "pack.json"
    samples_path = Path(__file__).resolve().parent / "data" / "extraction_samples.json"

    extractors_pack = json.loads(pack_path.read_text(encoding="utf-8"))
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    differences = []

    for idx, sample in enumerate(samples, start=1):
        text = sample["text"]
        expected = {key: _normalize(value) for key, value in sample["expected"].items()}
        extracted = dry_run(text, extractors_pack)["extracted"]
        normalized = {key: _normalize(value) for key, value in extracted.items()}

        for key, value in expected.items():
            if key not in normalized:
                differences.append(f"Sample {idx}: missing property {key}")
            elif normalized[key] != value:
                differences.append(
                    f"Sample {idx}: value mismatch for {key}: expected {value!r}, got {normalized[key]!r}"
                )
        for key, value in normalized.items():
            if key not in expected:
                differences.append(f"Sample {idx}: unexpected property {key}={value!r}")

    if differences:
        diff_text = "\n".join(differences)
        print(diff_text)
        raise AssertionError(f"Extraction differences detected:\n{diff_text}")
