import json
from pathlib import Path

from robimb.utils.registry_io import (
    ExtractorsPack,
    build_registry_extractors,
    load_extractors_pack,
    load_property_registry,
    merge_extractors_pack,
)


def test_load_property_registry_and_build_extractors() -> None:
    registry_path = Path("data/properties/registry.json")
    registry = load_property_registry(registry_path)
    assert registry is not None
    assert registry  # not empty
    pack = build_registry_extractors(registry)
    assert pack is not None
    mapping = pack.to_mapping()
    assert "patterns" in mapping
    assert mapping["patterns"]  # contains at least one pattern


def test_load_extractors_pack_and_merge(tmp_path: Path) -> None:
    primary_payload = {
        "patterns": [{"property_id": "p1", "regex": ["foo"]}],
        "normalizers": {"trim": {"type": "strip"}},
    }
    secondary_payload = {
        "patterns": [{"property_id": "p2", "regex": ["bar"]}],
        "normalizers": {"lower": {"type": "lower"}},
    }
    primary_path = tmp_path / "extractors.json"
    with primary_path.open("w", encoding="utf-8") as handle:
        json.dump(primary_payload, handle)

    pack = load_extractors_pack(primary_path)
    assert pack is not None

    merged = merge_extractors_pack(pack, ExtractorsPack.from_payload(secondary_payload))
    assert merged is not None
    mapping = merged.to_mapping()
    assert len(mapping["patterns"]) == 2
    assert set(mapping["normalizers"].keys()) == {"trim", "lower"}
