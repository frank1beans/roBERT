from __future__ import annotations

import json
from pathlib import Path

from robimb.data.pack_merge import build_merged_pack, write_pack_index


def test_build_merged_pack(tmp_path: Path) -> None:
    out_dir = tmp_path / "pack" / "v1"
    artifacts = build_merged_pack(
        data_dir=Path("data"),
        output_dir=out_dir,
        version="9.9.9",
        timestamp="2025-01-01T00:00:00Z",
    )

    expected_keys = {
        "registry",
        "extractors",
        "validators",
        "formulas",
        "views",
        "templates",
        "profiles",
        "contexts",
        "categories",
        "catmap",
        "manifest",
    }
    assert expected_keys.issubset(artifacts.files.keys())

    catmap_path = artifacts.files["catmap"]
    with catmap_path.open("r", encoding="utf-8") as handle:
        catmap = json.load(handle)
    laterizio = next(item for item in catmap["mappings"] if item["cat_label"] == "Elementi in laterizio")
    assert "frs.resistenza_fuoco" in laterizio.get("props_required", [])
    assert "slot_priority" in laterizio and laterizio["slot_priority"]

    registry_path = artifacts.files["registry"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert "idn.ifc_class" in registry["properties"]

    profiles_path = artifacts.files["profiles"]
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    arch_profile = next(item for item in profiles["profiles"] if item["id"] == "arch.concept")
    assert arch_profile.get("label")
    overrides = arch_profile.get("property_overrides", {})
    assert "frs.resistenza_fuoco" in overrides

    contexts_path = artifacts.files["contexts"]
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    assert contexts["dimensions"]["intervention_type"]["label"] == "Tipo intervento"

    extractors_path = artifacts.files["extractors"]
    extractors = json.loads(extractors_path.read_text(encoding="utf-8"))
    spessore_entry = next(item for item in extractors["patterns"] if item["property_id"] == "qty.spessore")
    assert "cm_to_mm?" in spessore_entry.get("normalizers", [])

    manifest_path = artifacts.files["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]
    manifest_extractors = next(item for item in manifest["files"] if item["name"] == "extractors")
    assert manifest_extractors["path"].endswith("src/robimb/extraction/resources/extractors.json")

    current_dir = tmp_path / "pack" / "current"
    index_path = write_pack_index(artifacts, current_dir)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["files"]["catmap"].endswith("catmap.json")
