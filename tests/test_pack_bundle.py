from __future__ import annotations

import json
from pathlib import Path

from robimb.registry import RegistryLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_pack_current_symlink_points_to_version() -> None:
    repo_root = _repo_root()
    pack_root = repo_root / "pack"
    current = pack_root / "current"

    assert current.exists(), "pack/current deve esistere"
    target = current.resolve()
    assert target.parent == pack_root
    assert target.name.startswith("v"), "La directory target deve essere versionata (vX)"

    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("metadata", {}).get("version") == target.name
    sources = manifest.get("sources", {})
    for name in ("registry", "extractors", "validators", "formulas", "views", "templates", "profiles", "contexts"):
        assert sources.get(name) == f"{name}.json"


def test_registry_loader_can_read_versioned_bundle() -> None:
    loader = RegistryLoader()
    bundle = loader.bundle()

    assert bundle.registry, "Il registry non dovrebbe essere vuoto"
    assert bundle.extractors.get("patterns"), "Gli extractors devono contenere pattern"
    assert isinstance(bundle.manifest, dict)

    categories = loader.load_registry()
    assert categories, "Attese categorie nel registry"
    for schema in categories.values():
        schema_dict = schema.json_schema()
        assert "slots" in schema_dict and schema_dict["slots"], "Ogni schema deve esporre slot"
