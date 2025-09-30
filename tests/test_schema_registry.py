from pathlib import Path

from robimb.extraction.schema_registry import load_registry


def test_load_registry_categories(tmp_path: Path) -> None:
    registry_path = Path("data/properties/registry.json")
    registry = load_registry(registry_path)
    category = registry.get("Porte HPL")
    assert category is not None
    assert category.id == "porte_hpl"
    assert category.schema_path.exists()
    assert {prop.id for prop in category.properties} >= {
        "larghezza_anta_mm",
        "altezza_anta_mm",
        "classe_rei",
    }


def test_list_all_categories() -> None:
    registry = load_registry(Path("data/properties/registry.json"))
    categories = list(registry.list())
    assert {category.id for category in categories} == {
        "porte_hpl",
        "controsoffitti_metallici",
        "pavimenti_sopraelevati",
    }
