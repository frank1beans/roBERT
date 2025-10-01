from pathlib import Path

from robimb.config import get_settings
from robimb.extraction.schema_registry import load_registry


def test_load_registry_categories(tmp_path: Path) -> None:
    registry_path = get_settings().registry_path
    registry = load_registry(registry_path)
    category = registry.get("Opere da cartongessista")
    assert category is not None
    assert category.id == "opere_da_cartongessista"
    assert category.schema_path.exists()
    assert {prop.id for prop in category.properties} >= {
        "tipologia_lastra",
        "spessore_mm",
        "classe_reazione_al_fuoco",
    }


def test_list_all_categories() -> None:
    registry = load_registry(get_settings().registry_path)
    categories = list(registry.list())
    assert {category.id for category in categories} == {
        "opere_da_cartongessista",
        "opere_di_rivestimento",
        "opere_di_pavimentazione",
        "opere_da_serramentista",
        "controsoffitti",
        "apparecchi_sanitari_accessori",
        "opere_da_falegname",
    }
