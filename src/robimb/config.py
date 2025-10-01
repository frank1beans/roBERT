"""Centralized configuration and resource resolution for robimb.

This module exposes :func:`get_settings` returning the canonical locations for
runtime assets such as the knowledge pack, registry and lexicons. Paths can be
customized via environment variables or by pointing ``ROBIMB_CONFIG_FILE`` to a
TOML/YAML document.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - safety for Python <3.11
    tomllib = None  # type: ignore[assignment]

try:  # Optional dependency
    import yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - PyYAML is optional at runtime
    yaml = None  # type: ignore[assignment]

__all__ = ["ResourcePaths", "get_settings", "reset_settings"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_CACHE: Optional["ResourcePaths"] = None
_CONFIG_SOURCE: Optional[Path] = None


@dataclass(frozen=True)
class ResourcePaths:
    """Resolved filesystem locations for project resources."""

    project_root: Path
    resources_dir: Path
    data_dir: Path
    pack_dir: Path
    registry_path: Path
    lexicon_dir: Path
    prompts_path: Path
    brand_lexicon: Path
    brand_lexicon_legacy: Path
    materials_lexicon: Path
    materials_lexicon_legacy: Path
    standards_lexicon: Path
    standards_by_category: Path
    producers_by_category: Path
    colors_ral: Path
    standards_prefixes: Path

    def as_dict(self) -> Dict[str, str]:
        """Expose the resolved paths as plain strings (useful for logging)."""

        return {
            "project_root": str(self.project_root),
            "resources_dir": str(self.resources_dir),
            "data_dir": str(self.data_dir),
            "pack_dir": str(self.pack_dir),
            "registry_path": str(self.registry_path),
            "lexicon_dir": str(self.lexicon_dir),
            "prompts_path": str(self.prompts_path),
            "brand_lexicon": str(self.brand_lexicon),
            "brand_lexicon_legacy": str(self.brand_lexicon_legacy),
            "materials_lexicon": str(self.materials_lexicon),
            "materials_lexicon_legacy": str(self.materials_lexicon_legacy),
            "standards_lexicon": str(self.standards_lexicon),
            "standards_by_category": str(self.standards_by_category),
            "producers_by_category": str(self.producers_by_category),
            "colors_ral": str(self.colors_ral),
            "standards_prefixes": str(self.standards_prefixes),
        }


def _normalize_path(value: Optional[str | Path], *, base: Optional[Path]) -> Optional[Path]:
    if value is None or value == "":
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        if base is not None:
            candidate = (base / candidate).expanduser()
        if not candidate.is_absolute():
            candidate = (_PROJECT_ROOT / candidate).expanduser()
    return candidate.resolve()


def _load_config_file(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        if tomllib is None:  # pragma: no cover - Python <3.11 fallback
            raise RuntimeError("TOML configuration requires Python 3.11 or tomllib")
        with path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("YAML configuration requires the 'PyYAML' package")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
            return loaded or {}
    raise ValueError(f"Unsupported config file format: '{suffix}'")


def _coalesce_mapping(source: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    return source


def _build_paths(config_file: Optional[Path]) -> ResourcePaths:
    config_data: Mapping[str, Any] = {}
    config_dir: Optional[Path] = None
    if config_file is not None:
        config_file = _normalize_path(config_file, base=_PROJECT_ROOT)
        if config_file is not None:
            config_data = _load_config_file(config_file)
            config_dir = config_file.parent

    paths_section = _coalesce_mapping(config_data.get("paths"))
    lexicon_section = _coalesce_mapping(paths_section.get("lexicon"))

    env = os.environ

    resources_dir = _normalize_path(
        env.get("ROBIMB_RESOURCES_DIR") or paths_section.get("resources"),
        base=config_dir,
    ) or (_PROJECT_ROOT / "resources").resolve()

    data_dir = _normalize_path(
        env.get("ROBIMB_DATA_DIR") or paths_section.get("data"),
        base=config_dir,
    ) or (resources_dir / "data").resolve()

    pack_dir = _normalize_path(
        env.get("ROBIMB_PACK_DIR") or paths_section.get("pack"),
        base=config_dir,
    ) or (resources_dir / "pack").resolve()

    registry_path = _normalize_path(
        env.get("ROBIMB_REGISTRY_PATH") or paths_section.get("registry"),
        base=config_dir,
    ) or (data_dir / "properties" / "registry.json").resolve()

    lexicon_dir = _normalize_path(
        env.get("ROBIMB_LEXICON_DIR") or paths_section.get("lexicon"),
        base=config_dir,
    ) or (data_dir / "properties" / "lexicon").resolve()

    prompts_path = _normalize_path(
        env.get("ROBIMB_PROMPTS_PATH") or paths_section.get("prompts"),
        base=config_dir,
    ) or (data_dir / "properties" / "prompts.json").resolve()

    brand_lexicon = _normalize_path(
        env.get("ROBIMB_BRANDS_PATH") or lexicon_section.get("brands"),
        base=config_dir,
    ) or (lexicon_dir / "brands.json").resolve()

    brand_legacy = _normalize_path(
        env.get("ROBIMB_BRANDS_LEGACY_PATH") or lexicon_section.get("brands_legacy"),
        base=config_dir,
    ) or brand_lexicon.with_suffix(".txt")

    materials_lexicon = _normalize_path(
        env.get("ROBIMB_MATERIALS_PATH") or lexicon_section.get("materials"),
        base=config_dir,
    ) or (lexicon_dir / "materials.json").resolve()

    materials_legacy = _normalize_path(
        env.get("ROBIMB_MATERIALS_LEGACY_PATH") or lexicon_section.get("materials_legacy"),
        base=config_dir,
    ) or materials_lexicon.with_suffix(".txt")

    standards_lexicon = _normalize_path(
        env.get("ROBIMB_STANDARDS_PATH") or lexicon_section.get("standards"),
        base=config_dir,
    ) or (lexicon_dir / "norms.json").resolve()

    standards_by_category = _normalize_path(
        env.get("ROBIMB_STANDARDS_BY_CATEGORY") or lexicon_section.get("standards_by_category"),
        base=config_dir,
    ) or (lexicon_dir / "norms_by_category.json").resolve()

    producers_by_category = _normalize_path(
        env.get("ROBIMB_PRODUCERS_PATH") or lexicon_section.get("producers"),
        base=config_dir,
    ) or (lexicon_dir / "producers_by_category.json").resolve()

    colors_ral = _normalize_path(
        env.get("ROBIMB_COLORS_RAL_PATH") or lexicon_section.get("colors_ral"),
        base=config_dir,
    ) or (lexicon_dir / "colors_ral.json").resolve()

    standards_prefixes = _normalize_path(
        env.get("ROBIMB_STANDARDS_PREFIXES_PATH") or lexicon_section.get("standards_prefixes"),
        base=config_dir,
    ) or (lexicon_dir / "standards_prefixes.json").resolve()

    return ResourcePaths(
        project_root=_PROJECT_ROOT.resolve(),
        resources_dir=resources_dir,
        data_dir=data_dir,
        pack_dir=pack_dir,
        registry_path=registry_path,
        lexicon_dir=lexicon_dir,
        prompts_path=prompts_path,
        brand_lexicon=brand_lexicon,
        brand_lexicon_legacy=brand_legacy,
        materials_lexicon=materials_lexicon,
        materials_lexicon_legacy=materials_legacy,
        standards_lexicon=standards_lexicon,
        standards_by_category=standards_by_category,
        producers_by_category=producers_by_category,
        colors_ral=colors_ral,
        standards_prefixes=standards_prefixes,
    )


def get_settings(*, refresh: bool = False, config_file: str | Path | None = None) -> ResourcePaths:
    """Return the cached :class:`ResourcePaths` configuration.

    Parameters
    ----------
    refresh:
        When ``True`` the cached configuration is discarded and recomputed.
    config_file:
        Optional explicit path to the configuration document. When provided the
        returned instance is not cached globally, allowing callers (e.g. tests)
        to override settings temporarily.
    """

    global _CONFIG_CACHE, _CONFIG_SOURCE

    explicit_path = Path(config_file).expanduser() if config_file is not None else None

    if explicit_path is not None:
        return _build_paths(explicit_path)

    env_path = os.getenv("ROBIMB_CONFIG_FILE")
    source_path = Path(env_path).expanduser() if env_path else None

    if refresh or _CONFIG_CACHE is None or _CONFIG_SOURCE != source_path:
        _CONFIG_CACHE = _build_paths(source_path)
        _CONFIG_SOURCE = source_path

    return _CONFIG_CACHE


def reset_settings() -> None:
    """Clear the cached configuration (mainly useful for tests)."""

    global _CONFIG_CACHE, _CONFIG_SOURCE
    _CONFIG_CACHE = None
    _CONFIG_SOURCE = None
