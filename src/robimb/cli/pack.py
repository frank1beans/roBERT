"""Utilities to pack properties folders into versioned knowledge bundles."""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

import typer

from ..registry import pack_folders_to_monolith
from ..config import get_settings

__all__ = ["pack_command"]

_DEFAULT_PACK_ROOT = get_settings().pack_dir
_EMPTY_COMPONENTS: Mapping[str, Mapping[str, object]] = {
    "validators.json": {"schema": "validators/v1", "rules": []},
    "formulas.json": {"schema": "formulas/v1", "formulas": []},
    "views.json": {"schema": "views/v1", "views": []},
    "templates.json": {"schema": "templates/v1", "templates": []},
    "profiles.json": {"schema": "profiles/v1", "profiles": []},
    "contexts.json": {"schema": "contexts/v1", "contexts": []},
}


def _next_version(pack_root: Path) -> str:
    pattern = re.compile(r"^v(\d+)$")
    existing = [
        int(match.group(1))
        for child in pack_root.iterdir()
        if child.is_dir() and (match := pattern.match(child.name))
    ]
    next_idx = max(existing) + 1 if existing else 1
    return f"v{next_idx}"


def _dump_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(version_dir: Path, version: str) -> None:
    manifest_path = version_dir / "manifest.json"
    manifest = {
        "schema": "manifest/v1",
        "metadata": {
            "version": version,
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "description": "Knowledge pack bundle generated via robimb pack",
        },
        "sources": {
            "registry": "registry.json",
            "extractors": "extractors.json",
            "validators": "validators.json",
            "formulas": "formulas.json",
            "views": "views.json",
            "templates": "templates.json",
            "profiles": "profiles.json",
            "contexts": "contexts.json",
        },
    }
    _dump_json(manifest_path, manifest)


def _update_current_symlink(pack_root: Path, target_dir: Path) -> None:
    current_link = pack_root / "current"
    if current_link.exists() or current_link.is_symlink():
        if current_link.is_symlink() or current_link.is_file():
            current_link.unlink()
        else:
            shutil.rmtree(current_link)
    current_link.symlink_to(target_dir.name, target_is_directory=True)


def _write_empty_components(version_dir: Path) -> None:
    for filename, payload in _EMPTY_COMPONENTS.items():
        _dump_json(version_dir / filename, payload)


def pack_command(
    properties_root: Path = typer.Option(..., "--properties-root", exists=True, file_okay=False),
    pack_root: Optional[Path] = typer.Option(None, "--pack-root", help="Directory that will contain versioned bundles"),
    version: Optional[str] = typer.Option(None, "--version", help="Version label for the new bundle (e.g. v3)"),
    set_current: bool = typer.Option(True, "--set-current/--no-set-current", help="Update the pack/current symlink"),
    out_registry: Optional[Path] = typer.Option(None, "--out-registry", help="Optional direct output path for registry.json"),
    out_extractors: Optional[Path] = typer.Option(None, "--out-extractors", help="Optional direct output path for extractors.json"),
) -> None:
    """Pack properties folders into versioned bundles or standalone JSON files."""

    if (out_registry is None) ^ (out_extractors is None):
        raise typer.BadParameter("Specificare entrambi --out-registry e --out-extractors oppure nessuno dei due.")

    if out_registry and out_extractors:
        pack_folders_to_monolith(properties_root, out_registry, out_extractors)
        typer.echo(
            (
                "Pack creato.\n"
                f"  registry:   {out_registry}\n"
                f"  extractors: {out_extractors}"
            )
        )
        return

    target_root = pack_root or _DEFAULT_PACK_ROOT
    target_root.mkdir(parents=True, exist_ok=True)

    bundle_version = version or _next_version(target_root)
    version_dir = target_root / bundle_version
    if version_dir.exists():
        raise typer.BadParameter(f"La directory {version_dir} esiste già: specificare un'altra versione.")

    registry_path = version_dir / "registry.json"
    extractors_path = version_dir / "extractors.json"
    pack_folders_to_monolith(properties_root, registry_path, extractors_path)

    _write_empty_components(version_dir)
    _write_manifest(version_dir, bundle_version)

    if set_current:
        _update_current_symlink(target_root, version_dir)

    typer.echo(
        (
            "Bundle versionato creato.\n"
            f"  directory:  {version_dir}\n"
            f"  registry:   {registry_path}\n"
            f"  extractors: {extractors_path}"
        )
    )
