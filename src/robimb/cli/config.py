"""Utility commands to inspect and validate robimb resource paths."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Tuple

import typer

from ..config import ResourcePaths, get_settings

__all__ = ["app"]

app = typer.Typer(
    help="Strumenti di diagnostica per la configurazione del knowledge pack.",
    add_completion=False,
)


def _inventory(paths: ResourcePaths) -> Dict[str, Dict[str, str | bool]]:
    def describe(path_str: str) -> Tuple[str, bool, str]:
        path = Path(path_str)
        exists = path.exists()
        if path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            kind = "missing"
        return str(path), exists, kind

    inventory: Dict[str, Dict[str, str | bool]] = {}
    for key, value in paths.as_dict().items():
        resolved, exists, kind = describe(value)
        inventory[key] = {
            "path": resolved,
            "exists": exists,
            "kind": kind,
        }
    return inventory


def _check_registry_version(registry_path: Path, expected_version: str) -> Dict[str, str | bool]:
    if not registry_path.exists():
        return {
            "path": str(registry_path),
            "status": "missing",
            "version_matches": False,
        }

    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        return {
            "path": str(registry_path),
            "status": f"invalid_json: {exc}",
            "version_matches": False,
        }

    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    version = ""
    schema = ""
    if isinstance(metadata, dict):
        version = str(metadata.get("version", ""))
        schema = str(metadata.get("schema", "")) if metadata.get("schema") else ""

    return {
        "path": str(registry_path),
        "status": "ok",
        "detected_version": version,
        "schema": schema,
        "version_matches": version == expected_version,
    }


@app.command("paths")
def show_paths(
    config_file: Path = typer.Option(
        None,
        "--config-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Configurazione TOML/YAML alternativa da usare al posto delle variabili d'ambiente.",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Ignora la cache e ricostruisce le impostazioni partendo da variabili d'ambiente o file.",
    ),
    check_registry: bool = typer.Option(
        True,
        "--check-registry/--no-check-registry",
        help="Valida la versione del registry rispetto al riferimento di produzione (0.2.0).",
    ),
) -> None:
    """Stampa i percorsi risolti dal resolver ``ResourcePaths`` in formato JSON."""

    settings = get_settings(refresh=refresh, config_file=config_file)
    payload = {
        "config_source": str(config_file) if config_file else "environment",
        "paths": _inventory(settings),
    }

    if check_registry:
        payload["registry"] = _check_registry_version(settings.registry_path, expected_version="0.2.0")

    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("lockfile")
def generate_lockfile(
    output: Path = typer.Option(
        Path("outputs/resource-paths.json"),
        "--output",
        dir_okay=False,
        writable=True,
        help="File JSON da produrre con l'inventario dei percorsi.",
    ),
    config_file: Path = typer.Option(
        None,
        "--config-file",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Configurazione TOML/YAML alternativa da usare per l'inventario.",
    ),
) -> None:
    """Persisti un lockfile dei percorsi per tracciarli nelle pipeline CI/CD."""

    settings = get_settings(refresh=True, config_file=config_file)
    payload = {
        "paths": _inventory(settings),
        "registry": _check_registry_version(settings.registry_path, expected_version="0.2.0"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"output": str(output)}, indent=2, ensure_ascii=False))
