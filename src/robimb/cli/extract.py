"""Utilities to unpack knowledge packs into property folders."""
from __future__ import annotations

from pathlib import Path

import typer

from ..props.unpack import convert_monolith_to_folders

__all__ = ["extract_command"]


def extract_command(
    in_registry: Path = typer.Option(..., "--registry", exists=True, dir_okay=False),
    in_extractors: Path = typer.Option(..., "--extractors", exists=True, dir_okay=False),
    out_dir: Path = typer.Option(..., "--out-dir"),
) -> None:
    """Expand registry/extractors bundle into the properties folder structure."""

    convert_monolith_to_folders(in_registry, in_extractors, out_dir)
    typer.echo(f"Pack estratto in: {out_dir}")
