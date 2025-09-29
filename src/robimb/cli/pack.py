"""Utilities to pack properties folders into registry/extractor bundles."""
from __future__ import annotations

from pathlib import Path

import typer

from ..registry import pack_folders_to_monolith

__all__ = ["pack_command"]


def pack_command(
    properties_root: Path = typer.Option(..., "--properties-root", exists=True, file_okay=False),
    out_registry: Path = typer.Option(..., "--out-registry"),
    out_extractors: Path = typer.Option(..., "--out-extractors"),
) -> None:
    """Pack a properties folder tree back into registry/extractor JSON bundles."""

    pack_folders_to_monolith(properties_root, out_registry, out_extractors)
    typer.echo(
        (
            "Pack creato.\n"
            f"  registry:   {out_registry}\n"
            f"  extractors: {out_extractors}"
        )
    )
