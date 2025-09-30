"""CLI entry point to evaluate regex extractors on a dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..extraction.legacy import run_pack_dataset_evaluation

__all__ = ["pack_test_command"]


def pack_test_command(
    dataset: Path = typer.Option(
        ..., "--dataset", exists=True, dir_okay=False, help="Percorso al dataset JSONL da utilizzare per i test"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False, help="Cartella dove salvare i risultati"),
    pack_path: Optional[Path] = typer.Option(
        None,
        "--pack",
        exists=True,
        help="Directory o file del knowledge pack da utilizzare (default: pack/current)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Numero massimo di record da analizzare (default: tutti)",
    ),
    text_field: str = typer.Option("text", "--text-field", help="Campo del dataset contenente il testo"),
    sample_size: int = typer.Option(
        20,
        "--sample-size",
        min=1,
        max=100,
        help="Numero di esempi da conservare nei file di anteprima",
    ),
) -> None:
    """Esegui i test del pack su un dataset e stampa un riepilogo JSON."""

    summary = run_pack_dataset_evaluation(
        dataset_path=dataset,
        output_dir=output_dir,
        pack_path=pack_path,
        limit=limit,
        text_field=text_field,
        sample_size=sample_size,
    )

    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))

