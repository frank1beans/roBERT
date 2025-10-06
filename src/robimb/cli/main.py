from pathlib import Path

import json
import typer

from .._version import __version__
from ..utils.sampling import sample_one_record_per_category
from .config import app as config_app
from .convert import convert_command
from .evaluate import evaluate_command
from .extract import app as extract_app
from .predict import app as predict_app
from .pack import pack_command
from .prepare import app as prepare_app
from .train import app as train_app


__all__ = ["app", "run"]


app = typer.Typer(help="Production-ready BIM NLP pipeline utilities", add_completion=False)


@app.callback(invoke_without_command=True)
def version_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show robimb version and exit", is_eager=True),
) -> None:
    """Handle global options before any sub-command executes."""

    if version:
        typer.echo(f"robimb {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


app.command("convert", help="Prepara dataset, label map e maschere ontologiche.")(convert_command)
app.add_typer(extract_app, name="extract")
app.add_typer(predict_app, name="predict")
app.add_typer(config_app, name="config")
app.add_typer(prepare_app, name="prepare")
app.add_typer(train_app, name="train")
app.command("evaluate", help="Valuta un modello esportato su un dataset etichettato.")(evaluate_command)
app.command("pack", help="Impacchetta le cartelle delle proprietà in registry/extractors.")(pack_command)


@app.command("sample-categories")
def sample_categories_command(
    dataset: Path = typer.Option(
        ..., "--dataset", exists=True, dir_okay=False, help="JSONL di partenza con le descrizioni"
    ),
    output: Path = typer.Option(
        ..., "--output", dir_okay=False, help="File JSONL di destinazione con una voce per categoria"
    ),
    category_field: str = typer.Option(
        "cat",
        "--category-field",
        help="Nome del campo che identifica la categoria (default: 'cat')",
    ),
) -> None:
    """Estrai la prima voce disponibile per ciascuna categoria nel dataset."""

    records = sample_one_record_per_category(dataset, category_field=category_field)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "num_records": len(records),
                "category_field": category_field,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run() -> None:
    """Entry point compatible with ``python -m robimb.cli.main`` and console scripts."""

    from typer.main import get_command

    cli = get_command(app)
    cli()
