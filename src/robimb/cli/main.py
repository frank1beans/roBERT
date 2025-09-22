"""Unified Typer-based command line interface for robimb."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import typer

from .._version import __version__

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


@app.command("convert")
def convert_command(
    train_file: Path = typer.Option(..., "--train-file", exists=True, dir_okay=False, help="Path to raw training JSONL"),
    val_file: Optional[Path] = typer.Option(None, "--val-file", exists=True, dir_okay=False, help="Optional validation JSONL"),
    ontology: Optional[Path] = typer.Option(None, "--ontology", exists=True, dir_okay=False, help="Ontology JSON mapping"),
    label_maps: Path = typer.Option(..., "--label-maps", dir_okay=False, help="Output path for generated label maps"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory that will receive processed data"),
    done_uids: Optional[Path] = typer.Option(None, "--done-uids", exists=True, dir_okay=False, help="Text file listing UIDs to skip"),
    val_split: float = typer.Option(0.2, "--val-split", min=0.0, max=0.5, help="Validation ratio when --val-file is missing"),
    random_state: int = typer.Option(42, "--random-state", help="Random seed used for deterministic splits"),
    make_mlm_corpus: bool = typer.Option(False, "--make-mlm-corpus", help="Produce MLM/TAPT corpus"),
    mlm_output: Optional[Path] = typer.Option(None, "--mlm-output", help="Corpus output path when --make-mlm-corpus is set"),
    extra_mlm: Optional[List[Path]] = typer.Option(
        None,
        "--extra-mlm",
        help="Additional JSONL files contributing text to the MLM corpus",
        metavar="PATH",
        show_default=False,
    ),
) -> None:
    """Prepare datasets, label maps and ontology masks."""

    from . import convert as convert_cli

    config = convert_cli.ConversionConfig(
        train_file=train_file,
        val_file=val_file,
        ontology=ontology,
        label_maps=label_maps,
        out_dir=out_dir,
        done_uids=done_uids,
        val_split=val_split,
        random_state=random_state,
        make_mlm_corpus=make_mlm_corpus,
        mlm_output=mlm_output,
        extra_mlm=tuple(extra_mlm or ()),
    )
    artifacts = convert_cli.run_conversion(config)
    typer.echo(json.dumps(artifacts.as_dict(), indent=2, ensure_ascii=False))


train_app = typer.Typer(
    help="Fine-tune BIM classifiers (label-embedding or hierarchical).",
    add_completion=False,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)


@train_app.command("label")
def train_label(ctx: typer.Context) -> None:
    """Proxy to the advanced label model trainer (argparse based)."""

    from . import train as train_cli

    argv: Sequence[str] = ["label", *ctx.args] if ctx.args else ["label", "--help"]
    try:
        train_cli.main(list(argv))
    except SystemExit as exc:  # pragma: no cover - align Typer exit codes
        raise typer.Exit(exc.code)


@train_app.command("hier")
def train_hier(ctx: typer.Context) -> None:
    """Proxy to the hierarchical masked model trainer (argparse based)."""

    from . import train as train_cli

    argv: Sequence[str] = ["hier", *ctx.args] if ctx.args else ["hier", "--help"]
    try:
        train_cli.main(list(argv))
    except SystemExit as exc:  # pragma: no cover - align Typer exit codes
        raise typer.Exit(exc.code)


app.add_typer(train_app, name="train")


@app.command("validate")
def validate_command(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=True, dir_okay=True),
    test_file: Path = typer.Option(..., "--test-file", exists=True, dir_okay=False),
    label_maps: Path = typer.Option(..., "--label-maps", exists=True, dir_okay=False),
    ontology: Optional[Path] = typer.Option(None, "--ontology", dir_okay=False, help="Optional ontology for reporting"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for evaluation"),
    max_length: int = typer.Option(256, "--max-length", help="Tokenizer max length"),
    output: Optional[Path] = typer.Option(None, "--output", help="Path where metrics JSON should be saved"),
    predictions: Optional[Path] = typer.Option(None, "--predictions", help="Optional JSONL with detailed predictions"),
) -> None:
    """Evaluate an exported model on labelled data."""

    from . import validate as validate_cli

    metrics = validate_cli.validate_model(
        validate_cli.ValidationConfig(
            model_dir=model_dir,
            test_file=test_file,
            label_maps=label_maps,
            ontology=ontology,
            batch_size=batch_size,
            max_length=max_length,
            output=output,
            predictions=predictions,
        )
    )
    if output is None:
        typer.echo(json.dumps(metrics, indent=2, ensure_ascii=False))


@app.command(
    "tapt",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def tapt_command(ctx: typer.Context) -> None:
    """Run domain TAPT/MLM pre-training (delegates to the power-user script)."""

    from ..training import tapt_mlm

    argv: Sequence[str] = list(ctx.args)
    if not argv:
        argv = ["--help"]
    try:
        tapt_mlm.main(list(argv))
    except SystemExit as exc:  # pragma: no cover
        raise typer.Exit(exc.code)


def run() -> None:
    """Entry point compatible with ``python -m robimb.cli.main`` and console scripts."""

    from typer.main import get_command

    cli = get_command(app)
    cli()
