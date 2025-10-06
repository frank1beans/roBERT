"""Main entry point for model training commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import typer

from ..training.hier_trainer import (
    HierTrainingArgs,
    build_arg_parser as build_hier_parser,
    train_hier_model,
)
from ..training.label_trainer import (
    LabelTrainingArgs,
    build_arg_parser as build_label_parser,
    train_label_model,
)
from ..training.span_trainer import (
    SpanTrainingArgs,
    build_arg_parser as build_span_parser,
    train_span_model,
)
from ..training.price_trainer import (
    PriceTrainingArgs,
    build_arg_parser as build_price_parser,
    train_price_model,
)

app = typer.Typer(help="Model training utilities", add_completion=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train BIM NLP models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("label", help="Train the label embedding model")
    subparsers.add_parser("hier", help="Train the hierarchical masked model")
    subparsers.add_parser("span", help="Train the span-based property extractor")
    subparsers.add_parser("price", help="Train the price regression model")

    args, remaining = parser.parse_known_args(argv)
    if args.command == "label":
        label_ns = build_label_parser().parse_args(remaining)
        train_label_model(LabelTrainingArgs(**vars(label_ns)))
    elif args.command == "hier":
        hier_ns = build_hier_parser().parse_args(remaining)
        train_hier_model(HierTrainingArgs(**vars(hier_ns)))
    elif args.command == "span":
        span_ns = build_span_parser().parse_args(remaining)
        train_span_model(SpanTrainingArgs(**vars(span_ns)))
    elif args.command == "price":
        price_ns = build_price_parser().parse_args(remaining)
        train_price_model(PriceTrainingArgs(**vars(price_ns)))
    else:
        parser.print_help()
        sys.exit(1)


@app.command("price")
def price_command(
    train_data: Path = typer.Option(..., "--train-data", help="Path to training JSONL file"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save model"),
    backbone_name: str = typer.Option("dbmdz/bert-base-italian-xxl-cased", "--backbone", help="BERT model name"),
    use_properties: bool = typer.Option(False, "--use-properties/--no-use-properties", help="Use property features"),
    property_map: Optional[Path] = typer.Option(None, "--property-map", help="JSON mapping property names to IDs"),
    property_unit_map: Optional[Path] = typer.Option(None, "--property-unit-map", help="JSON mapping properties to units"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", help="Learning rate"),
    val_split: float = typer.Option(0.1, "--val-split", help="Validation split ratio"),
    max_length: int = typer.Option(512, "--max-length", help="Maximum token length"),
    property_dim: int = typer.Option(64, "--property-dim", help="Property embedding dimension"),
    unit_dim: int = typer.Option(32, "--unit-dim", help="Unit embedding dimension"),
    hidden_dims: str = typer.Option("512,256", "--hidden-dims", help="Hidden layer dimensions"),
    dropout: float = typer.Option(0.1, "--dropout", help="Dropout rate"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Train the price regression model."""
    args = PriceTrainingArgs(
        backbone_name=backbone_name,
        train_data=str(train_data),
        output_dir=str(output_dir),
        property_map=str(property_map) if property_map else None,
        property_unit_map=str(property_unit_map) if property_unit_map else None,
        use_properties=use_properties,
        property_dim=property_dim,
        unit_dim=unit_dim,
        hidden_dims=hidden_dims,
        val_split=val_split,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_ratio=0.1,
        dropout=dropout,
        seed=seed,
    )
    train_price_model(args)


@app.command("span")
def span_command(
    train_data: Path = typer.Option(..., "--train-data", help="Path to training JSONL file"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to save model"),
    epochs: int = typer.Option(10, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", help="Learning rate"),
    val_data: Optional[Path] = typer.Option(None, "--val-data", help="Path to validation JSONL"),
) -> None:
    """Train the span-based property extractor."""
    # Use argparse parser for now
    argv = ["--train-data", str(train_data), "--output-dir", str(output_dir)]
    if val_data:
        argv.extend(["--val-data", str(val_data)])
    argv.extend(["--epochs", str(epochs), "--batch-size", str(batch_size)])
    argv.extend(["--learning-rate", str(learning_rate)])

    span_ns = build_span_parser().parse_args(argv)
    train_span_model(SpanTrainingArgs(**vars(span_ns)))


if __name__ == "__main__":
    main()
