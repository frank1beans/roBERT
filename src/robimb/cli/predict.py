"""CLI entrypoints for standalone prediction tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer
from tqdm import tqdm

from ..inference.category import CategoryInference
from ..inference.price_inference import PriceInference
from ..inference.span_inference import SpanInference

__all__ = ["app"]


app = typer.Typer(help="Prediction utilities", add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_single_source(text: Optional[str], input_path: Optional[Path]) -> None:
    if (text is None) == (input_path is None):
        raise typer.BadParameter("Provide exactly one between --text and --input")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            payload = line.strip()
            if not payload:
                continue
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(
                    f"Invalid JSON at line {line_no} of {path}: {exc}"
                ) from exc
    return records


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Optional[Path]) -> None:
    if path:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    out = typer.get_text_stream("stdout")
    for record in records:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_numeric_properties(data: Any) -> Dict[str, float]:
    if not isinstance(data, dict):
        return {}
    result: Dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            result[key] = float(value)
        elif isinstance(value, dict):
            candidate = value.get("value")
            if isinstance(candidate, (int, float)):
                result[key] = float(candidate)
    return result


def _maybe_iter(records: List[Any], description: str, enabled: bool):
    if enabled and len(records) > 0:
        return enumerate(tqdm(records, desc=description, unit="doc"))
    return enumerate(records)


def _pretty_print(payload: Dict[str, Any], pretty: bool) -> None:
    typer.echo(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("category")
def predict_category_command(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=True, help="Directory or HF repo containing the classifier"),
    text: Optional[str] = typer.Option(None, "--text", help="Single text to classify"),
    input_path: Optional[Path] = typer.Option(None, "--input", exists=True, dir_okay=False, help="JSONL file with records to classify"),
    output_path: Optional[Path] = typer.Option(None, "--output", dir_okay=False, help="Where to write predictions (JSONL). Defaults to stdout"),
    text_field: str = typer.Option("text", "--text-field", help="Field containing the text inside JSONL records"),
    output_field: str = typer.Option("_category_prediction", "--output-field", help="Field added to each record with prediction payload"),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Number of top categories to keep"),
    backend: str = typer.Option("auto", "--backend", case_sensitive=False, help="Backend selector: auto, label-embed, sequence-classifier"),
    label_map_path: Optional[Path] = typer.Option(None, "--label-map", exists=True, dir_okay=False, help="Optional id2label JSON mapping for sequence classifiers"),
    device: Optional[str] = typer.Option(None, "--device", help="Force device (cpu/cuda)"),
    include_scores: bool = typer.Option(False, "--include-scores", help="Include logits/probabilities in output"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", envvar="HF_TOKEN", help="Hugging Face token for private repos"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON for single-text mode"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar when processing files"),
) -> None:
    """Predict BIM categories for a single text or a JSONL dataset."""

    _validate_single_source(text, input_path)

    predictor = CategoryInference(
        model_dir=model_dir,
        backend=backend,
        device=device,
        hf_token=hf_token,
        label_map_path=label_map_path,
    )

    if text is not None:
        prediction = predictor.predict(text, top_k=top_k, return_scores=include_scores)
        _pretty_print(prediction, pretty)
        return

    records = _load_jsonl(input_path)
    typer.echo(f"Loaded {len(records)} records from {input_path}", err=True)

    for idx, record in _maybe_iter(records, "Predicting categories", progress):
        raw_text = _normalise_text(record.get(text_field))
        if raw_text.strip():
            record[output_field] = predictor.predict(raw_text, top_k=top_k, return_scores=include_scores)
        else:
            record[output_field] = None

    _write_jsonl(records, output_path)
    destination = output_path or "stdout"
    typer.echo(f"Category predictions written to {destination}", err=True)


@app.command("properties")
def predict_properties_command(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=False, help="Directory containing the span extraction model"),
    text: Optional[str] = typer.Option(None, "--text", help="Single text to extract properties from"),
    input_path: Optional[Path] = typer.Option(None, "--input", exists=True, dir_okay=False, help="JSONL dataset with product descriptions"),
    output_path: Optional[Path] = typer.Option(None, "--output", dir_okay=False, help="Destination JSONL (defaults to stdout)"),
    text_field: str = typer.Option("text", "--text-field", help="Field with the product description"),
    output_field: str = typer.Option("_property_predictions", "--output-field", help="Field added to records with extracted properties"),
    property_ids: List[str] = typer.Option([], "--property-id", "-p", help="Limit extraction to specific property IDs"),
    apply_parsers: bool = typer.Option(True, "--apply-parsers/--raw-spans", help="Apply domain parsers to spans"),
    batch_size: int = typer.Option(8, "--batch-size", min=1, help="Batch size for dataset mode"),
    device: Optional[str] = typer.Option(None, "--device", help="Force device (cpu/cuda)"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON for single-text mode"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar when processing files"),
) -> None:
    """Extract properties using the trained span extractor."""

    _validate_single_source(text, input_path)

    inferencer = SpanInference(Path(model_dir), device=device)
    prop_ids = property_ids or None

    if text is not None:
        prediction = inferencer.extract_properties(
            text,
            property_ids=prop_ids,
            apply_parsers=apply_parsers,
        )
        _pretty_print(prediction, pretty)
        return

    records = _load_jsonl(input_path)
    typer.echo(f"Loaded {len(records)} records from {input_path}", err=True)

    texts: List[str] = []
    valid_indices: List[int] = []

    for idx, record in enumerate(records):
        raw_text = _normalise_text(record.get(text_field))
        if raw_text.strip():
            texts.append(raw_text)
            valid_indices.append(idx)
        else:
            record[output_field] = {}

    if not texts:
        _write_jsonl(records, output_path)
        destination = output_path or "stdout"
        typer.echo("No valid texts found. Nothing to extract.", err=True)
        typer.echo(f"Property predictions written to {destination}", err=True)
        return

    iterator = range(0, len(texts), batch_size)
    if progress and len(texts) > batch_size:
        iterator = tqdm(iterator, desc="Extracting properties", unit="batch")

    for start in iterator:
        end = min(start + batch_size, len(texts))
        batch_predictions = inferencer.extract_batch(
            texts[start:end],
            property_ids=prop_ids,
            apply_parsers=apply_parsers,
        )
        for offset, prediction in enumerate(batch_predictions):
            records[valid_indices[start + offset]][output_field] = prediction

    _write_jsonl(records, output_path)
    destination = output_path or "stdout"
    typer.echo(f"Property predictions written to {destination}", err=True)


@app.command("price")
def predict_price_command(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=False, help="Directory containing the trained price regressor"),
    text: Optional[str] = typer.Option(None, "--text", help="Single text to score"),
    properties_json: Optional[str] = typer.Option(None, "--properties-json", help="Inline JSON with properties for single-text mode"),
    price_unit: str = typer.Option("cad", "--price-unit", help="Price unit for single-text predictions"),
    input_path: Optional[Path] = typer.Option(None, "--input", exists=True, dir_okay=False, help="JSONL dataset to score"),
    output_path: Optional[Path] = typer.Option(None, "--output", dir_okay=False, help="Destination JSONL (defaults to stdout)"),
    text_field: str = typer.Option("text", "--text-field", help="Field containing the text"),
    properties_field: str = typer.Option("properties", "--properties-field", help="Field containing extracted properties"),
    price_unit_field: str = typer.Option("price_unit", "--price-unit-field", help="Field containing the price unit"),
    default_price_unit: str = typer.Option("cad", "--default-price-unit", help="Fallback unit when missing"),
    output_field: str = typer.Option("_price_prediction", "--output-field", help="Field added to records with price prediction"),
    use_properties: bool = typer.Option(True, "--use-properties/--no-properties", help="Use properties when available"),
    device: Optional[str] = typer.Option(None, "--device", help="Force device (cpu/cuda)"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty-print JSON for single-text mode"),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Show progress bar when processing files"),
) -> None:
    """Predict prices using the regression model."""

    _validate_single_source(text, input_path)

    inferencer = PriceInference(Path(model_dir), device=device)

    if text is not None:
        properties = None
        if properties_json:
            try:
                properties_raw = json.loads(properties_json)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(f"Invalid JSON passed to --properties-json: {exc}") from exc
            properties = _extract_numeric_properties(properties_raw) if use_properties else None
        prediction = inferencer.predict(text=text, properties=properties, price_unit=price_unit)
        _pretty_print(prediction, pretty)
        return

    records = _load_jsonl(input_path)
    typer.echo(f"Loaded {len(records)} records from {input_path}", err=True)

    for idx, record in _maybe_iter(records, "Predicting prices", progress):
        raw_text = _normalise_text(record.get(text_field))
        if not raw_text.strip():
            record[output_field] = None
            continue

        properties = None
        if use_properties:
            properties_raw = record.get(properties_field)
            properties = _extract_numeric_properties(properties_raw)
            if not properties:
                properties = None

        unit_value = record.get(price_unit_field, default_price_unit) or default_price_unit
        prediction = inferencer.predict(
            text=raw_text,
            properties=properties,
            price_unit=str(unit_value),
        )
        record[output_field] = prediction

    _write_jsonl(records, output_path)
    destination = output_path or "stdout"
    typer.echo(f"Price predictions written to {destination}", err=True)
