"""CLI entrypoints for the property extraction pipeline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import typer
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from ..config import get_settings
from ..extraction.fuse import Fuser, FusePolicy
from ..extraction.orchestrator import Orchestrator, OrchestratorConfig
from ..extraction.property_qa import (
    QAExample,
    answer_properties,
    build_properties_for_category,
    predict_with_encoder,
    train_property_qa,
)
from ..extraction.qa_llm import AsyncHttpLLM, HttpLLM, MockLLM, QALLMConfig
from ..extraction.schema_registry import load_registry
from ..utils.logging import configure_json_logger, flush_handlers, generate_trace_id, log_event

__all__ = ["app"]

app = typer.Typer(help="Property extraction utilities", add_completion=False)

_SETTINGS = get_settings()


async def _extract_async(
    records: list,
    llm_endpoint: Optional[str],
    llm_model: Optional[str],
    llm_timeout: float,
    llm_max_retries: int,
    schema_registry_path: Path,
    max_workers: int,
    fail_fast: bool,
    logger,
    trace_id: str,
    *,
    use_qa: bool,
    fusion_mode: str,
    qa_null_threshold: float,
    qa_confident_threshold: float,
) -> list:
    """Async extraction with concurrent processing."""
    from ..extraction.orchestrator_async import AsyncOrchestrator

    llm_cfg = QALLMConfig(
        endpoint=llm_endpoint,
        model=llm_model,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
    )

    orchestrator_cfg = OrchestratorConfig(
        source_priority=["parser", "matcher", "qa_llm"],
        enable_matcher=True,
        enable_llm=True,
        registry_path=str(schema_registry_path),
        use_qa=use_qa,
        fusion_mode=fusion_mode,
        qa_null_threshold=qa_null_threshold,
        qa_confident_threshold=qa_confident_threshold,
    )

    results = []

    if llm_endpoint:
        async with AsyncHttpLLM(llm_cfg) as llm:
            fuse = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=orchestrator_cfg.source_priority)
            orchestrator = AsyncOrchestrator(fuse=fuse, llm=llm, cfg=orchestrator_cfg)

            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_workers)

            async def process_with_semaphore(idx: int, record: Dict[str, Any]):
                async with semaphore:
                    try:
                        result = await orchestrator.extract_document(record)
                        return (idx, result)
                    except Exception as exc:
                        if fail_fast:
                            raise
                        typer.echo(f"Error processing record {idx}: {exc}", err=True)
                        log_event(logger, "extract.properties.error", trace_id=trace_id, record_idx=idx, error=str(exc))
                        return (idx, None)

            # Process all records with progress bar
            tasks = [process_with_semaphore(idx, record) for idx, record in enumerate(records)]
            with tqdm(total=len(records), desc="Extracting properties", unit="doc") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)
    else:
        # Mock LLM - use sync processing
        llm = MockLLM()
        fuse = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=orchestrator_cfg.source_priority)
        orchestrator = Orchestrator(fuse=fuse, llm=llm, cfg=orchestrator_cfg)

        with tqdm(total=len(records), desc="Extracting properties", unit="doc") as pbar:
            for idx, record in enumerate(records):
                try:
                    result = orchestrator.extract_document(record)
                    results.append((idx, result))
                except Exception as exc:
                    if fail_fast:
                        raise
                    typer.echo(f"Error processing record {idx}: {exc}", err=True)
                    log_event(logger, "extract.properties.error", trace_id=trace_id, record_idx=idx, error=str(exc))
                    results.append((idx, None))
                pbar.update(1)

    return results


@app.command("properties")
def extract_properties(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, help="JSONL with input records"),
    output_path: Path = typer.Option(..., "--output", dir_okay=False, help="Destination JSONL for extracted properties"),
    pack_path: Path = typer.Option(
        (_SETTINGS.pack_dir / "current"),
        "--pack",
        exists=True,
        file_okay=False,
        help="Knowledge pack directory providing prompts and lexicons",
    ),
    schema_registry_path: Path = typer.Option(
        _SETTINGS.registry_path,
        "--schema",
        exists=True,
        dir_okay=False,
        help="Path to the schema registry JSON file",
    ),
    llm_endpoint: Optional[str] = typer.Option(None, "--llm-endpoint", help="LLM endpoint for QA extraction"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model identifier"),
    llm_timeout: float = typer.Option(30.0, "--llm-timeout", help="Timeout for LLM calls in seconds"),
    llm_max_retries: int = typer.Option(2, "--llm-max-retries", min=0, help="Maximum retries for LLM calls"),
    category_filter: Optional[str] = typer.Option(
        None, "--category-filter", help="Limit extraction to a single category (ID or name)"
    ),
    confidence_threshold: float = typer.Option(
        0.6, "--confidence-threshold", min=0.0, max=1.0, help="Minimum confidence accepted in the output"
    ),
    batch_size: int = typer.Option(16, "--batch-size", min=1, help="Number of records processed per batch"),
    max_workers: int = typer.Option(4, "--max-workers", min=1, help="Parallel workers for the pipeline"),
    sample: Optional[int] = typer.Option(None, "--sample", min=1, help="Process only first N records for testing"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", dir_okay=False, help="Optional JSONL log path"),
    fail_fast: bool = typer.Option(False, "--fail-fast/--no-fail-fast", help="Abort on validation errors"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without running the pipeline"),
    use_qa: bool = typer.Option(True, "--use-qa/--no-qa", help="Enable the QA encoder for property spans"),
    qa_model_dir: Optional[Path] = typer.Option(
        None,
        "--qa-model-dir",
        exists=True,
        file_okay=False,
        help="Directory containing the fine-tuned QA model",
    ),
    qa_null_th: float = typer.Option(0.25, "--qa-null-th", min=0.0, max=2.0, help="QA no-answer threshold"),
    fusion: Literal["rules_only", "qa_only", "fuse"] = typer.Option(
        "fuse",
        "--fusion",
        case_sensitive=False,
        help="Fusion strategy between rules and QA",
    ),
    qa_max_length: int = typer.Option(384, "--qa-max-length", min=32, help="Maximum QA sequence length"),
    qa_doc_stride: int = typer.Option(128, "--qa-doc-stride", min=16, help="QA sliding window stride"),
    qa_max_answer_length: int = typer.Option(64, "--qa-max-answer-length", min=1, help="Maximum QA answer length"),
) -> None:
    qa_confident_threshold = 0.60
    config = {
        "input": str(input_path),
        "output": str(output_path),
        "pack": str(pack_path),
        "schema": str(schema_registry_path),
        "llm_endpoint": llm_endpoint,
        "llm_model": llm_model,
        "llm_timeout": llm_timeout,
        "llm_max_retries": llm_max_retries,
        "category_filter": category_filter,
        "confidence_threshold": confidence_threshold,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "log_file": str(log_file) if log_file else None,
        "fail_fast": fail_fast,
        "dry_run": dry_run,
        "use_qa": use_qa,
        "qa_model_dir": str(qa_model_dir) if qa_model_dir else None,
        "qa_null_th": qa_null_th,
        "fusion": fusion,
        "qa_max_length": qa_max_length,
        "qa_doc_stride": qa_doc_stride,
        "qa_max_answer_length": qa_max_answer_length,
        "qa_confident_threshold": qa_confident_threshold,
    }
    logger = configure_json_logger(log_file)
    trace_id = generate_trace_id()
    log_event(
        logger,
        "extract.properties.start",
        trace_id=trace_id,
        input=str(input_path),
        output=str(output_path),
        schema=str(schema_registry_path),
        dry_run=dry_run,
        batch_size=batch_size,
    )
    flush_handlers(logger)
    if dry_run:
        typer.echo(json.dumps({"status": "dry_run", "config": config}, indent=2, ensure_ascii=False))
        log_event(logger, "extract.properties.dry_run", trace_id=trace_id, status="ack")
        flush_handlers(logger)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all records first
    records = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if category_filter and record.get("categoria") != category_filter:
                continue
            records.append(record)
            if sample and len(records) >= sample:
                break

    fusion_mode = fusion.lower()

    if use_qa:
        if qa_model_dir is None:
            raise typer.BadParameter("--qa-model-dir è obbligatorio quando si abilita --use-qa")
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_dir)
        qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_dir)
        prompt_cache: Dict[str, List[tuple[str, str]]] = {}
        for record in records:
            category_id = (
                record.get("categoria")
                or record.get("category")
                or record.get("cat")
            )
            if not category_id:
                continue
            if category_id not in prompt_cache:
                try:
                    prompt_cache[category_id] = build_properties_for_category(category_id, schema_registry_path)
                except ValueError:
                    prompt_cache[category_id] = []
            prompts = prompt_cache.get(category_id) or []
            if not prompts:
                continue
            text_value = record.get("text") or ""
            if not text_value:
                continue
            examples = [
                QAExample(
                    qid=f"{category_id}:{prop_id}",
                    context=text_value,
                    question=prompt,
                    answers=[],
                    answer_starts=[],
                    property_id=prop_id,
                )
                for prop_id, prompt in prompts
            ]
            predictions = predict_with_encoder(
                qa_model,
                qa_tokenizer,
                examples,
                null_threshold=qa_null_th,
                max_length=qa_max_length,
                doc_stride=qa_doc_stride,
                max_answer_length=qa_max_answer_length,
                batch_size=8,
            )
            if predictions:
                record["_qa_predictions"] = predictions

    # Run async processing
    results = asyncio.run(
        _extract_async(
            records=records,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
            llm_timeout=llm_timeout,
            llm_max_retries=llm_max_retries,
            schema_registry_path=schema_registry_path,
            max_workers=max_workers,
            fail_fast=fail_fast,
            logger=logger,
            trace_id=trace_id,
            use_qa=use_qa,
            fusion_mode=fusion_mode,
            qa_null_threshold=qa_null_th,
            qa_confident_threshold=qa_confident_threshold,
        )
    )

    # Sort results by original order and write to output
    results.sort(key=lambda x: x[0])
    processed = len([r for r in results if r[1] is not None])
    with output_path.open("w", encoding="utf-8") as dst:
        for _, result in results:
            if result is not None:
                json.dump(result, dst, ensure_ascii=False)
                dst.write("\n")

    typer.echo(json.dumps({"status": "completed", "documents": processed}, ensure_ascii=False))
    log_event(
        logger,
        "extract.properties.completed",
        trace_id=trace_id,
        documents=processed,
    )
    flush_handlers(logger)


@app.command("train-qa")
def train_qa_encoder(
    model: str = typer.Option(..., "--model", help="Base encoder name or local path"),
    train_jsonl: Path = typer.Option(..., "--train-jsonl", exists=True, dir_okay=False, help="Training QA JSONL"),
    eval_jsonl: Optional[Path] = typer.Option(None, "--eval-jsonl", exists=True, dir_okay=False, help="Optional evaluation QA JSONL"),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory where the fine-tuned model will be stored"),
    epochs: int = typer.Option(3, "--epochs", min=1, help="Number of fine-tuning epochs"),
    batch: int = typer.Option(8, "--batch", min=1, help="Per-device batch size"),
    lr: float = typer.Option(5e-5, "--lr", min=1e-6, help="Learning rate"),
    max_length: int = typer.Option(384, "--max-length", min=32, help="Maximum sequence length"),
    doc_stride: int = typer.Option(128, "--doc-stride", min=16, help="Sliding window stride"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Fine-tune the extractive QA encoder for property spans."""

    train_property_qa(
        model_name=model,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch,
        learning_rate=lr,
        max_length=max_length,
        doc_stride=doc_stride,
        seed=seed,
    )


@app.command("predict-qa")
def predict_qa_spans(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=False, help="Directory containing the fine-tuned QA model"),
    text: str = typer.Option(..., "--text", help="Text to analyse"),
    category: str = typer.Option(..., "--category", help="Category identifier"),
    registry: Path = typer.Option(..., "--registry", exists=True, dir_okay=False, help="Schema registry path"),
    null_th: float = typer.Option(0.25, "--null-th", min=0.0, max=2.0, help="No-answer threshold"),
    max_length: int = typer.Option(384, "--max-length", min=32, help="Maximum sequence length"),
    doc_stride: int = typer.Option(128, "--doc-stride", min=16, help="Sliding window stride"),
    max_answer_length: int = typer.Option(64, "--max-answer-length", min=1, help="Maximum answer token length"),
) -> None:
    """Predict property spans for a single text using a QA encoder."""

    predictions = answer_properties(
        model_dir=model_dir,
        text=text,
        category_id=category,
        registry_path=registry,
        null_threshold=null_th,
        max_length=max_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
    )
    typer.echo(json.dumps(predictions, ensure_ascii=False, indent=2))


@app.command("schemas")
def schemas_command(
    registry_path: Path = typer.Option(
        _SETTINGS.registry_path,
        "--registry",
        exists=True,
        dir_okay=False,
        help="Schema registry JSON file",
    ),
    list_only: bool = typer.Option(False, "--list", help="List available categories"),
    show: Optional[str] = typer.Option(None, "--show", help="Show details for the provided category ID or name"),
    print_schema: bool = typer.Option(False, "--print-schema/--no-print-schema", help="Print the JSON schema body"),
) -> None:
    """Inspect the available category schemas."""

    registry = load_registry(registry_path)
    categories = list(registry.list())
    if list_only or not show:
        typer.echo("Categorie disponibili:")
        for category in categories:
            typer.echo(f"- {category.id}: {category.name} (schema: {category.schema_path})")
        if not show:
            return
    category = registry.get(show)
    if category is None:
        raise typer.BadParameter(f"Categoria '{show}' non trovata")
    typer.echo(json.dumps(
        {
            "id": category.id,
            "name": category.name,
            "schema_path": str(category.schema_path),
            "required": list(category.required),
            "properties": [
                {
                    "id": prop.id,
                    "title": prop.title,
                    "type": prop.type,
                    "unit": prop.unit,
                    "required": prop.required,
                    "enum": list(prop.enum) if prop.enum else None,
                    "sources": list(prop.sources) if prop.sources else None,
                }
                for prop in category.properties
            ],
        },
        indent=2,
        ensure_ascii=False,
    ))
    if print_schema:
        schema_text = Path(category.schema_path).read_text(encoding="utf-8")
        typer.echo(schema_text)
