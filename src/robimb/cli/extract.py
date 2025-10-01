"""CLI entrypoints for the property extraction pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..extraction.fuse import Fuser, FusePolicy
from ..extraction.orchestrator import Orchestrator, OrchestratorConfig
from ..extraction.qa_llm import HttpLLM, MockLLM, QALLMConfig
from ..extraction.schema_registry import load_registry
from ..utils.logging import configure_json_logger, flush_handlers, generate_trace_id, log_event

__all__ = ["app"]

app = typer.Typer(help="Property extraction utilities", add_completion=False)


@app.command("properties")
def extract_properties(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False, help="JSONL with input records"),
    output_path: Path = typer.Option(..., "--output", dir_okay=False, help="Destination JSONL for extracted properties"),
    pack_path: Path = typer.Option(
        Path("pack/current"),
        "--pack",
        exists=True,
        file_okay=False,
        help="Knowledge pack directory providing prompts and lexicons",
    ),
    schema_registry_path: Path = typer.Option(
        Path("data/properties/registry.json"),
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
    log_file: Optional[Path] = typer.Option(None, "--log-file", dir_okay=False, help="Optional JSONL log path"),
    fail_fast: bool = typer.Option(False, "--fail-fast/--no-fail-fast", help="Abort on validation errors"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without running the pipeline"),
) -> None:
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

    llm_cfg = QALLMConfig(
        endpoint=llm_endpoint,
        model=llm_model,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
    )
    llm = HttpLLM(llm_cfg) if llm_endpoint else MockLLM()

    orchestrator_cfg = OrchestratorConfig(
        source_priority=["parser", "matcher", "qa_llm"],
        enable_matcher=True,
        enable_llm=bool(llm_endpoint),
        registry_path=str(schema_registry_path),
    )
    fuse = Fuser(policy=FusePolicy.VALIDATE_THEN_MAX_CONF, source_priority=orchestrator_cfg.source_priority)
    orchestrator = Orchestrator(fuse=fuse, llm=llm, cfg=orchestrator_cfg)

    processed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if category_filter and record.get("categoria") != category_filter:
                continue
            result = orchestrator.extract_document(record)
            json.dump(result, dst, ensure_ascii=False)
            dst.write("\n")
            processed += 1

    typer.echo(json.dumps({"status": "completed", "documents": processed}, ensure_ascii=False))
    log_event(
        logger,
        "extract.properties.completed",
        trace_id=trace_id,
        documents=processed,
    )
    flush_handlers(logger)


@app.command("schemas")
def schemas_command(
    registry_path: Path = typer.Option(
        Path("data/properties/registry.json"),
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
