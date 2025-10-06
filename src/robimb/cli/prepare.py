"""Data preparation command for robimb CLI.

Unified command to prepare datasets for classification, span extraction, and price prediction.
Handles validation against ontology and label maps from resources/data/wbs.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import typer
from tqdm import tqdm

from ..config import get_settings
from ..extraction.property_qa import prepare_qa_dataset
from ..extraction.qa_llm import AsyncHttpLLM, QALLMConfig, build_prompt
from ..extraction.schema_registry import load_registry
from ..utils.dataset_prep import (
    LabelMaps,
    create_or_load_label_maps,
    prepare_classification_dataset,
)
from ..utils.ontology_utils import load_ontology


_SETTINGS = get_settings()

app = typer.Typer(help="Prepare datasets for training")


def normalize_price_unit(raw_unit: str) -> str:
    """Normalize price_unit to standard format."""
    UNIT_MAP = {
        "a corpo": "a_corpo", "corpo": "a_corpo", "forfait": "a_corpo",
        "cadauno": "cad", "cad": "cad", "pezzo": "cad", "pz": "cad", "n": "cad", "nr": "cad",
        "m": "m", "metro": "m", "ml": "m", "metro lineare": "m",
        "m²": "m2", "m2": "m2", "mq": "m2", "metro quadrato": "m2",
        "m³": "m3", "m3": "m3", "mc": "m3", "metro cubo": "m3",
        "kg": "kg", "t": "t", "l": "l", "h": "h", "giorno": "giorno",
        "#n/d": "cad", "": "cad", "null": "cad",
    }
    if not raw_unit or not raw_unit.strip():
        return "cad"
    cleaned = raw_unit.lower().strip()
    return UNIT_MAP.get(cleaned, "cad")


_debug_categories_seen = set()

async def extract_properties_with_llm(
    text: str,
    category_id: str,
    registry_path: Path,
    llm_client: AsyncHttpLLM,
) -> Dict[str, Any]:
    """Extract properties using LLM for a given category."""
    global _debug_categories_seen

    registry = load_registry(registry_path)
    category = registry.get(category_id)

    if not category:
        # Debug: category not found
        if category_id not in _debug_categories_seen:
            print(f"  [Debug] Category not found in registry: {category_id}")
            _debug_categories_seen.add(category_id)
        return {}

    if not category.properties:
        # Debug: no properties defined
        if category_id not in _debug_categories_seen:
            print(f"  [Debug] Category has no properties: {category_id}")
            _debug_categories_seen.add(category_id)
        return {}

    # Debug: first time seeing this category
    if category_id not in _debug_categories_seen:
        print(f"  [Debug] Extracting {len(category.properties)} properties for: {category_id}")
        _debug_categories_seen.add(category_id)

    properties = {}
    first_error = None

    for prop in category.properties:
        question = f"Qual è {prop.title.lower()}?"
        schema = prop.json_schema() if hasattr(prop, 'json_schema') else {"type": "string"}

        try:
            result = await llm_client.ask(text, question, schema)
            if result and "value" in result:
                properties[prop.id] = result["value"]
        except Exception as e:
            # Capture first error to log later
            if first_error is None:
                first_error = (type(e).__name__, str(e)[:200])

    # Log first error encountered (outside loop to avoid spam)
    if first_error and len(properties) == 0:
        print(f"  [ERROR] LLM extraction failed - {first_error[0]}: {first_error[1]}")

    return properties


async def process_records_with_llm(
    records: pd.DataFrame,
    registry_path: Path,
    llm_endpoint: str,
    llm_model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Process records and extract properties using LLM."""
    config = QALLMConfig(
        endpoint=llm_endpoint,
        model=llm_model,
        timeout=30.0,
        max_retries=2,
    )

    properties_list = []

    # Use AsyncHttpLLM as context manager
    async with AsyncHttpLLM(config) as llm:
        for _, row in tqdm(records.iterrows(), total=len(records), desc="Extracting with LLM"):
            text = row.get("text", "")
            category = row.get("cat") or row.get("super", "")

            # Skip if category is NaN, None, or empty
            if not text or category is None or (isinstance(category, float) and pd.isna(category)):
                properties_list.append({})
                continue

            category = str(category).strip()
            if not category:
                properties_list.append({})
                continue

            properties = await extract_properties_with_llm(text, category, registry_path, llm)
            properties_list.append(properties)

    records = records.copy()
    records["properties"] = properties_list

    return records


def _convert_to_qa_format(df: pd.DataFrame, registry_path: Path) -> list:
    """Convert dataframe with extracted properties to QA format for span training.

    For each record with properties, creates QA examples with:
    - context: original text
    - question: property-specific question
    - answers: extracted value with span position in text
    """
    import re

    registry = load_registry(registry_path)
    qa_records = []

    skipped_no_props = 0
    skipped_no_cat = 0
    skipped_no_schema = 0
    skipped_no_span = 0

    for idx, row in df.iterrows():
        text = row.get("text", "")
        properties = row.get("properties", {})
        category = row.get("cat") or row.get("super", "")

        if not text or not properties or not category:
            skipped_no_props += 1
            continue

        # Get category schema
        cat_schema = registry.get(category)
        if not cat_schema:
            skipped_no_schema += 1
            continue

        # For each extracted property, create a QA example
        for prop_id, prop_value in properties.items():
            if not prop_value:
                continue

            # Find the property definition
            prop_def = None
            for p in cat_schema.properties:
                if p.id == prop_id:
                    prop_def = p
                    break

            if not prop_def:
                continue

            # Find span in text (case-insensitive)
            value_str = str(prop_value).strip()
            pattern = re.escape(value_str)
            match = re.search(pattern, text, re.IGNORECASE)

            if match:
                qa_records.append({
                    "id": f"{idx}:{prop_id}",
                    "context": text,
                    "question": f"Qual è {prop_def.title.lower()}?",
                    "answers": [{
                        "text": text[match.start():match.end()],
                        "start": match.start()
                    }],
                    "property_id": prop_id,
                })
            else:
                skipped_no_span += 1

    # Print debug stats
    if skipped_no_props > 0 or skipped_no_schema > 0 or skipped_no_span > 0:
        print(f"  [Debug] Skipped: {skipped_no_props} no props, {skipped_no_schema} no schema, {skipped_no_span} no span")

    return qa_records


def validate_and_prepare_base(
    input_jsonl: Path,
    label_maps: LabelMaps,
    task: str,
) -> pd.DataFrame:
    """Load and validate base dataset."""
    records = []

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # Validate required fields
            if "text" not in record:
                continue

            # Validate categories if present
            if "super" in record or "cat" in record:
                super_name = record.get("super", "")
                cat_name = record.get("cat", "")

                # Skip if categories not in label maps
                if super_name and super_name not in label_maps.super_name_to_id:
                    continue
                if cat_name and cat_name not in label_maps.cat_name_to_id:
                    continue

            # Normalize price_unit if present
            if "price_unit" in record:
                record["price_unit"] = normalize_price_unit(record["price_unit"])

            # Validate price data
            if task == "price":
                if "price" not in record or "price_unit" not in record:
                    continue
                try:
                    float(record["price"])
                except (ValueError, TypeError):
                    continue

            records.append(record)

    return pd.DataFrame(records)


@app.command("classification")
def prepare_classification(
    input: Path = typer.Option(..., "--input", exists=True, help="Input JSONL with text, super, cat"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory for processed datasets"),
    val_input: Optional[Path] = typer.Option(None, "--val-input", exists=True, help="Optional validation JSONL"),
    val_split: float = typer.Option(0.2, "--val-split", help="Validation split if no val_input"),
    ontology: Path = typer.Option(
        "resources/data/wbs/ontology.json",
        "--ontology",
        help="Path to ontology.json",
    ),
    label_maps: Path = typer.Option(
        "resources/data/wbs/label_maps.json",
        "--label-maps",
        help="Path to label_maps.json",
    ),
    registry: Optional[Path] = typer.Option(
        None,
        "--registry",
        help=f"Property registry for extraction (default: {_SETTINGS.registry_path})",
    ),
    llm_endpoint: Optional[str] = typer.Option(
        None,
        "--llm-endpoint",
        help="LLM endpoint for property extraction (e.g., http://localhost:8000/v1/chat/completions)",
    ),
    llm_model: str = typer.Option(
        "gpt-4o-mini",
        "--llm-model",
        help="LLM model to use for extraction",
    ),
) -> None:
    """Prepare dataset for classification (text -> super, cat)."""

    typer.echo("🔧 Preparing classification dataset...")

    # Prepare using existing function
    train_df, val_df, maps = prepare_classification_dataset(
        train_path=input,
        val_path=val_input,
        label_maps_path=label_maps,
        ontology_path=ontology,
        val_split=val_split,
        properties_registry_path=registry,
    )

    # Extract properties with LLM if endpoint provided
    if llm_endpoint and registry:
        typer.echo("🤖 Extracting properties with LLM...")
        registry_path = registry or Path(_SETTINGS.registry_path)
        train_df = asyncio.run(process_records_with_llm(train_df, registry_path, llm_endpoint, llm_model))
        val_df = asyncio.run(process_records_with_llm(val_df, registry_path, llm_endpoint, llm_model))

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train.jsonl"
    val_out = output_dir / "val.jsonl"

    for path, df in [(train_out, train_df), (val_out, val_df)]:
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    typer.echo(json.dumps({
        "train_records": len(train_df),
        "val_records": len(val_df),
        "train_output": str(train_out),
        "val_output": str(val_out),
        "num_super_classes": len(maps.super_name_to_id),
        "num_cat_classes": len(maps.cat_name_to_id),
    }, indent=2))


@app.command("span")
def prepare_span(
    input: Path = typer.Option(..., "--input", exists=True, help="Input JSONL with text, super, cat"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory for span datasets"),
    val_input: Optional[Path] = typer.Option(None, "--val-input", exists=True, help="Optional validation JSONL"),
    val_split: float = typer.Option(0.2, "--val-split", help="Validation split"),
    registry: Path = typer.Option(
        _SETTINGS.registry_path,
        "--registry",
        help="Property registry path",
    ),
    ontology: Path = typer.Option(
        "resources/data/wbs/ontology.json",
        "--ontology",
        help="Path to ontology.json",
    ),
    label_maps: Path = typer.Option(
        "resources/data/wbs/label_maps.json",
        "--label-maps",
        help="Path to label_maps.json",
    ),
    llm_endpoint: Optional[str] = typer.Option(
        None,
        "--llm-endpoint",
        help="LLM endpoint for property extraction",
    ),
    llm_model: str = typer.Option(
        "gpt-4o-mini",
        "--llm-model",
        help="LLM model to use",
    ),
) -> None:
    """Prepare dataset for span extraction (extract properties with LLM, then create QA format)."""

    typer.echo("🔍 Preparing span extraction dataset...")

    if not llm_endpoint:
        typer.echo("⚠️  Warning: No LLM endpoint provided. Span extraction requires property extraction.", fg=typer.colors.YELLOW)
        typer.echo("   Use --llm-endpoint to extract properties automatically.", fg=typer.colors.YELLOW)
        typer.echo("   Attempting to prepare dataset without properties...", fg=typer.colors.YELLOW)

    # Load input data
    from ..utils.sampling import load_jsonl_to_df

    typer.echo("  Loading input data...")
    train_df = load_jsonl_to_df(input)

    # Filter only records with categories in registry
    typer.echo("  Filtering records with categories in registry...")
    registry_obj = load_registry(Path(registry))
    valid_categories = set(registry_obj.categories.keys())

    # Category mapping from dataset names (super/cat from label_maps) to registry keys
    CATEGORY_MAPPING = {
        # Super categories from label_maps -> registry keys
        "Opere da cartongessista": "opere_da_cartongessista",
        "Controsoffitti": "controsoffitti",
        "Opere di pavimentazione": "opere_di_pavimentazione",
        "Opere di rivestimento": "opere_di_rivestimento",
        "Opere da serramentista": "opere_da_serramentista",
        "Apparecchi sanitari e accessori": "apparecchi_sanitari_accessori",
        "Opere da falegname": "opere_da_falegname",
    }

    def map_category(cat_name: str) -> Optional[str]:
        """Map dataset category to registry category."""
        if cat_name in valid_categories:
            return cat_name
        return CATEGORY_MAPPING.get(cat_name)

    def has_valid_category(row):
        # Try super first (more general), then cat
        super_name = row.get("super", "")
        cat_name = row.get("cat", "")

        if super_name and not (isinstance(super_name, float) and pd.isna(super_name)):
            super_name = str(super_name).strip()
            mapped = map_category(super_name)
            if mapped:
                return True

        if cat_name and not (isinstance(cat_name, float) and pd.isna(cat_name)):
            cat_name = str(cat_name).strip()
            mapped = map_category(cat_name)
            if mapped:
                return True

        return False

    def apply_category_mapping(row):
        """Apply category mapping to row - prioritize super over cat."""
        super_name = row.get("super", "")
        cat_name = row.get("cat", "")

        # Try super first
        if super_name and not (isinstance(super_name, float) and pd.isna(super_name)):
            super_name = str(super_name).strip()
            mapped = map_category(super_name)
            if mapped:
                row["cat"] = mapped
                return row

        # Fallback to cat
        if cat_name and not (isinstance(cat_name, float) and pd.isna(cat_name)):
            cat_name = str(cat_name).strip()
            mapped = map_category(cat_name)
            if mapped:
                row["cat"] = mapped

        return row

    initial_count = len(train_df)
    train_df = train_df[train_df.apply(has_valid_category, axis=1)].reset_index(drop=True)
    train_df = train_df.apply(apply_category_mapping, axis=1)
    filtered_count = initial_count - len(train_df)

    typer.echo(f"  ✓ Kept {len(train_df)} records with mapped categories ({filtered_count} filtered out)")
    if len(train_df) > 0:
        typer.echo(f"  Categories in filtered dataset: {train_df['cat'].unique().tolist()}")

    if val_input:
        val_df = load_jsonl_to_df(val_input)
        val_df = val_df[val_df.apply(has_valid_category, axis=1)].reset_index(drop=True)
        val_df = val_df.apply(apply_category_mapping, axis=1)
    else:
        # Split
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(train_df) * (1.0 - val_split))
        val_df = train_df.iloc[split_idx:].reset_index(drop=True)
        train_df = train_df.iloc[:split_idx].reset_index(drop=True)

    # Extract properties with LLM if endpoint provided
    if llm_endpoint:
        typer.echo("🤖 Extracting properties with LLM...")
        typer.echo(f"  Sample categories to extract: {train_df['cat'].unique()[:3].tolist()}")

        train_df = asyncio.run(process_records_with_llm(train_df, Path(registry), llm_endpoint, llm_model))
        val_df = asyncio.run(process_records_with_llm(val_df, Path(registry), llm_endpoint, llm_model))

        typer.echo(f"  ✓ Extracted properties for {len(train_df)} train + {len(val_df)} val records")

    # Now convert to QA format with extracted properties
    typer.echo("  Converting to QA format...")

    # Check how many have properties
    if llm_endpoint:
        props_count = train_df["properties"].apply(lambda x: len(x) > 0 if isinstance(x, dict) else False).sum()
        typer.echo(f"  Records with extracted properties: {props_count}/{len(train_df)}")

    train_qa = _convert_to_qa_format(train_df, registry)
    val_qa = _convert_to_qa_format(val_df, registry)

    typer.echo(f"  ✓ Generated {len(train_qa)} train + {len(val_qa)} val QA examples")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train_qa.jsonl"
    val_out = output_dir / "val_qa.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for record in train_qa:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for record in val_qa:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    typer.echo(json.dumps({
        "train_records": len(train_qa),
        "val_records": len(val_qa),
        "train_output": str(train_out),
        "val_output": str(val_out),
    }, indent=2))


@app.command("price")
def prepare_price(
    input: Path = typer.Option(..., "--input", exists=True, help="Input JSONL with text, price, price_unit"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output directory"),
    val_input: Optional[Path] = typer.Option(None, "--val-input", exists=True, help="Optional validation JSONL"),
    val_split: float = typer.Option(0.2, "--val-split", help="Validation split"),
    ontology: Path = typer.Option(
        "resources/data/wbs/ontology.json",
        "--ontology",
        help="Path to ontology.json",
    ),
    label_maps: Path = typer.Option(
        "resources/data/wbs/label_maps.json",
        "--label-maps",
        help="Path to label_maps.json",
    ),
    registry: Optional[Path] = typer.Option(
        None,
        "--registry",
        help=f"Property registry for extraction (default: {_SETTINGS.registry_path})",
    ),
) -> None:
    """Prepare dataset for price prediction (text -> price, price_unit)."""

    typer.echo("💰 Preparing price prediction dataset...")

    # Load and validate
    maps = create_or_load_label_maps(label_maps, ontology_path=ontology)

    train_df = validate_and_prepare_base(input, maps, task="price")

    if val_input:
        val_df = validate_and_prepare_base(val_input, maps, task="price")
    else:
        # Split
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(train_df) * (1.0 - val_split))
        val_df = train_df.iloc[split_idx:].reset_index(drop=True)
        train_df = train_df.iloc[:split_idx].reset_index(drop=True)

    # Extract properties if registry provided
    if registry and registry.exists():
        from ..utils.registry_io import load_property_registry, build_registry_extractors
        from ..extraction.legacy import extract_properties

        prop_registry = load_property_registry(registry)
        extractors = build_registry_extractors(prop_registry)
        pack = extractors.to_mapping() if extractors else None

        for df in [train_df, val_df]:
            properties_list = []
            for _, row in df.iterrows():
                extracted = extract_properties(row["text"], pack) if pack else {}
                properties_list.append(extracted)
            df["properties"] = properties_list

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train_price.jsonl"
    val_out = output_dir / "val_price.jsonl"

    for path, df in [(train_out, train_df), (val_out, val_df)]:
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    # Stats
    price_units = train_df["price_unit"].value_counts().to_dict()

    typer.echo(json.dumps({
        "train_records": len(train_df),
        "val_records": len(val_df),
        "train_output": str(train_out),
        "val_output": str(val_out),
        "price_units": price_units,
    }, indent=2))


@app.command("all")
def prepare_all(
    input: Path = typer.Option(..., "--input", exists=True, help="Input JSONL"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Root output directory"),
    val_input: Optional[Path] = typer.Option(None, "--val-input", exists=True, help="Optional validation JSONL"),
    val_split: float = typer.Option(0.2, "--val-split", help="Validation split"),
    ontology: Path = typer.Option(
        "resources/data/wbs/ontology.json",
        "--ontology",
        help="Path to ontology.json",
    ),
    label_maps: Path = typer.Option(
        "resources/data/wbs/label_maps.json",
        "--label-maps",
        help="Path to label_maps.json",
    ),
    registry: Optional[Path] = typer.Option(
        None,
        "--registry",
        help=f"Property registry path (default: {_SETTINGS.registry_path})",
    ),
    llm_endpoint: Optional[str] = typer.Option(
        None,
        "--llm-endpoint",
        help="LLM endpoint for property extraction (e.g., http://localhost:8000/v1/chat/completions)",
    ),
    llm_model: str = typer.Option(
        "gpt-4o-mini",
        "--llm-model",
        help="LLM model to use for extraction",
    ),
) -> None:
    """Prepare datasets for ALL tasks (classification, span, price) in one go."""

    typer.echo("🚀 Preparing ALL datasets...")

    # Read input to determine what tasks are possible
    with open(input, "r", encoding="utf-8") as f:
        first_line = f.readline()
        sample = json.loads(first_line)

    has_categories = "super" in sample and "cat" in sample
    has_price = "price" in sample and "price_unit" in sample

    results = {}

    # 1. Classification (if categories present)
    if has_categories:
        typer.echo("\n📊 Step 1/3: Classification...")
        from typer.main import get_command
        ctx = typer.Context(get_command(app))
        ctx.invoke(
            prepare_classification,
            input=input,
            output_dir=output_dir / "classification",
            val_input=val_input,
            val_split=val_split,
            ontology=ontology,
            label_maps=label_maps,
            registry=registry,
        )
        results["classification"] = str(output_dir / "classification")

    # 2. Span extraction (if categories + registry)
    registry_to_use = registry or _SETTINGS.registry_path
    if has_categories and Path(registry_to_use).exists():
        typer.echo("\n🔍 Step 2/3: Span extraction...")
        ctx = typer.Context(get_command(app))
        ctx.invoke(
            prepare_span,
            input=input,
            output_dir=output_dir / "span",
            val_input=val_input,
            val_split=val_split,
            registry=Path(registry_to_use),
            ontology=ontology,
            label_maps=label_maps,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
        )
        results["span"] = str(output_dir / "span")

    # 3. Price (if price fields present)
    if has_price:
        typer.echo("\n💰 Step 3/3: Price prediction...")
        ctx = typer.Context(get_command(app))
        ctx.invoke(
            prepare_price,
            input=input,
            output_dir=output_dir / "price",
            val_input=val_input,
            val_split=val_split,
            ontology=ontology,
            label_maps=label_maps,
            registry=registry,
        )
        results["price"] = str(output_dir / "price")

    typer.echo("\n✅ All datasets prepared!")
    typer.echo(json.dumps({"outputs": results}, indent=2))
