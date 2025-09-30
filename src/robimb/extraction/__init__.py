"""Schema-first property extraction primitives.

The :mod:`robimb.extraction` package now exposes the modern building blocks
for the hybrid extractor (schema registry, deterministic parsers, upcoming
LLM orchestration). The previous regex-based engine is still available
under :mod:`robimb.extraction.legacy` and is re-exported for backward
compatibility.
"""

from .schema_registry import CategorySchema, PropertySpec, SchemaRegistry, load_registry
from .parsers.dimensions import DimensionMatch, parse_dimensions
from .parsers.numbers import NumberSpan, extract_numbers, parse_number_it
from .parsers.units import UnitMatch, normalize_unit, scan_units

# Backwards compatible legacy engine ---------------------------------------
from . import legacy
from .legacy import (
    BUILTIN_NORMALIZERS,
    ExtractionRouter,
    Normalizer,
    NormalizerFactory,
    Pattern,
    PatternSpec,
    PatternSpecs,
    SchemaField,
    SpanTagger,
    SpanTaggerOutput,
    StructuredLLMAdapter,
    build_normalizer,
    build_stage,
    dry_run,
    extract_properties,
    run_pack_dataset_evaluation,
    run_rules_stage,
    validate_extractors_pack,
)

# Convenience aliases for the parser API -----------------------------------
extract_dimensions = parse_dimensions
extract_units = scan_units

__all__ = [
    "CategorySchema",
    "PropertySpec",
    "SchemaRegistry",
    "load_registry",
    "NumberSpan",
    "extract_numbers",
    "parse_number_it",
    "DimensionMatch",
    "parse_dimensions",
    "extract_dimensions",
    "UnitMatch",
    "normalize_unit",
    "scan_units",
    "extract_units",
    # Legacy exports kept for compatibility
    "legacy",
    "ExtractionRouter",
    "Pattern",
    "PatternSpec",
    "PatternSpecs",
    "BUILTIN_NORMALIZERS",
    "Normalizer",
    "NormalizerFactory",
    "build_normalizer",
    "build_stage",
    "dry_run",
    "extract_properties",
    "run_pack_dataset_evaluation",
    "run_rules_stage",
    "SchemaField",
    "StructuredLLMAdapter",
    "SpanTagger",
    "SpanTaggerOutput",
    "validate_extractors_pack",
]
