"""Legacy regex-based property extraction engine.

This module preserves the previous pattern-driven implementation so that
existing tooling keeps working while the new schema-first pipeline is
developed. Callers are encouraged to migrate to the modern primitives in
:mod:`robimb.extraction`.
"""

from .engine import (
    Pattern,
    PatternValidationError,
    dry_run,
    extract_properties,
    validate_extractors_pack,
)
from .dsl import ExtractorsPack, PatternSpec, PatternSpecs
from .formats import ExtractionCandidate, ExtractionResult, StageResult
from .llm_adapter import SchemaField, StructuredLLMAdapter
from .normalizers import BUILTIN_NORMALIZERS, Normalizer, NormalizerFactory, build_normalizer
from .pack_testing import run_pack_dataset_evaluation
from .postprocess import PostProcessResult, apply_postprocess
from .router import ExtractionRouter, RouterOutput
from .rules import run_rules_stage
from .span_tagger import SpanTagger, SpanTaggerOutput, build_stage

__all__ = [
    "Pattern",
    "PatternValidationError",
    "dry_run",
    "extract_properties",
    "validate_extractors_pack",
    "ExtractorsPack",
    "PatternSpec",
    "PatternSpecs",
    "ExtractionCandidate",
    "ExtractionResult",
    "StageResult",
    "SchemaField",
    "StructuredLLMAdapter",
    "BUILTIN_NORMALIZERS",
    "Normalizer",
    "NormalizerFactory",
    "build_normalizer",
    "run_pack_dataset_evaluation",
    "PostProcessResult",
    "apply_postprocess",
    "ExtractionRouter",
    "RouterOutput",
    "run_rules_stage",
    "SpanTagger",
    "SpanTaggerOutput",
    "build_stage",
]
