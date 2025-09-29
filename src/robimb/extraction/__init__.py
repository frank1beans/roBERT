"""Property extraction package."""

from .dsl import ExtractorsPack, PatternSpec, PatternSpecs
from .engine import Pattern, dry_run, extract_properties
from .formats import ExtractionCandidate, ExtractionResult, StageResult
from .llm_adapter import SchemaField, StructuredLLMAdapter
from .normalizers import BUILTIN_NORMALIZERS, Normalizer, NormalizerFactory, build_normalizer
from .postprocess import PostProcessResult, apply_postprocess
from .router import ExtractionRouter, RouterOutput
from .rules import run_rules_stage
from .span_tagger import SpanTagger, SpanTaggerOutput, build_stage

__all__ = [
    "ExtractorsPack",
    "PatternSpec",
    "PatternSpecs",
    "Pattern",
    "dry_run",
    "extract_properties",
    "BUILTIN_NORMALIZERS",
    "Normalizer",
    "NormalizerFactory",
    "build_normalizer",
    "ExtractionCandidate",
    "ExtractionResult",
    "StageResult",
    "SchemaField",
    "StructuredLLMAdapter",
    "PostProcessResult",
    "apply_postprocess",
    "ExtractionRouter",
    "RouterOutput",
    "run_rules_stage",
    "SpanTagger",
    "SpanTaggerOutput",
    "build_stage",
]
