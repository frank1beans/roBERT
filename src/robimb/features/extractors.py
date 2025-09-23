"""Compatibility layer exposing the extraction engine under ``robimb.features``."""

from __future__ import annotations

from ..extraction import (
    BUILTIN_NORMALIZERS,
    Normalizer,
    NormalizerFactory,
    Pattern,
    build_normalizer,
    dry_run,
    extract_properties,
)
from ..extraction.dsl import ExtractorsPack, PatternSpec, PatternSpecs

# Backwards compatible alias used by older code.
NormalizerFn = NormalizerFactory

__all__ = [
    "Normalizer",
    "NormalizerFactory",
    "NormalizerFn",
    "Pattern",
    "PatternSpec",
    "PatternSpecs",
    "ExtractorsPack",
    "BUILTIN_NORMALIZERS",
    "build_normalizer",
    "extract_properties",
    "dry_run",
]
