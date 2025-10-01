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
from .parsers.colors import RALColor, parse_ral_colors
from .parsers.standards import StandardMatch, parse_standards
from .prompts import PromptLibrary, PromptTemplate, load_prompt_library
from .lexicon import load_norms_by_category, load_producers_by_category
from .matchers.brands import BrandMatcher
from .matchers.materials import MaterialMatcher
from .normalize import (
    normalize_boolean,
    normalize_confidence,
    normalize_dimension_mm,
    normalize_string,
)
from .validators import (
    ALLOWED_SOURCES,
    PropertyPayload,
    ValidationIssue,
    ValidationResult,
    validate_properties,
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
    "RALColor",
    "parse_ral_colors",
    "StandardMatch",
    "parse_standards",
    "PromptTemplate",
    "PromptLibrary",
    "load_prompt_library",
    "BrandMatcher",
    "MaterialMatcher",
    "load_norms_by_category",
    "load_producers_by_category",
    "normalize_string",
    "normalize_boolean",
    "normalize_dimension_mm",
    "normalize_confidence",
    "ALLOWED_SOURCES",
    "PropertyPayload",
    "ValidationIssue",
    "ValidationResult",
    "validate_properties"
]
