"""Core package exposing training and inference utilities for roBERT."""
from .config import (  # noqa: F401
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    PipelineConfig,
)

from .data.ontology import (  # noqa: F401
    load_ontology,
    load_label_maps,
    build_name_maps,
    build_mask,
)

from .properties.registry import (  # noqa: F401
    PropertyRegistry,
    PropertyGroup,
    PropertySlot,
)

from .properties.extractors import (  # noqa: F401
    RegexPropertyExtractor,
    PropertyExtractionResult,
)

from .pipelines.inference import InferencePipeline  # noqa: F401
from .pipelines.training import (  # noqa: F401
    MaskedMLMTrainingPipeline,
    LabelEmbeddingTrainingPipeline,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "PipelineConfig",
    "load_ontology",
    "load_label_maps",
    "build_name_maps",
    "build_mask",
    "PropertyRegistry",
    "PropertyGroup",
    "PropertySlot",
    "RegexPropertyExtractor",
    "PropertyExtractionResult",
    "InferencePipeline",
    "MaskedMLMTrainingPipeline",
    "LabelEmbeddingTrainingPipeline",
]
