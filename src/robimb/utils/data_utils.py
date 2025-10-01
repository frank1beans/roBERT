"""Backward compatibility layer for legacy imports.

This module preserves the previous public API of ``utils.data_utils`` by
re-exporting the new specialised helpers introduced in ``dataset_prep``,
``registry_io`` and ``sampling``.
"""
from __future__ import annotations

from .dataset_prep import (
    LabelMaps,
    build_mask_and_report,
    create_or_load_label_maps,
    prepare_classification_dataset,
    prepare_mlm_corpus,
    save_datasets,
)
from .registry_io import (
    ExtractorsPack,
    build_registry_extractors,
    load_extractors_pack,
    load_property_registry,
    merge_extractors_pack,
)
from .sampling import load_jsonl_to_df, sample_one_record_per_category

__all__ = [
    "LabelMaps",
    "ExtractorsPack",
    "load_jsonl_to_df",
    "sample_one_record_per_category",
    "prepare_classification_dataset",
    "save_datasets",
    "prepare_mlm_corpus",
    "create_or_load_label_maps",
    "build_mask_and_report",
    "load_property_registry",
    "build_registry_extractors",
    "load_extractors_pack",
    "merge_extractors_pack",
]
