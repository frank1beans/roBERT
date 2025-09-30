"""High level orchestrator coordinating all extraction stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

from .formats import ExtractionResult, StageResult
from .llm_adapter import StructuredLLMAdapter
from .postprocess import PostProcessResult, apply_postprocess
from .rules import run_rules_stage
from .span_tagger import SpanTagger

__all__ = ["ExtractionRouter", "RouterOutput"]


def _normalize_category_labels(category: Any) -> Sequence[str]:
    if category is None:
        return []
    if isinstance(category, str):
        return [category]
    if isinstance(category, Mapping):
        value = category.get("label")
        return [value] if value else []
    labels: list[str] = []
    try:
        iterator = iter(category)
    except TypeError:
        return labels
    for item in iterator:
        labels.extend(_normalize_category_labels(item))
    return labels


def _extractor_property_ids(pack: Any) -> Set[str]:
    cache = getattr(pack, "_extractor_property_ids", None)
    if cache is None:
        extractors = getattr(pack, "extractors", {})
        patterns = extractors.get("patterns", []) if isinstance(extractors, Mapping) else []
        cache = {
            item.get("property_id")
            for item in patterns
            if isinstance(item, Mapping) and item.get("property_id")
        }
        setattr(pack, "_extractor_property_ids", cache)
    return set(cache)


def _category_property_index(pack: Any) -> Dict[str, Set[str]]:
    cache = getattr(pack, "_category_property_index", None)
    if cache is not None:
        return cache
    index: Dict[str, Set[str]] = {}
    groups = pack.registry.get("groups", {}) if getattr(pack, "registry", None) else {}

    def collect_from_groups(group_ids: Iterable[str]) -> Set[str]:
        props: Set[str] = set()
        for gid in group_ids or []:
            group = groups.get(gid)
            if isinstance(group, Mapping):
                props.update(item for item in group.get("properties", []) if isinstance(item, str))
        return props

    catmap = getattr(pack, "catmap", {})
    mappings = catmap.get("mappings", []) if isinstance(catmap, Mapping) else []
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        allowed: Set[str] = set()
        for key in ("props_required", "props_recommended"):
            allowed.update(item for item in mapping.get(key, []) if isinstance(item, str))
        allowed.update(collect_from_groups(mapping.get("groups_required", [])))
        allowed.update(collect_from_groups(mapping.get("groups_recommended", [])))
        keynote = mapping.get("keynote_mapping", {})
        if isinstance(keynote, Mapping):
            for target in keynote.values():
                if isinstance(target, str) and target:
                    allowed.add(target)
        cat_label = str(mapping.get("cat_label", "")).lower()
        cat_id = str(mapping.get("cat_id", "")).lower()
        frozen = set(allowed)
        if cat_label:
            index[cat_label] = frozen
        if cat_id:
            index[cat_id] = frozen
    setattr(pack, "_category_property_index", index)
    return index


def _find_cat_entry(pack: Any, cat_label: str) -> Optional[Mapping[str, Any]]:
    catmap = getattr(pack, "catmap", {})
    mappings = catmap.get("mappings", []) if isinstance(catmap, Mapping) else []
    for mapping in mappings:
        if not isinstance(mapping, Mapping):
            continue
        if str(mapping.get("cat_label", "")).lower() == cat_label.lower():
            return mapping
    return None


def _resolve_category(pack: Any, category_label: str):
    models = getattr(pack, "category_models", None)
    if isinstance(models, Mapping):
        for definition in models.values():
            candidate = getattr(definition, "category_label", None) or getattr(definition, "category", None)
            if isinstance(candidate, str) and candidate.lower() == category_label.lower():
                return definition
    categories = getattr(pack, "categories", {})
    if isinstance(categories, Mapping):
        for definition in categories.values():
            candidate = getattr(definition, "category_label", None) or getattr(definition, "category", None)
            if isinstance(candidate, str) and candidate.lower() == category_label.lower():
                return definition
    return None


@dataclass
class RouterOutput:
    text: str
    categories: Sequence[str]
    extraction: ExtractionResult
    postprocess: PostProcessResult

    def values(self) -> Dict[str, Any]:
        return dict(self.postprocess.values)


class ExtractionRouter:
    """Compose the multi-stage extraction pipeline."""

    def __init__(
        self,
        pack: Any,
        *,
        span_tagger: Optional[SpanTagger] = None,
        llm_adapter: Optional[StructuredLLMAdapter] = None,
    ) -> None:
        self.pack = pack
        self.span_tagger = span_tagger
        self.llm_adapter = llm_adapter

    def allowed_properties(self, categories: Any) -> Optional[Set[str]]:
        labels = {label.strip() for label in _normalize_category_labels(categories) if label}
        if not labels:
            return None
        index = _category_property_index(self.pack)
        collected: Set[str] = set()
        for label in labels:
            key = label.lower()
            collected.update(index.get(key, set()))
        return collected if collected else None

    def build_llm_prompt(self, text: str, *, extra_instructions: Optional[str] = None) -> str:
        if self.llm_adapter is None:
            raise RuntimeError("LLM adapter non configurato")
        return self.llm_adapter.build_prompt(text, extra_instructions=extra_instructions)

    def extract(
        self,
        text: str,
        *,
        categories: Any = None,
        context: Optional[Mapping[str, Any]] = None,
        target_tags: Optional[Iterable[str]] = None,
        llm_response: Optional[str] = None,
    ) -> RouterOutput:
        allowed = self.allowed_properties(categories)
        stages: list[StageResult] = []
        extractor_ids = _extractor_property_ids(self.pack)
        allowed_for_rules = None
        if allowed is not None:
            allowed_for_rules = allowed.intersection(extractor_ids)
        stages.append(
            run_rules_stage(
                text,
                getattr(self.pack, "extractors", {}),
                allowed_properties=allowed_for_rules,
                target_tags=target_tags,
            )
        )
        labels = _normalize_category_labels(categories)
        if self.span_tagger is not None:
            stages.append(
                self.span_tagger(
                    text,
                    allowed_properties=list(allowed) if allowed else None,
                    categories=labels,
                )
            )
        if self.llm_adapter is not None and llm_response:
            stages.append(self.llm_adapter.build_stage(llm_response))

        extraction = ExtractionResult(stages)
        best_values = extraction.as_value_mapping()

        primary_label = labels[0] if labels else ""
        category_def = _resolve_category(self.pack, primary_label) if primary_label else None
        cat_entry = _find_cat_entry(self.pack, primary_label) if primary_label else None
        postprocess = apply_postprocess(
            best_values,
            category=category_def,
            validators=getattr(self.pack, "validators", None),
            category_label=primary_label,
            context=context,
            cat_entry=cat_entry,
        )

        return RouterOutput(
            text=text,
            categories=labels,
            extraction=extraction,
            postprocess=postprocess,
        )

