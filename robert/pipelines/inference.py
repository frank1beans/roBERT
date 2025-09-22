"""Inference pipeline combining hierarchy prediction and property extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import torch
from transformers import AutoConfig, AutoTokenizer

from ..config import InferenceConfig
from ..data.ontology import build_mask, load_label_maps, load_ontology
from ..models.masked import MultiTaskBERTMasked
from ..models.label import LabelEmbedModel
from ..properties.extractors import PropertyExtractionResult, RegexPropertyExtractor
from ..properties.registry import PropertyRegistry


__all__ = ["InferencePipeline", "PredictionOutput"]


@dataclass(slots=True)
class PredictionOutput:
    text: str
    super_id: int
    super_label: str
    super_score: float
    cat_id: int
    cat_label: str
    cat_score: float
    properties: Dict[str, str]
    property_matches: List[PropertyExtractionResult]


class InferencePipeline:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.masked_model_path)

        if config.label_maps_path is None:
            raise ValueError("InferenceConfig.label_maps_path must be provided")
        (
            self.super_name_to_id,
            self.cat_name_to_id,
            self.super_id_to_name,
            self.cat_id_to_name,
        ) = load_label_maps(config.label_maps_path)

        mask_matrix = None
        if config.ontology_path:
            ontology = load_ontology(config.ontology_path)
            mask, _ = build_mask(
                ontology,
                self.super_name_to_id,
                self.cat_name_to_id,
                return_report=True,
            )
            mask_matrix = torch.tensor(mask, dtype=torch.float32)
        else:
            mask_matrix = torch.ones(
                (len(self.super_name_to_id), len(self.cat_name_to_id)), dtype=torch.float32
            )

        model_config = AutoConfig.from_pretrained(config.masked_model_path)
        self.masked_model = MultiTaskBERTMasked.from_pretrained(
            str(config.masked_model_path),
            config=model_config,
            num_super=len(self.super_name_to_id),
            num_cat=len(self.cat_name_to_id),
            mask_matrix=mask_matrix,
            nd_id=None,
            ban_nd_in_eval=False,
        ).to(self.device)
        self.masked_model.eval()

        self.label_model: Optional[LabelEmbedModel] = None
        if config.label_model_path:
            label_config = AutoConfig.from_pretrained(config.label_model_path)
            super_labels = [label for _, label in sorted(self.super_id_to_name.items())]
            cat_labels = [label for _, label in sorted(self.cat_id_to_name.items())]
            self.label_model = LabelEmbedModel.from_pretrained(
                str(config.label_model_path),
                config=label_config,
                num_super=len(self.super_name_to_id),
                num_cat=len(self.cat_name_to_id),
                mask_matrix=mask_matrix,
                nd_id=None,
                backbone_src=str(config.label_model_path),
                label_texts_super=super_labels,
                label_texts_cat=cat_labels,
                tokenizer=self.tokenizer,
            ).to(self.device)
            self.label_model.eval()

        self.property_registry: Optional[PropertyRegistry] = None
        self.property_extractor: Optional[RegexPropertyExtractor] = None
        if config.properties_registry_path:
            self.property_registry = PropertyRegistry.from_json(config.properties_registry_path)
            self.property_extractor = RegexPropertyExtractor(self.property_registry)

    def _batch(self, iterable: Iterable[str], size: int) -> Iterable[List[str]]:
        batch: List[str] = []
        for item in iterable:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    @torch.inference_mode()
    def predict(self, texts: Iterable[str]) -> List[PredictionOutput]:
        results: List[PredictionOutput] = []
        for chunk in self._batch(texts, self.config.batch_size):
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.masked_model(**encoded)
            logits_super = outputs["logits_super"]
            logits_cat = outputs["logits_cat_pred_masked"]
            probs_super = torch.softmax(logits_super, dim=-1)
            probs_cat = torch.softmax(logits_cat, dim=-1)
            top_super = probs_super.argmax(dim=-1)
            top_cat = probs_cat.argmax(dim=-1)
            for i, text in enumerate(chunk):
                sid = int(top_super[i].item())
                cid = int(top_cat[i].item())
                s_label = self.super_id_to_name.get(sid, str(sid))
                c_label = self.cat_id_to_name.get(cid, str(cid))
                s_score = float(probs_super[i, sid].item())
                c_score = float(probs_cat[i, cid].item())
                properties: Dict[str, str] = {}
                matches: List[PropertyExtractionResult] = []
                if self.property_extractor is not None and self.property_registry is not None:
                    group_key = f"{s_label}|{c_label}"
                    if group_key in self.property_registry:
                        matches = self.property_extractor.extract(text, group_key)
                        properties = {m.slot.name: m.value for m in matches}
                results.append(
                    PredictionOutput(
                        text=text,
                        super_id=sid,
                        super_label=s_label,
                        super_score=s_score,
                        cat_id=cid,
                        cat_label=c_label,
                        cat_score=c_score,
                        properties=properties,
                        property_matches=matches,
                    )
                )
        return results
