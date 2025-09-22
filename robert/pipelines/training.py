"""High level training pipelines for masked multi-task and label embedding models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..config import PipelineConfig
from ..data.ontology import Ontology, build_mask, load_label_maps, load_ontology
from ..models.masked import MultiTaskBERTMasked
from ..models.label import LabelEmbedModel
from ..properties.registry import PropertyRegistry


__all__ = [
    "MaskedMLMTrainingPipeline",
    "LabelEmbeddingTrainingPipeline",
]


def _load_jsonl(path: str | Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class _DatasetFields:
    text: str
    super_name: Optional[str]
    cat_name: Optional[str]
    super_id: Optional[str]
    cat_id: Optional[str]


class _BaseTrainingPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        label_maps_path: str | Path,
        ontology_path: Optional[str | Path] = None,
        properties_registry: Optional[PropertyRegistry] = None,
    ):
        self.config = config
        self.label_maps_path = Path(label_maps_path)
        self.ontology_path = Path(ontology_path) if ontology_path else None
        self.properties_registry = properties_registry
        (
            self.super_name_to_id,
            self.cat_name_to_id,
            self.super_id_to_name,
            self.cat_id_to_name,
        ) = load_label_maps(self.label_maps_path)
        self.ontology: Optional[Ontology] = None
        self.mask_matrix: Optional[np.ndarray] = None
        if self.ontology_path and self.ontology_path.exists():
            self.ontology = load_ontology(self.ontology_path)
            self.mask_matrix, _ = build_mask(
                self.ontology, self.super_name_to_id, self.cat_name_to_id, return_report=True
            )
        extra = self.config.extra_args
        self.fields = _DatasetFields(
            text=extra.get("text_field", "text"),
            super_name=extra.get("super_field"),
            cat_name=extra.get("cat_field"),
            super_id=extra.get("super_id_field"),
            cat_id=extra.get("cat_id_field"),
        )

    def _resolve_super(self, record: Dict[str, object]) -> int:
        if self.fields.super_id and self.fields.super_id in record:
            return int(record[self.fields.super_id])
        if self.fields.super_name and self.fields.super_name in record:
            value = str(record[self.fields.super_name])
            if value in self.super_name_to_id:
                return self.super_name_to_id[value]
        # fallback to normalized search
        value = str(record.get(self.fields.super_name or "super", ""))
        normalized = value.lower()
        for name, idx in self.super_name_to_id.items():
            if name.lower() == normalized:
                return idx
        raise KeyError(f"Cannot resolve super label for record: {record}")

    def _resolve_cat(self, record: Dict[str, object]) -> int:
        if self.fields.cat_id and self.fields.cat_id in record:
            return int(record[self.fields.cat_id])
        if self.fields.cat_name and self.fields.cat_name in record:
            value = str(record[self.fields.cat_name])
            if value in self.cat_name_to_id:
                return self.cat_name_to_id[value]
        value = str(record.get(self.fields.cat_name or "cat", ""))
        normalized = value.lower()
        for name, idx in self.cat_name_to_id.items():
            if name.lower() == normalized:
                return idx
        raise KeyError(f"Cannot resolve cat label for record: {record}")

    def _prepare_records(self, records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        prepared: List[Dict[str, object]] = []
        text_field = self.fields.text
        for record in records:
            text = str(record.get(text_field, "")).strip()
            if not text:
                continue
            prepared.append(
                {
                    "text": text,
                    "super_labels": self._resolve_super(record),
                    "cat_labels": self._resolve_cat(record),
                }
            )
        return prepared

    def _build_dataset(self, path: str | Path) -> Dataset:
        records = _load_jsonl(path)
        prepared = self._prepare_records(records)
        return Dataset.from_list(prepared)

    def _tokenize(self, dataset: Dataset, tokenizer, max_length: int) -> Dataset:
        def _tokenize_batch(batch):
            tokens = tokenizer(batch["text"], padding=False, truncation=True, max_length=max_length)
            tokens["super_labels"] = batch["super_labels"]
            tokens["cat_labels"] = batch["cat_labels"]
            return tokens

        return dataset.map(_tokenize_batch, batched=True, remove_columns=["text"])


class MaskedMLMTrainingPipeline(_BaseTrainingPipeline):
    """End-to-end training pipeline for :class:`MultiTaskBERTMasked`."""

    def fit(
        self,
        train_path: str | Path,
        val_path: Optional[str | Path] = None,
        class_weights: Optional[Tuple[List[float], List[float]]] = None,
    ) -> Trainer:
        cfg = self.config
        training_cfg = cfg.training
        _set_seed(training_cfg.seed)

        tokenizer_name = cfg.model.tokenizer_name or cfg.model.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_dataset = self._tokenize(
            self._build_dataset(train_path), tokenizer, training_cfg.max_length
        )
        eval_dataset = (
            self._tokenize(self._build_dataset(val_path), tokenizer, training_cfg.max_length)
            if val_path
            else None
        )

        config_overrides = dict(cfg.model.config_overrides)
        config_overrides.setdefault("num_labels", len(self.super_name_to_id))
        hf_config = AutoConfig.from_pretrained(cfg.model.name_or_path, **config_overrides)

        mask_tensor = None
        if self.mask_matrix is not None:
            mask_tensor = torch.tensor(self.mask_matrix, dtype=torch.float32)

        model = MultiTaskBERTMasked(
            config=hf_config,
            num_super=len(self.super_name_to_id),
            num_cat=len(self.cat_name_to_id),
            mask_matrix=mask_tensor if mask_tensor is not None else torch.ones(
                (len(self.super_name_to_id), len(self.cat_name_to_id)), dtype=torch.float32
            ),
            backbone_src=cfg.model.name_or_path,
            nd_id=cfg.model.nd_label_id,
            use_mean_pool=cfg.model.use_mean_pool,
            proj_dim=cfg.model.proj_dim,
            use_arcface=cfg.model.arcface,
        )

        if class_weights:
            super_w, cat_w = class_weights
            if super_w:
                model.set_super_class_weights(torch.tensor(super_w, dtype=torch.float32))
            if cat_w:
                model.set_cat_class_weights(torch.tensor(cat_w, dtype=torch.float32))

        data_collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir=str(training_cfg.output_dir),
            num_train_epochs=training_cfg.epochs,
            per_device_train_batch_size=training_cfg.batch_size,
            per_device_eval_batch_size=training_cfg.batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            save_total_limit=training_cfg.save_total_limit,
            seed=training_cfg.seed,
            weight_decay=training_cfg.weight_decay,
            fp16=training_cfg.fp16,
            push_to_hub=training_cfg.push_to_hub,
            hub_model_id=training_cfg.hub_repo,
            logging_steps=50,
        )

        def _compute_metrics(eval_pred):
            preds = getattr(eval_pred, "predictions", eval_pred)
            labels = getattr(eval_pred, "label_ids", None)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            logits = np.asarray(preds)
            logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            if isinstance(labels, dict):
                y_s = np.asarray(labels["super_labels"])
                y_c = np.asarray(labels["cat_labels"])
            elif isinstance(labels, (tuple, list)):
                y_s = np.asarray(labels[0])
                y_c = np.asarray(labels[1])
            else:
                labels = np.asarray(labels)
                y_s = labels[:, 0]
                y_c = labels[:, 1]

            S = len(self.super_name_to_id)
            C = len(self.cat_name_to_id)
            logits_super = logits[:, :S]
            logits_cat = logits[:, S:S + C]
            pred_super = logits_super.argmax(-1)
            pred_cat = logits_cat.argmax(-1)
            acc_super = (pred_super == y_s).mean()
            valid = y_c != -100
            acc_cat = (pred_cat[valid] == y_c[valid]).mean() if valid.any() else 0.0
            return {
                "acc_super": float(acc_super),
                "acc_cat": float(acc_cat),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics if eval_dataset is not None else None,
        )
        return trainer


class LabelEmbeddingTrainingPipeline(_BaseTrainingPipeline):
    """Training pipeline for the label embedding model."""

    def __init__(
        self,
        config: PipelineConfig,
        label_maps_path: str | Path,
        ontology_path: Optional[str | Path] = None,
        properties_registry: Optional[PropertyRegistry] = None,
        label_texts_super: Optional[List[str]] = None,
        label_texts_cat: Optional[List[str]] = None,
    ):
        super().__init__(config, label_maps_path, ontology_path, properties_registry)
        self.label_texts_super = label_texts_super or list(self.super_name_to_id.keys())
        self.label_texts_cat = label_texts_cat or list(self.cat_name_to_id.keys())

    def fit(
        self,
        train_path: str | Path,
        val_path: Optional[str | Path] = None,
    ) -> Trainer:
        cfg = self.config
        training_cfg = cfg.training
        _set_seed(training_cfg.seed)

        tokenizer_name = cfg.model.tokenizer_name or cfg.model.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        train_dataset = self._tokenize(
            self._build_dataset(train_path), tokenizer, training_cfg.max_length
        )
        eval_dataset = (
            self._tokenize(self._build_dataset(val_path), tokenizer, training_cfg.max_length)
            if val_path
            else None
        )

        config_overrides = dict(cfg.model.config_overrides)
        config_overrides.setdefault("num_labels", len(self.super_name_to_id))
        hf_config = AutoConfig.from_pretrained(cfg.model.name_or_path, **config_overrides)

        mask_tensor = None
        if self.mask_matrix is not None:
            mask_tensor = torch.tensor(self.mask_matrix, dtype=torch.float32)

        model = LabelEmbedModel(
            config=hf_config,
            num_super=len(self.super_name_to_id),
            num_cat=len(self.cat_name_to_id),
            label_texts_super=self.label_texts_super,
            label_texts_cat=self.label_texts_cat,
            tokenizer=tokenizer,
            backbone_src=cfg.model.name_or_path,
            proj_dim=cfg.model.proj_dim or 256,
            mask_matrix=mask_tensor,
            nd_id=cfg.model.nd_label_id,
        )

        data_collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir=str(training_cfg.output_dir),
            num_train_epochs=training_cfg.epochs,
            per_device_train_batch_size=training_cfg.batch_size,
            per_device_eval_batch_size=training_cfg.batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            save_total_limit=training_cfg.save_total_limit,
            seed=training_cfg.seed,
            weight_decay=training_cfg.weight_decay,
            fp16=training_cfg.fp16,
            push_to_hub=training_cfg.push_to_hub,
            hub_model_id=training_cfg.hub_repo,
            logging_steps=50,
        )

        def _compute_metrics(eval_pred):
            preds = getattr(eval_pred, "predictions", eval_pred)
            labels = getattr(eval_pred, "label_ids", None)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            logits = np.asarray(preds)
            logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            if isinstance(labels, dict):
                y_s = np.asarray(labels["super_labels"])
                y_c = np.asarray(labels["cat_labels"])
            elif isinstance(labels, (tuple, list)):
                y_s = np.asarray(labels[0])
                y_c = np.asarray(labels[1])
            else:
                labels = np.asarray(labels)
                y_s = labels[:, 0]
                y_c = labels[:, 1]

            S = len(self.super_name_to_id)
            C = len(self.cat_name_to_id)
            logits_super = logits[:, :S]
            logits_cat = logits[:, S:S + C]
            pred_super = logits_super.argmax(-1)
            pred_cat = logits_cat.argmax(-1)
            acc_super = (pred_super == y_s).mean()
            valid = y_c != -100
            acc_cat = (pred_cat[valid] == y_c[valid]).mean() if valid.any() else 0.0
            return {
                "acc_super": float(acc_super),
                "acc_cat": float(acc_cat),
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics if eval_dataset is not None else None,
        )
        return trainer
