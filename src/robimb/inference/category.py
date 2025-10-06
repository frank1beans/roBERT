"""Category inference utilities.

This module exposes :class:`CategoryInference`, a lightweight helper that
wraps either the domain-specific :class:`~robimb.models.label_model.LabelEmbedModel`
classifier or generic Hugging Face sequence classification checkpoints.

The goal is to offer a unified interface for CLI consumers and higher level
pipelines while keeping the implementation focused on inference-only concerns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from ..models.label_model import LabelEmbedModel

__all__ = ["CategoryInference", "ScoredLabel"]


@dataclass
class ScoredLabel:
    """Container for a label scored by the classifier."""

    id: int
    label: str
    score: float

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "label": self.label, "score": self.score}


class CategoryInference:
    """Unified interface for category prediction models."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        backend: str = "auto",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        label_map_path: Optional[str | Path] = None,
    ) -> None:
        """Initialise the predictor.

        Args:
            model_dir: Directory or Hugging Face model identifier.
            backend: ``"auto"`` (default) tries LabelEmbed first, then
                fallback to sequence classifier. ``"label-embed"`` forces the
                ontology-aware classifier, ``"sequence-classifier"`` forces
                Hugging Face style checkpoints.
            device: Device override (``"cpu"``/``"cuda"``). If ``None`` an
                available CUDA device is preferred.
            hf_token: Optional Hugging Face token for private models.
            label_map_path: Optional JSON file providing ``id -> label`` mapping
                when using sequence classifiers with missing metadata.
        """
        backend = backend.lower().replace("_", "-")
        if backend not in {"auto", "label-embed", "sequence-classifier"}:
            raise ValueError(f"Unsupported backend '{backend}'")

        self.model_dir = str(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self.label_map_path = Path(label_map_path) if label_map_path else None

        self.model = None
        self.tokenizer = None
        self.model_type = ""
        self.cat_labels: Dict[int, str] = {}
        self.super_labels: Dict[int, str] = {}

        init_errors: List[str] = []

        if backend in {"auto", "label-embed"}:
            try:
                self._init_label_embed()
                return
            except Exception as exc:  # pragma: no cover - defensive guard
                init_errors.append(f"label-embed: {exc}")
                if backend == "label-embed":
                    raise

        if backend in {"auto", "sequence-classifier"}:
            try:
                self._init_sequence_classifier()
                return
            except Exception as exc:  # pragma: no cover - defensive guard
                init_errors.append(f"sequence-classifier: {exc}")
                if backend == "sequence-classifier":
                    raise

        raise RuntimeError(
            f"Unable to initialise CategoryInference for '{model_dir}'. Errors: {init_errors}"
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _tokenizer_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {"use_fast": True}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        return kwargs

    def _init_label_embed(self) -> None:
        config = AutoConfig.from_pretrained(self.model_dir)
        overrides = self._build_label_embed_overrides()
        for key, value in overrides.items():
            setattr(config, key, value)

        if not getattr(config, "label_texts_super", None) or not getattr(config, "label_texts_cat", None):
            raise ValueError(
                "Label embed configuration missing label texts; expected to derive them from config.json or label_maps.json"
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, **self._tokenizer_kwargs())

        mask_matrix = getattr(config, "mask_matrix", None)
        mask_tensor = torch.tensor(mask_matrix, dtype=torch.float32) if mask_matrix is not None else None
        num_properties = getattr(config, "num_properties", 0)
        cat_mask = getattr(config, "property_cat_mask", None)
        property_cat_tensor = (
            torch.tensor(cat_mask, dtype=torch.bool) if cat_mask is not None else None
        )
        numeric_mask = getattr(config, "property_numeric_mask", None)
        property_numeric_tensor = (
            torch.tensor(numeric_mask, dtype=torch.bool) if numeric_mask is not None else None
        )
        presence_weight = getattr(config, "property_presence_weight", 1.0)
        regression_weight = getattr(config, "property_regression_weight", 1.0)

        model = LabelEmbedModel.from_pretrained(
            self.model_dir,
            config=config,
            num_super=config.num_labels_super,
            num_cat=config.num_labels_cat,
            label_texts_super=config.label_texts_super,
            label_texts_cat=config.label_texts_cat,
            tokenizer=tokenizer,
            mask_matrix=mask_tensor,
            ban_nd_in_eval=getattr(config, "ban_nd_in_eval", True),
            nd_id=getattr(config, "nd_id", None),
            num_properties=num_properties,
            property_cat_mask=property_cat_tensor,
            property_numeric_mask=property_numeric_tensor,
            property_presence_weight=presence_weight,
            property_regression_weight=regression_weight,
        )
        model.eval()
        model.to(self.device)

        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Label embed model is missing config metadata")

        label_texts_cat = list(getattr(config, "label_texts_cat", []))
        if not label_texts_cat:
            raise ValueError("Config does not expose label_texts_cat")
        label_texts_super = list(getattr(config, "label_texts_super", []))

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = "label-embed"
        self.cat_labels = {idx: str(label) for idx, label in enumerate(label_texts_cat)}
        self.super_labels = {idx: str(label) for idx, label in enumerate(label_texts_super)}

    def _init_sequence_classifier(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, **self._tokenizer_kwargs())

        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        except AttributeError as exc:
            if "metadata" in str(exc) or "NoneType" in str(exc):
                model = self._load_sequence_model_from_safetensors()
            else:
                raise
        model.eval()
        model.to(self.device)

        label_map: Dict[int, str] = {}

        config = getattr(model, "config", None)
        if config is not None:
            label_map = self._coerce_id2label(getattr(config, "id2label", None))

        if not label_map and self.label_map_path:
            label_map = self._load_id2label_from_file(self.label_map_path)

        if not label_map and config is not None:
            label_map = self._load_id2label_from_config(config)

        if not label_map:
            label_map = self._load_id2label_from_label_maps()

        if not label_map:
            raise ValueError("Unable to resolve label mapping for sequence classifier")

        self.model = model
        self.tokenizer = tokenizer
        self.model_type = "sequence-classifier"
        self.cat_labels = label_map
        self.super_labels = {}

    @staticmethod
    def _coerce_id2label(raw: object) -> Dict[int, str]:
        if not isinstance(raw, dict):
            return {}
        mapping: Dict[int, str] = {}
        for key, value in raw.items():
            try:
                mapping[int(key)] = str(value)
            except Exception:
                continue
        if mapping:
            return mapping

        inverted: Dict[int, str] = {}
        for label, idx in raw.items():
            try:
                inverted[int(idx)] = str(label)
            except Exception:
                continue
        return inverted

    @staticmethod
    def _load_id2label_from_file(path: Path) -> Dict[int, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return CategoryInference._coerce_id2label(data)
        raise ValueError(f"Invalid id2label file format: {path}")

    def _build_label_embed_overrides(self) -> Dict[str, object]:
        config_path = Path(self.model_dir) / "config.json"
        overrides: Dict[str, object] = {}
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            if "label_texts_super" not in payload:
                labels = self._ordered_labels_from_mapping(payload.get("id2super"))
                if labels:
                    overrides["label_texts_super"] = labels
            if "label_texts_cat" not in payload:
                labels = self._ordered_labels_from_mapping(payload.get("id2cat"))
                if labels:
                    overrides["label_texts_cat"] = labels
        if not overrides.get("label_texts_cat"):
            label_maps = self._load_id2label_from_label_maps()
            if label_maps:
                overrides["label_texts_cat"] = [label_maps[idx] for idx in sorted(label_maps)]
        return overrides

    @staticmethod
    def _ordered_labels_from_mapping(mapping: object) -> List[str]:
        if not isinstance(mapping, dict):
            return []
        ordered: List[str] = []
        try:
            for idx in sorted(mapping, key=lambda value: int(value)):
                ordered.append(str(mapping[idx]))
        except Exception:
            return []
        return ordered

    def _load_id2label_from_config(self, config: AutoConfig) -> Dict[int, str]:
        mapping = self._coerce_id2label(getattr(config, "id2label", None))
        if mapping:
            return mapping
        label2id = getattr(config, "label2id", None)
        if isinstance(label2id, dict):
            inverted: Dict[int, str] = {}
            for label, idx in label2id.items():
                try:
                    inverted[int(idx)] = str(label)
                except Exception:
                    continue
            return inverted
        num_labels = getattr(config, "num_labels", None)
        if isinstance(num_labels, int) and num_labels > 0:
            return {idx: str(idx) for idx in range(num_labels)}
        return {}

    def _load_id2label_from_label_maps(self) -> Dict[int, str]:
        candidate = Path(self.model_dir) / "label_maps.json"
        if not candidate.exists():
            return {}

        data = json.loads(candidate.read_text(encoding="utf-8"))
        cats = data.get("id2label")
        if isinstance(cats, dict):
            return self._coerce_id2label(cats)

        cats = data.get("cats")
        if isinstance(cats, dict):
            mapping: Dict[int, str] = {}
            for label, idx in cats.items():
                try:
                    mapping[int(idx)] = str(label)
                except Exception:
                    continue
            if mapping:
                return mapping
        return {}

    def _load_sequence_model_from_safetensors(self) -> AutoModelForSequenceClassification:
        config = AutoConfig.from_pretrained(self.model_dir)
        model = AutoModelForSequenceClassification.from_config(config)
        safetensor_path = Path(self.model_dir) / "model.safetensors"
        if not safetensor_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {self.model_dir}")

        from safetensors.torch import load_file

        state_dict = load_file(str(safetensor_path))
        info = model.load_state_dict(state_dict, strict=False)
        if getattr(info, "missing_keys", []) or getattr(info, "unexpected_keys", []):
            # Ensure we surface potential loading issues for debugging
            missing = ", ".join(info.missing_keys)
            unexpected = ", ".join(info.unexpected_keys)
            if missing or unexpected:
                raise ValueError(
                    f"Checkpoint at {safetensor_path} has missing keys [{missing}] "
                    f"and unexpected keys [{unexpected}]"
                )
        return model

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict(
        self,
        text: str,
        *,
        top_k: int = 5,
        max_length: int = 512,
        return_scores: bool = False,
    ) -> Dict[str, object]:
        """Predict the most likely categories for ``text``."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("CategoryInference is not initialised")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        result: Dict[str, object] = {"backend": self.model_type}

        if self.model_type == "label-embed":
            logits_super = outputs["logits_super"].detach()[0]
            logits_cat = outputs["logits_cat_pred_masked"].detach()[0]

            probs_super = F.softmax(logits_super.float(), dim=-1).cpu()
            probs_cat = F.softmax(logits_cat.float(), dim=-1).cpu()

            super_topk = self._build_topk(probs_super, self.super_labels, top_k)
            cat_topk = self._build_topk(probs_cat, self.cat_labels, top_k)

            if super_topk:
                result["supercategory"] = super_topk[0].to_dict()
            if cat_topk:
                result["category"] = cat_topk[0].to_dict()

            result["supercategories"] = [item.to_dict() for item in super_topk]
            result["categories"] = [item.to_dict() for item in cat_topk]

            if return_scores:
                result["raw_scores"] = {
                    "super_logits": logits_super.cpu().tolist(),
                    "super_probabilities": probs_super.tolist(),
                    "category_logits": logits_cat.cpu().tolist(),
                    "category_probabilities": probs_cat.tolist(),
                }
        else:
            logits = outputs.logits.detach()[0]
            probs = F.softmax(logits.float(), dim=-1).cpu()

            cat_topk = self._build_topk(probs, self.cat_labels, top_k)
            if cat_topk:
                result["category"] = cat_topk[0].to_dict()
            result["categories"] = [item.to_dict() for item in cat_topk]

            if return_scores:
                result["raw_scores"] = {
                    "category_logits": logits.cpu().tolist(),
                    "category_probabilities": probs.tolist(),
                }

        return result

    def predict_batch(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 5,
        max_length: int = 512,
        return_scores: bool = False,
    ) -> List[Dict[str, object]]:
        """Run predictions for multiple texts."""
        return [
            self.predict(text, top_k=top_k, max_length=max_length, return_scores=return_scores)
            for text in texts
        ]

    def _build_topk(
        self,
        probs: torch.Tensor,
        label_map: Dict[int, str],
        top_k: int,
    ) -> List[ScoredLabel]:
        if probs.ndim != 1:
            probs = probs.view(-1)

        num_labels = probs.numel()
        k = max(1, min(top_k, num_labels))
        values, indices = torch.topk(probs, k=k)

        top_items: List[ScoredLabel] = []
        for value, index in zip(values.tolist(), indices.tolist()):
            idx = int(index)
            label = label_map.get(idx, str(idx))
            top_items.append(ScoredLabel(id=idx, label=label, score=float(value)))
        return top_items
