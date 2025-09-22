"""Label embedding classifier with ontology-aware masking.

This module exposes :class:`LabelEmbedModel`, a Transformer-based
classifier that scores classes by comparing document embeddings with
label prototypes.  The implementation is adapted from the historical
``robert`` package and refactored to live in the ``robimb`` namespace.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

__all__ = ["LabelEmbedModel"]


def _very_neg_like(t: torch.Tensor) -> torch.Tensor:
    """Return a tensor filled with a very negative value matching ``t``."""

    return torch.tensor(-1e4, dtype=t.dtype, device=t.device)


class MeanPool(nn.Module):
    """Mean pooling that respects the attention mask."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class EmbHead(nn.Module):
    """Two-layer projection head with optional L2 normalisation."""

    def __init__(self, in_dim: int, hid: int = 384, out_dim: int = 256, l2_normalize: bool = True):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1) if self.l2_normalize else z


class LabelEmbedModel(nn.Module):
    """Transformer encoder plus ontology-aware label embeddings."""

    def __init__(
        self,
        config: AutoConfig,
        num_super: int,
        num_cat: int,
        label_texts_super: List[str],
        label_texts_cat: List[str],
        tokenizer,
        backbone_src: str,
        proj_dim: int = 256,
        temperature: float = 0.07,
        l2_normalize_emb: bool = True,
        use_mean_pool: bool = True,
        mask_matrix: Optional[torch.Tensor] = None,
        ban_nd_in_eval: bool = True,
        nd_id: Optional[int] = None,
        freeze_encoder: bool = False,
        train_label_emb: bool = True,
        num_properties: int = 0,
        property_cat_mask: Optional[torch.Tensor] = None,
        property_numeric_mask: Optional[torch.Tensor] = None,
        property_presence_weight: float = 1.0,
        property_regression_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_super = int(num_super)
        self.num_cat = int(num_cat)
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.use_mean_pool = bool(use_mean_pool)
        self.l2_normalize_emb = bool(l2_normalize_emb)
        self.nd_id = nd_id
        self.ban_nd_in_eval = bool(ban_nd_in_eval)
        self.config = config
        self.tokenizer = tokenizer
        self.num_properties = int(num_properties or 0)
        self.property_presence_weight = float(property_presence_weight)
        self.property_regression_weight = float(property_regression_weight)

        self.backbone = AutoModel.from_pretrained(backbone_src, config=config)
        hidden = getattr(self.backbone.config, "hidden_size", getattr(config, "hidden_size", 768))

        self.pool = MeanPool()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.emb_head = EmbHead(hidden, 384, self.proj_dim, l2_normalize=self.l2_normalize_emb)
        self.logit_scale = nn.Parameter(torch.tensor(float(1.0 / max(1e-6, self.temperature))))

        with torch.no_grad():
            sup_init = self._encode_label_texts(label_texts_super)
            cat_init = self._encode_label_texts(label_texts_cat)

        if train_label_emb:
            self.super_emb = nn.Parameter(sup_init)
            self.cat_emb = nn.Parameter(cat_init)
        else:
            self.register_buffer("super_emb", sup_init)
            self.register_buffer("cat_emb", cat_init)

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if mask_matrix is not None:
            if mask_matrix.dtype != torch.float32:
                mask_matrix = mask_matrix.to(dtype=torch.float32)
            if mask_matrix.shape != (self.num_super, self.num_cat):
                raise ValueError(
                    "mask_matrix shape errata: atteso (%s, %s), trovato %s"
                    % (self.num_super, self.num_cat, tuple(mask_matrix.shape))
                )
        self.register_buffer("mask_matrix", mask_matrix, persistent=False)

        if self.num_properties > 0:
            self.property_presence_head = nn.Linear(self.proj_dim, self.num_properties)
            self.property_regression_head = nn.Linear(self.proj_dim, self.num_properties)

            if property_cat_mask is None:
                property_cat_mask = torch.ones(
                    (self.num_cat, self.num_properties), dtype=torch.bool
                )
            elif property_cat_mask.shape != (self.num_cat, self.num_properties):
                raise ValueError(
                    "property_cat_mask shape errata: atteso (%s, %s), trovato %s"
                    % (self.num_cat, self.num_properties, tuple(property_cat_mask.shape))
                )
            self.register_buffer(
                "property_cat_mask",
                property_cat_mask.to(dtype=torch.bool),
                persistent=False,
            )

            if property_numeric_mask is None:
                property_numeric_mask = torch.zeros(self.num_properties, dtype=torch.bool)
            elif property_numeric_mask.shape != (self.num_properties,):
                raise ValueError(
                    "property_numeric_mask shape errata: atteso (%s,), trovato %s"
                    % (self.num_properties, tuple(property_numeric_mask.shape))
                )
            self.register_buffer(
                "property_numeric_mask",
                property_numeric_mask.to(dtype=torch.bool),
                persistent=False,
            )
        else:
            self.property_presence_head = None
            self.property_regression_head = None
            self.property_cat_mask = None
            self.property_numeric_mask = None

    @torch.no_grad()
    def _encode_label_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        ).to(self.logit_scale.device)
        outputs = self.backbone(**tokens).last_hidden_state
        pooled = self.pool(outputs, tokens["attention_mask"]) if self.use_mean_pool else outputs[:, 0, :]
        embeddings = self.emb_head(pooled)
        return F.normalize(embeddings, dim=-1)

    def _build_pred_mask(self, logits_super_pred: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mask_matrix is None:
            return None
        with torch.no_grad():
            pred_super = logits_super_pred.argmax(dim=-1)
            mask = self.mask_matrix[pred_super]
            allowed = mask > 0.0
            rows_all_zero = (~allowed).all(dim=1)
            if rows_all_zero.any():
                allowed[rows_all_zero] = True
            if self.ban_nd_in_eval and self.nd_id is not None and 0 <= self.nd_id < self.num_cat:
                allowed[:, self.nd_id] = False
        return allowed

    def _build_gold_mask(self, super_labels: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mask_matrix is None:
            return None
        mask = self.mask_matrix[super_labels]
        allowed = mask > 0.0
        rows_all_zero = (~allowed).all(dim=1)
        if rows_all_zero.any():
            allowed[rows_all_zero] = True
        if self.ban_nd_in_eval and self.nd_id is not None and 0 <= self.nd_id < self.num_cat:
            allowed[:, self.nd_id] = False
        return allowed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        super_labels: Optional[torch.Tensor] = None,
        cat_labels: Optional[torch.Tensor] = None,
        property_slot_mask: Optional[torch.Tensor] = None,
        property_presence_labels: Optional[torch.Tensor] = None,
        property_regression_targets: Optional[torch.Tensor] = None,
        property_regression_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.pool(outputs.last_hidden_state, attention_mask) if self.use_mean_pool else outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        emb = self.emb_head(pooled)

        sup_weights = F.normalize(self.super_emb, dim=-1)
        cat_weights = F.normalize(self.cat_emb, dim=-1)
        scale = self.logit_scale
        logits_super = (emb @ sup_weights.t()) * scale
        logits_cat = (emb @ cat_weights.t()) * scale

        very_neg = _very_neg_like(logits_cat)
        pred_mask = self._build_pred_mask(logits_super)
        gold_mask = self._build_gold_mask(super_labels) if super_labels is not None else pred_mask
        logits_cat_pred_masked = torch.where(pred_mask, logits_cat, very_neg) if pred_mask is not None else logits_cat
        logits_cat_gold_masked = torch.where(gold_mask, logits_cat, very_neg) if gold_mask is not None else logits_cat

        loss = None
        if super_labels is not None:
            loss_super = F.cross_entropy(logits_super.float(), super_labels)
            loss = loss_super
            if cat_labels is not None:
                if gold_mask is not None:
                    valid = cat_labels != -100
                    if valid.any():
                        gm = gold_mask.clone()
                        gm[valid, cat_labels[valid]] = True
                        logits_cat_for_loss = torch.where(gm, logits_cat, very_neg)
                    else:
                        logits_cat_for_loss = logits_cat
                else:
                    logits_cat_for_loss = logits_cat
                loss_cat = F.cross_entropy(logits_cat_for_loss.float(), cat_labels, ignore_index=-100)
                loss = loss + loss_cat

        if self.num_properties > 0:
            property_presence_logits = self.property_presence_head(emb)
            property_regression_pred = self.property_regression_head(emb)

            allowed_mask = None
            if self.property_cat_mask is not None:
                if cat_labels is not None:
                    allowed_mask = self.property_cat_mask[cat_labels]
                else:
                    pred_cats = logits_cat_pred_masked.argmax(dim=-1)
                    allowed_mask = self.property_cat_mask[pred_cats]

            if allowed_mask is not None:
                allowed_bool = allowed_mask.to(dtype=torch.bool)
                property_presence_logits_masked = torch.where(
                    allowed_bool, property_presence_logits, _very_neg_like(property_presence_logits)
                )
                zeros = torch.zeros_like(property_regression_pred)
                property_regression_pred_masked = torch.where(allowed_bool, property_regression_pred, zeros)
            else:
                property_presence_logits_masked = property_presence_logits
                property_regression_pred_masked = property_regression_pred

            if self.property_numeric_mask is not None:
                numeric_mask = self.property_numeric_mask.to(property_regression_pred_masked.dtype)
                property_regression_pred_masked = property_regression_pred_masked * numeric_mask

            if property_slot_mask is not None and property_presence_labels is not None:
                mask = property_slot_mask.to(device=property_presence_logits.device, dtype=torch.bool)
                labels_presence = property_presence_labels.to(
                    device=property_presence_logits.device, dtype=torch.float32
                )
                if mask.any():
                    presence_loss = F.binary_cross_entropy_with_logits(
                        property_presence_logits[mask], labels_presence[mask]
                    )
                    weighted_presence = self.property_presence_weight * presence_loss
                    loss = weighted_presence if loss is None else loss + weighted_presence

            if (
                property_regression_targets is not None
                and property_regression_mask is not None
            ):
                reg_mask = property_regression_mask.to(
                    device=property_regression_pred.device, dtype=torch.bool
                )
                targets_reg = property_regression_targets.to(
                    device=property_regression_pred.device, dtype=torch.float32
                )
                if reg_mask.any():
                    regression_loss = F.mse_loss(
                        property_regression_pred[reg_mask], targets_reg[reg_mask]
                    )
                    weighted_reg = self.property_regression_weight * regression_loss
                    loss = weighted_reg if loss is None else loss + weighted_reg
        else:
            property_presence_logits_masked = torch.empty(
                emb.size(0), 0, device=emb.device, dtype=emb.dtype
            )
            property_regression_pred_masked = torch.empty(
                emb.size(0), 0, device=emb.device, dtype=emb.dtype
            )

        return {
            "loss": loss,
            "logits_super": logits_super,
            "logits_cat_pred_masked": logits_cat_pred_masked,
            "logits_cat_gold_masked": logits_cat_gold_masked,
            "property_presence_logits": property_presence_logits_masked,
            "property_regression": property_regression_pred_masked,
            "emb": emb,
            "logits": (
                logits_super,
                logits_cat_pred_masked,
                logits_cat_gold_masked,
                property_presence_logits_masked,
                property_regression_pred_masked,
            ),
        }

    # ---- Hugging Face style save/load helpers ----
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True) -> None:
        import os

        os.makedirs(save_directory, exist_ok=True)
        state = self.state_dict()
        try:
            if safe_serialization:
                from safetensors.torch import save_file

                save_file(state, os.path.join(save_directory, "model.safetensors"))
            else:
                torch.save(state, os.path.join(save_directory, "pytorch_model.bin"))
        except Exception:
            torch.save(state, os.path.join(save_directory, "pytorch_model.bin"))
        cfg = getattr(self, "config", None)
        if cfg is not None and hasattr(cfg, "save_pretrained"):
            cfg.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        config,
        num_super,
        num_cat,
        label_texts_super,
        label_texts_cat,
        tokenizer,
        backbone_src=None,
        **kwargs,
    ) -> "LabelEmbedModel":
        import os

        bb_src = backbone_src or getattr(config, "backbone_src", None)
        if bb_src is None:
            print("[WARN] backbone_src non specificato: uso model_dir come fallback.")
            bb_src = model_dir
        try:
            config.backbone_src = bb_src
        except Exception:
            pass

        model = cls(
            config=config,
            num_super=num_super,
            num_cat=num_cat,
            label_texts_super=label_texts_super,
            label_texts_cat=label_texts_cat,
            tokenizer=tokenizer,
            backbone_src=bb_src,
            **kwargs,
        )

        safetensor_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(safetensor_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensor_path, device="cpu")
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Nessun peso trovato in {model_dir}")

        def _strip_prefix(state, prefix):
            return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

        if any(key.startswith("core.") for key in state_dict.keys()):
            state_dict = _strip_prefix(state_dict, "core.")
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = _strip_prefix(state_dict, "module.")

        info = model.load_state_dict(state_dict, strict=False)
        print(
            f"[from_pretrained] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}"
        )
        return model


def load_label_embed_model(
    model_dir: str,
    *,
    backbone_src: Optional[str] = None,
    tokenizer=None,
    config_overrides: Optional[dict] = None,
    **kwargs,
) -> LabelEmbedModel:
    """Utility to load a model previously exported with :meth:`save_pretrained`."""

    overrides = dict(config_overrides or {})
    overrides.update(kwargs)
    config = AutoConfig.from_pretrained(model_dir, **overrides)
    backbone_src = backbone_src or model_dir
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    mask = getattr(config, "mask_matrix", None)
    mask_tensor = torch.tensor(mask, dtype=torch.float32) if mask is not None else None
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
    return LabelEmbedModel.from_pretrained(
        model_dir,
        config=config,
        num_super=config.num_labels_super,
        num_cat=config.num_labels_cat,
        label_texts_super=config.label_texts_super,
        label_texts_cat=config.label_texts_cat,
        tokenizer=tokenizer,
        backbone_src=backbone_src,
        mask_matrix=mask_tensor,
        ban_nd_in_eval=getattr(config, "ban_nd_in_eval", True),
        nd_id=getattr(config, "nd_id", None),
        num_properties=num_properties,
        property_cat_mask=property_cat_tensor,
        property_numeric_mask=property_numeric_tensor,
        property_presence_weight=presence_weight,
        property_regression_weight=regression_weight,
    )
