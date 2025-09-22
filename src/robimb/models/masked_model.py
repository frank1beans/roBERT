"""Hierarchical multi-task classifier with ontology masking and ArcFace support."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

__all__ = ["MultiTaskBERTMasked", "ArcMarginProduct", "load_masked_model"]


def _very_neg_like(t: torch.Tensor) -> torch.Tensor:
    return torch.tensor(-1e4, dtype=t.dtype, device=t.device)


class MeanPool(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        s = (last_hidden_state * m).sum(dim=1)
        d = m.sum(dim=1).clamp(min=1e-6)
        return s / d


class EmbHead(nn.Module):
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
        if self.l2_normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z


class ArcMarginProduct(nn.Module):
    """ArcFace; margin applied only when labels are provided."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.30):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = F.normalize(self.W, p=2, dim=1)
        cosine = F.linear(emb, W)
        if labels is None:
            return cosine * self.s
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=W.size(0)).to(cosine.dtype)
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return logits * self.s


class MultiTaskBERTMasked(nn.Module):
    def __init__(
        self,
        config: AutoConfig,
        num_super: int,
        num_cat: int,
        mask_matrix: torch.Tensor,
        backbone_src: str,
        nd_id: int | None = None,
        ban_nd_in_eval: bool = True,
        lambda_cat: float = 1.0,
        label_smoothing_super: float = 0.0,
        use_mean_pool: bool | None = None,
        proj_dim: int | None = None,
        use_arcface: bool | None = None,
        arcface_s: float | None = None,
        arcface_m: float | None = None,
        l2_normalize_emb: bool | None = None,
        return_concat_logits: bool | None = None,
        num_properties: int = 0,
        property_cat_mask: torch.Tensor | None = None,
        property_numeric_mask: torch.Tensor | None = None,
        property_presence_weight: float = 1.0,
        property_regression_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not backbone_src:
            raise ValueError("A backbone checkpoint must be provided.")

        self.num_super = int(num_super)
        self.num_cat = int(num_cat)
        self.nd_id = nd_id
        self.ban_nd_in_eval = bool(ban_nd_in_eval)
        self.lambda_cat = float(lambda_cat)
        self.label_smoothing_super = float(label_smoothing_super)
        self.config = config
        self.use_mean_pool = bool(getattr(config, "use_mean_pool", True) if use_mean_pool is None else use_mean_pool)
        self.proj_dim = int(getattr(config, "proj_dim", 256) if proj_dim is None else proj_dim)
        self.use_arcface = bool(getattr(config, "use_arcface", True) if use_arcface is None else use_arcface)
        self.arcface_s = float(getattr(config, "arcface_s", 30.0) if arcface_s is None else arcface_s)
        self.arcface_m = float(getattr(config, "arcface_m", 0.30) if arcface_m is None else arcface_m)
        self.l2_normalize_emb = bool(
            getattr(config, "l2_normalize_emb", True) if l2_normalize_emb is None else l2_normalize_emb
        )
        self.return_concat_logits = bool(
            getattr(config, "return_concat_logits", True) if return_concat_logits is None else return_concat_logits
        )
        self.num_properties = int(num_properties or 0)
        self.property_presence_weight = float(property_presence_weight)
        self.property_regression_weight = float(property_regression_weight)

        self.backbone = AutoModel.from_pretrained(backbone_src, config=config)
        hidden = getattr(self.backbone.config, "hidden_size", getattr(config, "hidden_size", 768))

        self.pool = MeanPool()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.emb_head = EmbHead(hidden, 384, self.proj_dim, l2_normalize=self.l2_normalize_emb)

        if self.use_arcface:
            self.super_head = ArcMarginProduct(self.proj_dim, self.num_super, s=self.arcface_s, m=self.arcface_m)
            self.cat_head = ArcMarginProduct(self.proj_dim, self.num_cat, s=self.arcface_s, m=self.arcface_m)
        else:
            self.super_head = nn.Linear(self.proj_dim, self.num_super)
            self.cat_head = nn.Linear(self.proj_dim, self.num_cat)

        if mask_matrix.dtype != torch.float32:
            mask_matrix = mask_matrix.to(dtype=torch.float32)
        if mask_matrix.shape != (self.num_super, self.num_cat):
            raise ValueError(
                f"mask_matrix shape errata: atteso ({self.num_super}, {self.num_cat}), trovato {tuple(mask_matrix.shape)}"
            )
        self.register_buffer("mask_matrix", mask_matrix, persistent=False)

        self.super_class_weights: Optional[torch.Tensor] = None
        self.cat_class_weights: Optional[torch.Tensor] = None

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

    def set_super_class_weights(self, weights: Optional[torch.Tensor]) -> None:
        self.super_class_weights = None if weights is None else weights.to(dtype=torch.float32, device=self.mask_matrix.device)

    def set_cat_class_weights(self, weights: Optional[torch.Tensor]) -> None:
        self.cat_class_weights = None if weights is None else weights.to(dtype=torch.float32, device=self.mask_matrix.device)

    def _build_pred_mask(self, logits_super_pred: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred_super = logits_super_pred.argmax(dim=-1)
            mask = self.mask_matrix[pred_super]
            if self.ban_nd_in_eval and self.nd_id is not None and 0 <= self.nd_id < self.num_cat:
                mask[:, self.nd_id] = 0.0
        return mask > 0.0

    def _build_gold_mask(self, super_labels: torch.Tensor) -> torch.Tensor:
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
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        super_labels: Optional[torch.Tensor] = None,
        cat_labels: Optional[torch.Tensor] = None,
        property_slot_mask: Optional[torch.Tensor] = None,
        property_presence_labels: Optional[torch.Tensor] = None,
        property_regression_targets: Optional[torch.Tensor] = None,
        property_regression_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.pool(out.last_hidden_state, attention_mask) if self.use_mean_pool else out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        emb = self.emb_head(pooled)

        logits_super_pred = self.super_head(emb, None) if self.use_arcface else self.super_head(emb)
        logits_cat_pred = self.cat_head(emb, None) if self.use_arcface else self.cat_head(emb)

        pred_mask = self._build_pred_mask(logits_super_pred)
        gold_mask = self._build_gold_mask(super_labels) if super_labels is not None else pred_mask

        very_neg = _very_neg_like(logits_cat_pred)
        logits_cat_pred_masked = torch.where(pred_mask, logits_cat_pred, very_neg)
        logits_cat_gold_masked = torch.where(gold_mask, logits_cat_pred, very_neg)

        logits_super_loss = (
            self.super_head(emb, super_labels) if (self.use_arcface and super_labels is not None) else logits_super_pred
        )
        logits_cat_loss_src = logits_cat_pred
        if cat_labels is not None:
            valid = cat_labels != -100
            if valid.any():
                gold_mask = gold_mask.clone()
                gold_mask[valid, cat_labels[valid]] = True
        logits_cat_gold_for_loss = torch.where(gold_mask, logits_cat_loss_src, very_neg)

        loss = None
        if super_labels is not None:
            dev = logits_super_loss.device
            use_weights = self.training
            w_super = self.super_class_weights.to(dev) if (self.super_class_weights is not None and use_weights) else None
            w_cat = self.cat_class_weights.to(dev) if (self.cat_class_weights is not None and use_weights) else None

            ls = float(self.label_smoothing_super) if self.label_smoothing_super > 0 else 0.0
            loss_super = F.cross_entropy(
                logits_super_loss.float(), super_labels, weight=w_super, label_smoothing=ls
            )

            loss_cat = None
            if cat_labels is not None:
                valid = cat_labels != -100
                if valid.any():
                    logits_cat_gold_for_loss = torch.clamp(logits_cat_gold_for_loss, min=-1e4, max=1e4)
                    loss_cat = F.cross_entropy(
                        logits_cat_gold_for_loss.float(),
                        cat_labels,
                        weight=w_cat,
                        ignore_index=-100,
                    )
            loss = loss_super if loss_cat is None else (loss_super + self.lambda_cat * loss_cat)

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

        if self.return_concat_logits:
            concat_logits = torch.cat(
                [logits_super_pred, logits_cat_pred_masked, logits_cat_gold_masked], dim=-1
            )
        else:
            concat_logits = logits_super_pred

        if return_dict:
            return {
                "loss": loss,
                "logits": (
                    logits_super_pred,
                    logits_cat_pred_masked,
                    logits_cat_gold_masked,
                    property_presence_logits_masked,
                    property_regression_pred_masked,
                ),
                "logits_concat": concat_logits,
                "logits_super": logits_super_pred,
                "logits_cat_pred_masked": logits_cat_pred_masked,
                "logits_cat_gold_masked": logits_cat_gold_masked,
                "property_presence_logits": property_presence_logits_masked,
                "property_regression": property_regression_pred_masked,
                "emb": emb,
            }
        return loss, concat_logits

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
        mask_matrix,
        nd_id=None,
        ban_nd_in_eval=True,
        backbone_src=None,
        **kwargs,
    ) -> "MultiTaskBERTMasked":
        import os

        model = cls(
            config=config,
            num_super=num_super,
            num_cat=num_cat,
            mask_matrix=mask_matrix,
            backbone_src=backbone_src or model_dir,
            nd_id=nd_id,
            ban_nd_in_eval=ban_nd_in_eval,
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

        if any(k.startswith("core.") for k in state_dict.keys()):
            state_dict = _strip_prefix(state_dict, "core.")
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = _strip_prefix(state_dict, "module.")

        info = model.load_state_dict(state_dict, strict=False)
        print(f"[from_pretrained] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
        return model


def load_masked_model(model_dir: str, **kwargs) -> MultiTaskBERTMasked:
    config = AutoConfig.from_pretrained(model_dir, **kwargs)
    mask_matrix = torch.tensor(config.mask_matrix, dtype=torch.float32)
    cat_mask = getattr(config, "property_cat_mask", None)
    property_cat_tensor = (
        torch.tensor(cat_mask, dtype=torch.bool) if cat_mask is not None else None
    )
    numeric_mask = getattr(config, "property_numeric_mask", None)
    property_numeric_tensor = (
        torch.tensor(numeric_mask, dtype=torch.bool) if numeric_mask is not None else None
    )
    return MultiTaskBERTMasked.from_pretrained(
        model_dir,
        config=config,
        num_super=config.num_labels_super,
        num_cat=config.num_labels_cat,
        mask_matrix=mask_matrix,
        nd_id=getattr(config, "nd_id", None),
        ban_nd_in_eval=getattr(config, "ban_nd_in_eval", True),
        backbone_src=getattr(config, "backbone_src", None),
        lambda_cat=getattr(config, "lambda_cat", 1.0),
        label_smoothing_super=getattr(config, "label_smoothing_super", 0.0),
        use_mean_pool=getattr(config, "use_mean_pool", True),
        proj_dim=getattr(config, "proj_dim", 256),
        use_arcface=getattr(config, "use_arcface", True),
        arcface_s=getattr(config, "arcface_s", 30.0),
        arcface_m=getattr(config, "arcface_m", 0.30),
        l2_normalize_emb=getattr(config, "l2_normalize_emb", True),
        return_concat_logits=getattr(config, "return_concat_logits", True),
        num_properties=getattr(config, "num_properties", 0),
        property_cat_mask=property_cat_tensor,
        property_numeric_mask=property_numeric_tensor,
        property_presence_weight=getattr(config, "property_presence_weight", 1.0),
        property_regression_weight=getattr(config, "property_regression_weight", 1.0),
    )
