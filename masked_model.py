# -*- coding: utf-8 -*-
"""
masked_model.py — versione "a prova di bomba"
---------------------------------------------
- Encoder HF (BERT/RoBERTa/CamemBERT/UmBERTo) + embedding head condivisa.
- Pooling: mean-pooling (default) o CLS.
- Teste: ArcFace (default) o Linear. Margine applicato SOLO a SUPER nella loss
  per evitare problemi con `cat_labels == -100`.
- Mascheramento ontologico per CAT (vista predetta e vista gold).
- Class weights per SUPER e CAT con robusto allineamento di device.
- Uscite retro-compatibili: `logits = [S | C_pred | C_gold]` + `emb`.

Parametri (via AutoConfig o kwargs del costruttore):
- use_mean_pool: bool = True
- proj_dim: int = 256
- use_arcface: bool = True
- arcface_s: float = 30.0
- arcface_m: float = 0.30
- l2_normalize_emb: bool = True
- return_concat_logits: bool = True
- ban_nd_in_eval: bool = True
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

__all__ = ["MultiTaskBERTMasked"]

def _very_neg_like(t):
    # Valore basso ma stabile per CE/softmax in fp32/fp16/bf16
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
            nn.Linear(in_dim, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Linear(hid, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.l2_normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z


class ArcMarginProduct(nn.Module):
    """ArcFace; se labels=None → no margine (usato per pred view)."""
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
        # margine sul target
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
        mask_matrix: torch.Tensor,  # (S, C)
        backbone_src: str | None = None,
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
    ):
        super().__init__()
        if backbone_src is None:
            raise ValueError("Devi passare backbone_src (checkpoint HF).")

        # flags/config
        self.num_super = int(num_super)
        self.num_cat = int(num_cat)
        self.nd_id = nd_id if nd_id is not None else None
        self.ban_nd_in_eval = bool(ban_nd_in_eval)
        self.lambda_cat = float(lambda_cat)
        self.label_smoothing_super = float(label_smoothing_super)
        self.config = config
        self.use_mean_pool = bool(getattr(config, "use_mean_pool", True) if use_mean_pool is None else use_mean_pool)
        self.proj_dim = int(getattr(config, "proj_dim", 256) if proj_dim is None else proj_dim)
        self.use_arcface = bool(getattr(config, "use_arcface", True) if use_arcface is None else use_arcface)
        self.arcface_s = float(getattr(config, "arcface_s", 30.0) if arcface_s is None else arcface_s)
        self.arcface_m = float(getattr(config, "arcface_m", 0.30) if arcface_m is None else arcface_m)
        self.l2_normalize_emb = bool(getattr(config, "l2_normalize_emb", True) if l2_normalize_emb is None else l2_normalize_emb)
        self.return_concat_logits = bool(getattr(config, "return_concat_logits", True) if return_concat_logits is None else return_concat_logits)

        # backbone
        self.backbone = AutoModel.from_pretrained(backbone_src, config=config)
        hidden = getattr(self.backbone.config, "hidden_size", getattr(config, "hidden_size", 768))

        # heads
        self.pool = MeanPool()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.emb_head = EmbHead(hidden, 384, self.proj_dim, l2_normalize=self.l2_normalize_emb)

        if self.use_arcface:
            self.super_head = ArcMarginProduct(self.proj_dim, self.num_super, s=self.arcface_s, m=self.arcface_m)
            self.cat_head = ArcMarginProduct(self.proj_dim, self.num_cat, s=self.arcface_s, m=self.arcface_m)
        else:
            self.super_head = nn.Linear(self.proj_dim, self.num_super)
            self.cat_head = nn.Linear(self.proj_dim, self.num_cat)

        # mask ontologica
        if mask_matrix.dtype != torch.float32:
            mask_matrix = mask_matrix.to(dtype=torch.float32)
        if mask_matrix.shape != (self.num_super, self.num_cat):
            raise ValueError(f"mask_matrix shape errata: atteso ({self.num_super}, {self.num_cat}), trovato {tuple(mask_matrix.shape)}")
        self.register_buffer("mask_matrix", mask_matrix, persistent=False)

        # class weights (non come buffer None; li riallineiamo in forward)
        self.super_class_weights: Optional[torch.Tensor] = None
        self.cat_class_weights: Optional[torch.Tensor] = None

    # set pesi classe
    def set_super_class_weights(self, w: Optional[torch.Tensor]):
        self.super_class_weights = None if w is None else w.to(dtype=torch.float32, device=self.mask_matrix.device)
    
    # set pesi classe
    def set_cat_class_weights(self, w: Optional[torch.Tensor]):
        self.cat_class_weights = None if w is None else w.to(dtype=torch.float32, device=self.mask_matrix.device)
        
    # masks
    def _build_pred_mask(self, logits_super_pred: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred_super = logits_super_pred.argmax(dim=-1)
            mask = self.mask_matrix[pred_super]
            if self.ban_nd_in_eval and self.nd_id is not None and 0 <= self.nd_id < self.num_cat:
                mask[:, self.nd_id] = 0.0
        return (mask > 0.0)

    def _build_gold_mask(self, super_labels: torch.Tensor) -> torch.Tensor:
        mask = self.mask_matrix[super_labels]
        m = (mask > 0.0)
        rows_all_zero = (~m).all(dim=1)
        if rows_all_zero.any():
            m[rows_all_zero] = True
        if self.ban_nd_in_eval and self.nd_id is not None and 0 <= self.nd_id < self.num_cat:
            m[:, self.nd_id] = False
        return m

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        super_labels: Optional[torch.Tensor] = None,
        cat_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # 1) Encoder → pooling → embedding
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.pool(out.last_hidden_state, attention_mask) if self.use_mean_pool else out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        emb = self.emb_head(pooled)  # (B,d)

        # 2) Pred-view logits (senza labels: niente margine)
        logits_super_pred = self.super_head(emb, None) if self.use_arcface else self.super_head(emb)
        logits_cat_pred   = self.cat_head(emb, None)   if self.use_arcface else self.cat_head(emb)

        # 3) Masks
        pred_mask = self._build_pred_mask(logits_super_pred)
        gold_mask = self._build_gold_mask(super_labels) if super_labels is not None else pred_mask

        # 4) Logits mascherati per metriche (usa pred-view)
        very_neg = _very_neg_like(logits_cat_pred)
        logits_cat_pred_masked   = torch.where(pred_mask, logits_cat_pred,   very_neg)
        logits_cat_gold_masked_m = torch.where(gold_mask, logits_cat_pred,   very_neg)

        # 5) Logits per loss (SUPER con margine se labels presenti; CAT senza margine per evitare -100)
        logits_super_loss = self.super_head(emb, super_labels) if (self.use_arcface and super_labels is not None) else logits_super_pred
        logits_cat_loss_src = logits_cat_pred  # niente labels qui per evitare ArcFace + -100
 # --- FIX: non mascherare MAI il target (se presente)
        if cat_labels is not None:
            valid = (cat_labels != -100)
            if valid.any():
                gold_mask = gold_mask.clone()
                gold_mask[valid, cat_labels[valid]] = True
        logits_cat_gold_masked_L = torch.where(gold_mask, logits_cat_loss_src, very_neg)

        # 6) Loss
        loss = None
        if super_labels is not None:
            dev = logits_super_loss.device
            # usa i pesi SOLO in training per evitare NaN quando classi unseen compaiono in eval
            use_weights = self.training
            w_super = self.super_class_weights.to(dev) if (self.super_class_weights is not None and use_weights) else None
            w_cat   = self.cat_class_weights.to(dev)   if (self.cat_class_weights   is not None and use_weights) else None
        
            ls = float(self.label_smoothing_super) if self.label_smoothing_super > 0 else 0.0
            loss_super = F.cross_entropy(logits_super_loss.float(), super_labels, weight=w_super, label_smoothing=ls)
        
            loss_cat = None
            if cat_labels is not None:
                valid = (cat_labels != -100)
                if valid.any():
                    # clamp opzionale per extra stabilità
                    logits_cat_gold_masked_L = torch.clamp(logits_cat_gold_masked_L, min=-1e4, max=1e4)
                    loss_cat = F.cross_entropy(
                        logits_cat_gold_masked_L.float(),
                        cat_labels,
                        weight=w_cat,
                        ignore_index=-100,
                    )
            loss = loss_super if loss_cat is None else (loss_super + self.lambda_cat * loss_cat)

        # 7) Concat retro-compatibile
        if self.return_concat_logits:
            logits = torch.cat([logits_super_pred, logits_cat_pred_masked, logits_cat_gold_masked_m], dim=-1)
        else:
            logits = logits_super_pred

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "logits_super": logits_super_pred,
                "logits_cat_pred_masked": logits_cat_pred_masked,
                "logits_cat_gold_masked": logits_cat_gold_masked_m,
                "emb": emb,
            }
        return (loss, logits)
    # -------------------- HF-like save/load helpers --------------------
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        import os, torch
        os.makedirs(save_directory, exist_ok=True)
        # salva pesi
        state = self.state_dict()
        try:
            if safe_serialization:
                from safetensors.torch import save_file
                save_file(state, os.path.join(save_directory, "model.safetensors"))
            else:
                torch.save(state, os.path.join(save_directory, "pytorch_model.bin"))
        except Exception:
            torch.save(state, os.path.join(save_directory, "pytorch_model.bin"))
        # salva config se presente
        cfg = getattr(self, "config", None)
        if cfg is not None and hasattr(cfg, "save_pretrained"):
            cfg.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls, model_dir: str, *, config, num_super, num_cat, mask_matrix,
        nd_id=None, ban_nd_in_eval=True, backbone_src=None, **kwargs
    ):
        import os, torch
        model = cls(
            config=config,
            num_super=num_super,
            num_cat=num_cat,
            mask_matrix=mask_matrix,
            backbone_src=backbone_src or model_dir,
            nd_id=nd_id,
            ban_nd_in_eval=ban_nd_in_eval,
        )
        sf = os.path.join(model_dir, "model.safetensors")
        pt = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(sf):
            from safetensors.torch import load_file
            sd = load_file(sf, device="cpu")
        elif os.path.isfile(pt):
            sd = torch.load(pt, map_location="cpu")
        else:
            raise FileNotFoundError(f"Nessun peso trovato in {model_dir}")
    
        # ---- strip prefissi comuni per checkpoint del wrapper/DDP
        def _strip_prefix(state, pref):
            return {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
        if any(k.startswith("core.") for k in sd.keys()):
            sd = _strip_prefix(sd, "core.")
        if any(k.startswith("module.") for k in sd.keys()):
            sd = _strip_prefix(sd, "module.")
    
        info = model.load_state_dict(sd, strict=False)
        print(f"[from_pretrained] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
        return model

