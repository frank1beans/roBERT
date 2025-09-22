# -*- coding: utf-8 -*-
"""
label_model.py — Label-Embedding model "a prova di bomba"
- Condivide lo spirito di masked_model.MultiTaskBERTMasked:
  * backbone HF + head di embedding (proiezione + l2 norm)
  * teste "similarità con label-emb" per SUPER e CAT (no ArcFace qui)
  * salva/legge in stile HF, con default a safetensors
- Maschera ontologica opzionale per la CAT (gold/pred view) come per il trainer ontologico.
"""
from __future__ import annotations
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

__all__ = ["LabelEmbedModel"]

def _very_neg_like(t):  # stabile per CE/softmax
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
        return F.normalize(z, p=2, dim=-1) if self.l2_normalize else z

class LabelEmbedModel(nn.Module):
    """
    Backbone HF + proiezione; le "teste" sono semplicemente i **vettori embedding delle label**.
    - super_emb: (S, d)  - cat_emb: (C, d)
    - logits = scale * (emb @ label_emb.T)
    """
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
        mask_matrix: Optional[torch.Tensor] = None,  # (S, C) opzionale
        ban_nd_in_eval: bool = True,
        nd_id: Optional[int] = None,
        freeze_encoder: bool = False,
        train_label_emb: bool = True,
    ):
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

        # backbone
        self.backbone = AutoModel.from_pretrained(backbone_src, config=config)
        hidden = getattr(self.backbone.config, "hidden_size", getattr(config, "hidden_size", 768))

        # heads
        self.pool = MeanPool()
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.emb_head = EmbHead(hidden, 384, self.proj_dim, l2_normalize=self.l2_normalize_emb)

        # scala logit = 1/T
        self.logit_scale = nn.Parameter(torch.tensor(float(1.0 / max(1e-6, self.temperature))))

        # inizializza label-emb da testo (senza grad), poi opz. li rende trainabili
        with torch.no_grad():
            sup_init = self._encode_label_texts(label_texts_super)
            cat_init = self._encode_label_texts(label_texts_cat)

        if train_label_emb:
            self.super_emb = nn.Parameter(sup_init)  # (S,d)
            self.cat_emb   = nn.Parameter(cat_init)  # (C,d)
        else:
            self.register_buffer("super_emb", sup_init)
            self.register_buffer("cat_emb",   cat_init)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # mask
        if mask_matrix is not None:
            if mask_matrix.dtype != torch.float32:
                mask_matrix = mask_matrix.to(dtype=torch.float32)
            if mask_matrix.shape != (self.num_super, self.num_cat):
                raise ValueError(f"mask_matrix shape errata: atteso ({self.num_super}, {self.num_cat}), trovato {tuple(mask_matrix.shape)}")
        self.register_buffer("mask_matrix", mask_matrix, persistent=False)

    @torch.no_grad()
    def _encode_label_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=48, return_tensors="pt").to(self.logit_scale.device)
        out = self.backbone(**toks).last_hidden_state
        pooled = self.pool(out, toks["attention_mask"]) if self.use_mean_pool else out[:, 0, :]
        emb = self.emb_head(pooled)
        return F.normalize(emb, dim=-1)

    def _build_pred_mask(self, logits_super_pred: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mask_matrix is None:
            return None
        with torch.no_grad():
            pred_super = logits_super_pred.argmax(dim=-1)
            mask = self.mask_matrix[pred_super]  # (B, C)
            m = (mask > 0.0)
            # Fallback robusto: se una riga è tutta zero, rendila tutta True (come facciamo nella gold_mask)
            rows_all_zero = (~m).all(dim=1)
            if rows_all_zero.any():
                m[rows_all_zero] = True
            # Opzionale: escludi #N/D solo in eval
            if self.ban_nd_in_eval and self.nd_id is not None and 0 <= self.nd_id < self.num_cat:
                m[:, self.nd_id] = False
        return m

    def _build_gold_mask(self, super_labels: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mask_matrix is None:
            return None
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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        super_labels: Optional[torch.Tensor] = None,
        cat_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # 1) encode → pooling → emb
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.pool(out.last_hidden_state, attention_mask) if self.use_mean_pool else out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        emb = self.emb_head(pooled)  # (B,d)

        # 2) logits vs label-emb
        supW = F.normalize(self.super_emb, dim=-1)
        catW = F.normalize(self.cat_emb,   dim=-1)
        scale = self.logit_scale
        logits_super = (emb @ supW.t()) * scale
        logits_cat   = (emb @ catW.t())   * scale

        # 3) mask views (solo per metriche)
        very_neg = _very_neg_like(logits_cat)
        pred_mask = self._build_pred_mask(logits_super)
        gold_mask = self._build_gold_mask(super_labels) if super_labels is not None else pred_mask
        logits_cat_pred_masked = torch.where(pred_mask, logits_cat, very_neg) if pred_mask is not None else logits_cat
        logits_cat_gold_masked = torch.where(gold_mask, logits_cat, very_neg) if gold_mask is not None else logits_cat

        # 4) loss (super sempre, cat con ignore_index=-100)
        loss = None
        if super_labels is not None:
            loss_super = F.cross_entropy(logits_super.float(), super_labels)
            loss = loss_super
            if cat_labels is not None:
                # garantisci che il target non sia mai mascherato
                if gold_mask is not None:
                    valid = (cat_labels != -100)
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
        out = {
            "loss": loss,
            "logits_super": logits_super,
            "logits_cat_pred_masked": logits_cat_pred_masked,
            "logits_cat_gold_masked": logits_cat_gold_masked,
            "emb": emb,
        }
        return out

    # ---- HF-like save/load con safetensors preferito ----
    def save_pretrained(self, save_directory: str, safe_serialization: bool = True):
        import os, torch
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
        **kwargs
    ):
        import os, torch
    
        # 1) Sorgente backbone: priorità a parametro, poi config, infine fallback (warning)
        bb_src = backbone_src or getattr(config, "backbone_src", None)
        if bb_src is None:
            print("[WARN] backbone_src non specificato: uso model_dir come fallback (potrebbe re-inizializzare l’encoder).")
            bb_src = model_dir
    
        # 2) Persisti nel config (così viene salvato nel prossimo save_pretrained)
        try:
            config.backbone_src = bb_src
        except Exception:
            pass
    
        # 3) Costruisci il modello passando il backbone corretto
        model = cls(
            config=config,
            num_super=num_super,
            num_cat=num_cat,
            label_texts_super=label_texts_super,
            label_texts_cat=label_texts_cat,
            tokenizer=tokenizer,
            backbone_src=bb_src,
            **kwargs
        )
    
        # 4) Carica i pesi del modello label-embed
        sf = os.path.join(model_dir, "model.safetensors")
        pt = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.isfile(sf):
            from safetensors.torch import load_file
            sd = load_file(sf, device="cpu")
        elif os.path.isfile(pt):
            sd = torch.load(pt, map_location="cpu")
        else:
            raise FileNotFoundError(f"Nessun peso trovato in {model_dir}")
    
        # strip prefissi comuni (DDP/wrapper)
        def _strip_prefix(state, pref):
            return {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
        if any(k.startswith("core.") for k in sd.keys()):
            sd = _strip_prefix(sd, "core.")
        if any(k.startswith("module.") for k in sd.keys()):
            sd = _strip_prefix(sd, "module.")
    
        info = model.load_state_dict(sd, strict=False)
        print(f"[from_pretrained] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
        return model

