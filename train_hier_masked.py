# -*- coding: utf-8 -*-
"""
Trainer per MultiTaskBERTMasked con **staging progressivo** e **export pubblicabile**.

Novità principali:
- `--init_from` per ripartire **interamente** da uno stage precedente (backbone + emb + heads).
- `--rebuild_super_head/--rebuild_cat_head` per cambiare SOLO le teste (Linear/ArcFace) tra gli stage.
- Export pulito in `export/` con: pesi `.safetensors`, config arricchita (id2label/label2id, mask, mapping), tokenizer,
  ontologia/label_maps, **README.md** (model card), e **predict.py**.
- Opzionale: **push su Hugging Face Hub** con `--publish_hub --hub_repo <org/name>`.
- Metriche robuste, CE-cat con `ignore_index=-100`, bilanciamento opzionale, class weights (effective number).

Compatibile con `masked_model.MultiTaskBERTMasked`.
"""
import os, json, argparse, random, glob, shutil, textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding
)
from torch.optim import AdamW
from transformers import TrainerCallback

from robert.data.ontology import load_ontology, load_label_maps, build_mask
from robert.models.masked import MultiTaskBERTMasked, ArcMarginProduct

# -------------------------- Utils --------------------------
def pick_latest_checkpoint(run_dir: str) -> str | None:
    pats = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not pats:
        return None
    pats.sort(key=lambda p: int(p.rsplit("-", 1)[1]))
    return pats[-1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_to_df(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def ensure_has_weights(path: str):
    ok = any(os.path.isfile(os.path.join(path, f)) for f in ("model.safetensors", "pytorch_model.bin"))
    if not ok:
        raise SystemExit(f"[ERRORE] '{path}' non contiene pesi (model.safetensors/pytorch_model.bin).")


class SanitizeGrads(TrainerCallback):
    """Azzera eventuali gradienti non finiti per evitare NaN-ception."""
    def on_after_backward(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        fixed = 0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    fixed += 1
        if fixed:
            print(f"[SAN] Riparati {fixed} gradienti NaN/Inf")


# ---------- Metrics (Trainer) ----------
def make_compute_metrics(num_super: int, num_cat: int):
    def _compute(eval_pred):
        preds = getattr(eval_pred, "predictions", eval_pred)
        labels = getattr(eval_pred, "label_ids", None)

        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        logits = np.asarray(preds)
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if isinstance(labels, dict):
            y_s = np.asarray(labels["super_labels"]) ; y_c = np.asarray(labels["cat_labels"])
        elif isinstance(labels, (tuple, list)):
            y_s = np.asarray(labels[0]); y_c = np.asarray(labels[1])
        else:
            labels = np.asarray(labels)
            assert labels.ndim == 2 and labels.shape[1] >= 2, f"Label IDs formato inatteso: {labels.shape}"
            y_s, y_c = labels[:, 0], labels[:, 1]

        S = num_super; C = num_cat
        assert logits.shape[1] == S + 2*C, f"atteso (N, S+2C) = {S+2*C}, trovato {logits.shape}"

        logits_s      = logits[:, :S]
        logits_c_pred = logits[:, S:S + C]
        logits_c_gold = logits[:, S + C:S + 2*C]

        pred_s = logits_s.argmax(-1)
        pred_c_pred_super = logits_c_pred.argmax(-1)
        pred_c_gold_super = logits_c_gold.argmax(-1)

        m = (y_c != -100)
        acc_cat_pred_super = accuracy_score(y_c[m], pred_c_pred_super[m]) if m.any() else float("nan")
        f1_cat_pred_super  = f1_score(y_c[m], pred_c_pred_super[m], average="macro") if m.any() else float("nan")
        acc_cat_gold_super = accuracy_score(y_c[m], pred_c_gold_super[m]) if m.any() else float("nan")
        f1_cat_gold_super  = f1_score(y_c[m], pred_c_gold_super[m], average="macro") if m.any() else float("nan")

        mc = m & (pred_s == y_s)
        acc_cat_cond = accuracy_score(y_c[mc], pred_c_gold_super[mc]) if mc.any() else float("nan")
        f1_cat_cond  = f1_score(y_c[mc], pred_c_gold_super[mc], average="macro") if mc.any() else float("nan")

        return {
            "acc_super": accuracy_score(y_s, pred_s),
            "macro_f1_super": f1_score(y_s, pred_s, average="macro"),
            "acc_cat_pred_super": acc_cat_pred_super,
            "macro_f1_cat_pred_super": f1_cat_pred_super,
            "acc_cat_given_super": acc_cat_cond,
            "macro_f1_cat_given_super": f1_cat_cond,
            "acc_cat_gold_super": acc_cat_gold_super,
            "macro_f1_cat_gold_super": f1_cat_gold_super,
        }
    return _compute

# ---------- SupCon opzionale ----------
class HierSupConLoss(nn.Module):
    def __init__(self, tau=0.07, w_same_cat=1.0, w_same_super=0.5):
        super().__init__()
        self.tau = tau
        self.w_same_cat = w_same_cat
        self.w_same_super = w_same_super
    def forward(self, z, y_super, y_cat):
        sim = (z @ z.t()) / self.tau
        B = z.size(0)
        eye = torch.eye(B, device=z.device, dtype=torch.bool)
        same_cat   = (y_cat.unsqueeze(1) == y_cat.unsqueeze(0))
        same_super = (y_super.unsqueeze(1) == y_super.unsqueeze(0))
        pos  = same_cat & ~eye
        weak = same_super & ~same_cat & ~eye
        sim = sim - sim.max(dim=1, keepdim=True).values
        exp = sim.exp()
        denom = (exp * (~eye)).sum(dim=1, keepdim=True).clamp_min(1e-9)
        log_prob = sim - torch.log(denom)
        w = pos.float() * self.w_same_cat + weak.float() * self.w_same_super
        denom_pos = w.sum(dim=1).clamp_min(1e-6)
        loss_i = -(log_prob * w).sum(dim=1) / denom_pos
        return loss_i.mean()


# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--backbone_src", default=None)
    ap.add_argument("--tokenizer_src", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default=None)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--label_maps", required=True)
    ap.add_argument("--out_dir", required=True)

    # staging progressivo
    ap.add_argument("--init_from", type=str, default=None,
                    help="Cartella di un modello precedente (tipicamente out/.../export) da cui caricare TUTTO")
    ap.add_argument("--rebuild_super_head", choices=["none","linear","arcface"], default="none",
                    help="Ricostruisce la testa SUPER (default: none → conserva quella caricata)")
    ap.add_argument("--rebuild_cat_head", choices=["none","linear","arcface"], default="none",
                    help="Ricostruisce la testa CAT (default: none → conserva quella caricata)")

    # training knobs
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--unfreeze_last", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # multitask knobs
    ap.add_argument("--lambda_cat", type=float, default=1.3)
    ap.add_argument("--label_smoothing_super", type=float, default=0.02)

    # embedding/arcface flags
    ap.add_argument("--use_arcface", type=lambda x: str(x).lower() != "false", default=True)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--arcface_s", type=float, default=30.0)
    ap.add_argument("--arcface_m", type=float, default=0.30)
    ap.add_argument("--use_mean_pool", type=lambda x: str(x).lower() != "false", default=True)

    # contrastive opzionale
    ap.add_argument("--lambda_supcon", type=float, default=0.0)
    ap.add_argument("--supcon_tau", type=float, default=0.07)
    ap.add_argument("--balanced_sampler", type=lambda x: str(x).lower() != "false", default=False)

    # scheduler
    ap.add_argument("--scheduler", choices=["linear", "cosine"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.08)

    # resume
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_from", type=str, default=None)

    # export/publish
    ap.add_argument("--best_metric", type=str, default="eval_macro_f1_cat_pred_super",
                    help="Metrica per load_best_model_at_end")
    ap.add_argument("--publish_hub", action="store_true")
    ap.add_argument("--hub_repo", type=str, default=None, help="es. org/name")
    ap.add_argument("--hub_private", action="store_true")

    args = ap.parse_args()

    # ---------- Resume ----------
    resume_ckpt = None
    if args.resume_from:
        cpath = args.resume_from
        if cpath == "latest":
            last = pick_latest_checkpoint(args.out_dir)
            if last:
                print(f"[RESUME] ultimo checkpoint: {last}")
                resume_ckpt = last
        else:
            resume_ckpt = cpath
    elif args.resume:
        last = pick_latest_checkpoint(args.out_dir)
        if last:
            print(f"[RESUME] ultimo checkpoint: {last}")
            resume_ckpt = last

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # ---------- Ontologia & Label maps ----------
    ontology = load_ontology(args.ontology)
    super_name_to_id, cat_name_to_id, id_to_super_name, id_to_cat_name = load_label_maps(args.label_maps)

    mask_matrix, report = build_mask(
        ontology,
        super_name_to_id,
        cat_name_to_id,
        return_report=True,
    )
    with open(os.path.join(args.out_dir, "mask_report.json"), "w", encoding="utf-8") as w:
        json.dump(report, w, ensure_ascii=False, indent=2)

    num_super = max(super_name_to_id.values()) + 1
    num_cat   = max(cat_name_to_id.values()) + 1
    nd_id     = cat_name_to_id.get("#N/D", None)

    # ---------- Dataset mapping ----------
    def map_row(row):
        sname = row["super"]; cname = row["cat"]
        sid = super_name_to_id.get(sname, None)
        cid = cat_name_to_id.get(cname, None)
        if sid is None:
            return None
        cat_lab = -100
        if cid is not None and mask_matrix[sid][cid] == 1:
            cat_lab = cid
        return {"text": row["text"], "super_label": sid, "cat_label": cat_lab}

    def load_df(jsonl_path):
        df0 = load_jsonl_to_df(jsonl_path)
        mapped = [m for m in df0.apply(map_row, axis=1).tolist() if m is not None]
        df = pd.DataFrame(mapped)
        df = df[df["cat_label"].notna()]
        df["super_label"] = df["super_label"].astype(int)
        df["cat_label"]   = df["cat_label"].astype(int)
        return df

    train_df = load_df(args.train_jsonl)
    val_df = load_df(args.val_jsonl) if args.val_jsonl and os.path.isfile(args.val_jsonl) else train_df.iloc[0:0].copy()

    # ---------- Tokenizer ----------
    tok_src = args.tokenizer_src or args.backbone_src or args.base_model
    try:
        tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    except Exception as e:
        print("[WARN] tokenizer fast KO → fallback slow:", e)
        tok = AutoTokenizer.from_pretrained(tok_src, use_fast=False)

    def to_hfds(d: pd.DataFrame):
        return Dataset.from_dict({
            "text": d["text"].tolist(),
            "super_label": d["super_label"].tolist(),
            "cat_label": d["cat_label"].tolist(),
        })

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length, return_token_type_ids=False)

    train_ds = to_hfds(train_df).map(tokenize, batched=True, remove_columns=["text"])
    eval_ds  = to_hfds(val_df).map(tokenize, batched=True, remove_columns=["text"]) if len(val_df) else None

    do_eval = eval_ds is not None and len(eval_ds) > 0

    # ---------- Model ----------
    bb_src = args.backbone_src or args.base_model
    ensure_has_weights(bb_src)

    cfg = AutoConfig.from_pretrained(bb_src)
    cfg.num_labels_super = num_super
    cfg.num_labels_cat   = num_cat
    cfg.use_mean_pool    = bool(args.use_mean_pool)
    cfg.proj_dim         = int(args.proj_dim)
    cfg.use_arcface      = bool(args.use_arcface)
    cfg.arcface_s        = float(args.arcface_s)
    cfg.arcface_m        = float(args.arcface_m)

    mask_tensor = torch.tensor(mask_matrix, dtype=torch.float32)

    if args.init_from:
        print(f"[INIT] Carico interamente da: {args.init_from}")
        model = MultiTaskBERTMasked.from_pretrained(
            args.init_from,
            config=cfg,
            num_super=num_super,
            num_cat=num_cat,
            mask_matrix=mask_tensor,
            nd_id=nd_id,
            ban_nd_in_eval=True,
            backbone_src=bb_src,
        )
    else:
        print(f"[INIT] Inizializzo da backbone HF: {bb_src}")
        model = MultiTaskBERTMasked(
            config=cfg,
            num_super=num_super,
            num_cat=num_cat,
            mask_matrix=mask_tensor,
            backbone_src=bb_src,
            nd_id=nd_id,
            ban_nd_in_eval=True,
            lambda_cat=float(args.lambda_cat),
            label_smoothing_super=float(args.label_smoothing_super),
        )

    # --- opzionale: rimpiazza heads se richiesto dallo stage ---
    def _make_head(use_arc: bool, dim: int, ncls: int, s: float, m: float):
        if use_arc:
            return ArcMarginProduct(dim, ncls, s=s, m=m)
        else:
            return nn.Linear(dim, ncls)

    if args.rebuild_super_head != "none":
        want_arc = (args.rebuild_super_head == "arcface")
        model.super_head = _make_head(want_arc, cfg.proj_dim, num_super, cfg.arcface_s, cfg.arcface_m)
        print(f"[HEAD] SUPER ricostruita → {'ArcFace' if want_arc else 'Linear'}")

    if args.rebuild_cat_head != "none":
        want_arc = (args.rebuild_cat_head == "arcface")
        model.cat_head = _make_head(want_arc, cfg.proj_dim, num_cat, cfg.arcface_s, cfg.arcface_m)
        print(f"[HEAD] CAT ricostruita → {'ArcFace' if want_arc else 'Linear'}")

    # --- Pesi di classe (effective number) ---
    def effective_num_weights(labels, num_classes, beta=0.999):
        counts = np.bincount(labels.astype(int), minlength=num_classes)
        eff_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.clip(eff_num, 1e-9, None)
        weights[counts == 0] = 0.0
        weights = weights / (weights.mean() + 1e-9)
        return torch.tensor(weights, dtype=torch.float32)

    # SUPER
    w_super = effective_num_weights(train_df["super_label"].values, num_super)
    model.set_super_class_weights(w_super)
    # CAT
    vc = train_df.loc[train_df["cat_label"] != -100, "cat_label"].values
    if vc.size > 0:
        w_cat = effective_num_weights(vc, num_cat)
        model.set_cat_class_weights(w_cat)

    # --- Freeze / unfreeze PRIMA dell’optimizer ---
    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad = False
    if args.unfreeze_last > 0 and hasattr(bb, "encoder") and hasattr(bb.encoder, "layer"):
        for layer in bb.encoder.layer[-args.unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad = True
    print(f"[INFO] sbloccati gli ultimi {args.unfreeze_last} layer dell'encoder." if args.unfreeze_last > 0 else "[INFO] encoder congelato")

    # --- Optimizer (param groups) ---
    head_params = list(model.emb_head.parameters()) + list(model.super_head.parameters()) + list(model.cat_head.parameters())
    enc_params  = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": head_params, "lr": args.lr_head},
        {"params": enc_params,  "lr": args.lr_encoder},
    ], weight_decay=0.01)

    # ---- Sampler bilanciato opzionale ----
    collator = DataCollatorWithPadding(tokenizer=tok, padding=True)

    class DS(torch.utils.data.Dataset):
        def __init__(self, hfds): self.ds = hfds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            item = {k: self.ds[idx][k] for k in self.ds.column_names if k not in ("super_label", "cat_label")}
            item["super_labels"] = torch.tensor(self.ds[idx]["super_label"], dtype=torch.long)
            item["cat_labels"]   = torch.tensor(self.ds[idx]["cat_label"],   dtype=torch.long)
            return item

    train_wrapped = DS(train_ds)
    eval_wrapped  = DS(eval_ds) if do_eval else None

    train_sampler = None
    if args.balanced_sampler:
        labels = train_df["cat_label"].values
        by_cat = {}
        for i, c in enumerate(labels):
            by_cat.setdefault(int(c), []).append(i)
        pools = [idxs for idxs in by_cat.values() if len(idxs) >= 2]
        if pools:
            class BalancedSampler(torch.utils.data.Sampler):
                def __init__(self, pools, batch_size, n_samples):
                    self.pools = [np.array(p) for p in pools]
                    self.batch = batch_size
                    self.n = n_samples
                def __iter__(self):
                    rng = np.random.default_rng()
                    buf = []
                    produced = 0
                    while produced < self.n:
                        rng.shuffle(self.pools)
                        for pool in self.pools:
                            if len(pool) < 2:
                                continue
                            pick = rng.choice(pool, size=min(2, len(pool)), replace=False).tolist()
                            buf.extend(pick)
                            if len(buf) >= self.batch:
                                batch = buf[:self.batch]
                                buf = buf[self.batch:]
                                for i in batch:
                                    yield i
                                    produced += 1
                                    if produced >= self.n:
                                        return
                    for i in buf:
                        yield i
                def __len__(self):
                    return self.n
            train_sampler = BalancedSampler(pools, args.batch_size, len(train_wrapped))
        else:
            print("[WARN] balanced_sampler richiesto ma nessuna cat con >=2 esempi. Uso sampler default.")

    # ---- Wrap per Trainer (aggiunge SupCon e concat logits) ----
    class HierSupConWrap(nn.Module):
        def __init__(self, core, lambda_supcon=0.0, supcon_tau=0.07):
            super().__init__()
            self.core = core
            self.lambda_supcon = float(lambda_supcon)
            self.supcon = HierSupConLoss(tau=supcon_tau) if self.lambda_supcon > 0 else None
        def forward(self, **batch):
            out = self.core(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids", None),
                super_labels=batch.get("super_labels"),
                cat_labels=batch.get("cat_labels"),
                return_dict=True,
            )
            loss = out.get("loss", None)
            if self.lambda_supcon > 0 and (batch.get("super_labels") is not None) and (batch.get("cat_labels") is not None):
                z = out["emb"]; y_s = batch["super_labels"]; y_c = batch["cat_labels"]
                mask = (y_c != -100)
                if mask.any():
                    loss_con = self.supcon(z[mask], y_s[mask], y_c[mask])
                    loss = loss + self.lambda_supcon * loss_con if loss is not None else self.lambda_supcon * loss_con
            ls = out["logits_super"]
            lc = out.get("logits_cat_pred_masked", out.get("logits_cat"))
            lg = out.get("logits_cat_gold_masked", lc)
            logits_concat = torch.cat([ls, lc, lg], dim=-1)
            return (loss, logits_concat) if loss is not None else logits_concat

    wrap = HierSupConWrap(model, lambda_supcon=args.lambda_supcon, supcon_tau=args.supcon_tau)

    # ---------- TrainingArguments ----------
    tr_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(64, args.batch_size),
        num_train_epochs=args.epochs,
        learning_rate=args.lr_head,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        logging_steps=50,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=False,
        report_to=["none"],
        seed=args.seed,
        remove_unused_columns=False,
        save_total_limit=2,
        max_grad_norm=1.0,
        eval_strategy=("epoch" if do_eval else "no"),
        save_strategy=("epoch" if do_eval else "no"),
        load_best_model_at_end=bool(do_eval),
        metric_for_best_model=(args.best_metric if do_eval else None),
        greater_is_better=True if do_eval else None,
        label_names=["super_labels", "cat_labels"],
    )

    trainer = Trainer(
        model=wrap,
        args=tr_args,
        train_dataset=train_wrapped,
        eval_dataset=eval_wrapped if do_eval else None,
        compute_metrics=make_compute_metrics(num_super, num_cat) if do_eval else None,
        data_collator=collator,
        tokenizer=tok,
        optimizers=(optimizer, None),
        callbacks=[SanitizeGrads()],
    )

    if train_sampler is not None:
        trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
            train_wrapped,
            batch_size=tr_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=collator,
        )

    # ---------- Train ----------
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    # ---------- Save (checkpoint finale + clean core) ----------
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    clean_dir = os.path.join(args.out_dir, "model_clean")
    os.makedirs(clean_dir, exist_ok=True)
    wrap.core.save_pretrained(clean_dir, safe_serialization=True)

    # ---------- Eval ----------
    final_metrics = {}
    if do_eval:
        final_metrics = trainer.evaluate(eval_dataset=eval_wrapped)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as w:
        json.dump(final_metrics, w, ensure_ascii=False, indent=2)

    # ---------- EXPORT DEPLOY-READY / PUBBLICABILE ----------
    export_dir = os.path.join(args.out_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    # arricchisci config del core prima di salvare
    cfg = model.config
    cfg.num_labels_super = int(getattr(cfg, "num_labels_super", 0) or 0) or int(num_super)
    cfg.num_labels_cat   = int(getattr(cfg, "num_labels_cat", 0) or 0)   or int(num_cat)
    cfg.nd_id            = None if nd_id is None else int(nd_id)
    cfg.use_mean_pool    = bool(getattr(cfg, "use_mean_pool", True))
    cfg.proj_dim         = int(getattr(cfg, "proj_dim", 256))
    cfg.use_arcface      = bool(getattr(cfg, "use_arcface", True))
    cfg.arcface_s        = float(getattr(cfg, "arcface_s", 30.0))
    cfg.arcface_m        = float(getattr(cfg, "arcface_m", 0.30))
    cfg.mask_matrix = np.array(mask_matrix).tolist()
    cfg.super2id = {k: int(v) for k, v in (getattr(cfg, "super2id", {}) or {}).items()} or {k: int(v) for k, v in super_name_to_id.items()}
    cfg.cat2id   = {k: int(v) for k, v in (getattr(cfg, "cat2id", {}) or {}).items()}   or {k: int(v) for k, v in cat_name_to_id.items()}
    cfg.id2super = {int(v): k for k, v in cfg.super2id.items()}
    cfg.id2cat   = {int(v): k for k, v in cfg.cat2id.items()}

    # salva core + tokenizer
    model.save_pretrained(export_dir, safe_serialization=True)
    tok.save_pretrained(export_dir)

    # copia artefatti ontologia/label maps
    for src, name in [
        (args.label_maps, "label_maps.json"),
        (args.ontology, "ontology.json"),
        (os.path.join(args.out_dir, "mask_report.json"), "mask_report.json"),
    ]:
        try:
            if src and os.path.isfile(src):
                shutil.copy2(src, os.path.join(export_dir, name))
        except Exception as e:
            print(f"[WARN] copia {name} fallita:", e)

    # genera predict.py minimale
    predict_py = os.path.join(export_dir, "predict.py")
    with open(predict_py, "w", encoding="utf-8") as w:
        w.write(textwrap.dedent(f"""
        #!/usr/bin/env python3
        import json, torch
        from transformers import AutoTokenizer, AutoConfig
        from masked_model import MultiTaskBERTMasked
        import numpy as np
        tok = AutoTokenizer.from_pretrained('.')
        cfg = AutoConfig.from_pretrained('.')
        mask = torch.tensor(np.array(cfg.mask_matrix), dtype=torch.float32)
        model = MultiTaskBERTMasked.from_pretrained('.', config=cfg,
            num_super=cfg.num_labels_super, num_cat=cfg.num_labels_cat, mask_matrix=mask)
        model.eval()
        def predict(texts):
            enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
            with torch.inference_mode():
                out = model(**enc, return_dict=True)
                S = cfg.num_labels_super; C = cfg.num_labels_cat
                logits = out['logits']
                ls, lc = logits[:, :S], logits[:, S:S+C]
                return ls.argmax(-1).tolist(), lc.argmax(-1).tolist()
        if __name__ == '__main__':
            import sys
            texts = [line.strip() for line in sys.stdin if line.strip()]
            ps, pc = predict(texts)
            for a,b in zip(ps, pc):
                print(json.dumps({'super_id': int(a), 'cat_id': int(b)}))
        """))

    # genera README.md (model card) basilare
    readme = os.path.join(export_dir, "README.md")
    md = f"""# {os.path.basename(args.out_dir)} — MultiTaskBERTMasked (IT AEC)

Modello gerarchico (SUPER/CAT) con maschere ontologiche.

## Config
- Backbone: `{args.backbone_src or args.base_model}`
- ArcFace: `{cfg.use_arcface}` (s={cfg.arcface_s}, m={cfg.arcface_m})
- Embedding dim: {cfg.proj_dim}
- Super: {cfg.num_labels_super} — Cat: {cfg.num_labels_cat}

## Files
- `model.safetensors`, `config.json`, `tokenizer.json`/`tokenizer_config.json`
- `label_maps.json`, `ontology.json`, `mask_report.json`
- `predict.py`

## Uso rapido
```python
from transformers import AutoTokenizer, AutoConfig
from masked_model import MultiTaskBERTMasked
import torch, numpy as np
cfg = AutoConfig.from_pretrained('PATH')
mask = torch.tensor(np.array(cfg.mask_matrix), dtype=torch.float32)
mdl = MultiTaskBERTMasked.from_pretrained('PATH', config=cfg,
        num_super=cfg.num_labels_super, num_cat=cfg.num_labels_cat, mask_matrix=mask)
mdl.eval(); tok = AutoTokenizer.from_pretrained('PATH')
```
"""
    with open(readme, "w", encoding="utf-8") as w:
        w.write(md)

    print(f"[EXPORT] Pacchetto salvato in: {export_dir}")

    # ---------- Publish su Hugging Face Hub (opzionale) ----------
    if args.publish_hub and args.hub_repo:
        try:
            from huggingface_hub import create_repo, upload_folder
            create_repo(args.hub_repo, exist_ok=True, private=bool(args.hub_private))
            upload_folder(folder_path=export_dir, repo_id=args.hub_repo, repo_type="model")
            print(f"[HUB] Caricato su https://huggingface.co/{args.hub_repo}")
        except Exception as e:
            print("[WARN] Publish fallito:", e)

    print("[DONE] Salvato in", args.out_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
