# -*- coding: utf-8 -*-
"""
Trainer per Label-Embedding allineato al trainer ontologico:
- CLI/feature parallele (resume, init_from, unfreeze_last, scheduler, export, safetensors)
- Dataset con NOME 'super'/'cat' (storico) — mappato via label_maps (name->id)
- Mask ontologica opzionale (se passi un'ontologia compatibile)
- Class weights (effective number), balanced sampler opzionale
- Verbose logging
"""
from __future__ import annotations
import os
import json, argparse, random, glob, shutil, textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.optim import AdamW
from transformers import TrainerCallback

from robert.data.ontology import load_label_maps, load_ontology, build_mask
from label_model import LabelEmbedModel

# -------------------------- Utils --------------------------
def pick_latest_checkpoint(run_dir: str) -> str | None:
    pats = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not pats:
        return None
    pats.sort(key=lambda p: int(p.rsplit("-", 1)[1]))
    return pats[-1]

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_jsonl_to_df(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def build_mask_from_ontology(ontology_path: str, super_name_to_id, cat_name_to_id):
    if not ontology_path or not os.path.isfile(ontology_path):
        return None, {"note": "no ontology provided"}
    ontology = load_ontology(ontology_path)
    mask, report = build_mask(ontology, super_name_to_id, cat_name_to_id, return_report=True)
    return mask, report

class SanitizeGrads(TrainerCallback):
    def on_after_backward(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None: return
        fixed = 0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0); fixed += 1
        if fixed: print(f"[SAN] Riparati {fixed} gradienti NaN/Inf")

# ---------- Metrics (Trainer) ----------
def make_compute_metrics(num_super: int, num_cat: int):
    def _compute(eval_pred):
        preds = getattr(eval_pred, "predictions", eval_pred)
        labels = getattr(eval_pred, "label_ids", None)
        if isinstance(preds, (tuple, list)): preds = preds[0]
        if torch.is_tensor(preds): preds = preds.detach().cpu().numpy()
        logits = np.asarray(preds); logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if isinstance(labels, dict):
            y_s = np.asarray(labels["super_labels"]) ; y_c = np.asarray(labels["cat_labels"])
        elif isinstance(labels, (tuple, list)):
            y_s = np.asarray(labels[0]); y_c = np.asarray(labels[1])
        else:
            labels = np.asarray(labels); y_s, y_c = labels[:, 0], labels[:, 1]

        S = num_super; C = num_cat
        assert logits.shape[1] == S + 2*C, f"atteso (N, S+2C) = {S+2*C}, trovato {logits.shape}"
        ls      = logits[:, :S]
        lc_pred = logits[:, S:S + C]
        lc_gold = logits[:, S + C:S + 2*C]

        ps = ls.argmax(-1); pc_pred = lc_pred.argmax(-1); pc_gold = lc_gold.argmax(-1)
        m = (y_c != -100)
        return {
            "acc_super": accuracy_score(y_s, ps),
            "macro_f1_super": f1_score(y_s, ps, average="macro"),
            "acc_cat_pred_super": accuracy_score(y_c[m], pc_pred[m]) if m.any() else float("nan"),
            "macro_f1_cat_pred_super": f1_score(y_c[m], pc_pred[m], average="macro") if m.any() else float("nan"),
            "acc_cat_gold_super": accuracy_score(y_c[m], pc_gold[m]) if m.any() else float("nan"),
            "macro_f1_cat_gold_super": f1_score(y_c[m], pc_gold[m], average="macro") if m.any() else float("nan"),
        }
    return _compute

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    # parallela al trainer ontologico
    ap.add_argument("--base_model", required=True, help="Backbone HF o checkpoint locale (dir con config/pesi/tokenizer)")
    ap.add_argument("--backbone_src", default=None)
    ap.add_argument("--tokenizer_src", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--label_maps",  required=True)
    ap.add_argument("--ontology",    default=None, help="Opzionale, per costruire la mask")
    ap.add_argument("--out_dir",     required=True)

    # resume/init
    ap.add_argument("--init_from", type=str, default=None, help="Cartella modello precedente (export) da cui ripartire")
    ap.add_argument("--unfreeze_last", type=int, default=0, help="Sblocca ultimi N layer del backbone")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # label-emb knobs
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--use_mean_pool", type=lambda x: str(x).lower() != "false", default=True)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--train_label_emb", type=lambda x: str(x).lower() != "false", default=True)

    # training knobs
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--lr_head", type=float, default=2e-4, help="LR per label-emb/proiezione")
    ap.add_argument("--lr_encoder", type=float, default=1e-5, help="LR per backbone")
    ap.add_argument("--weight_decay", type=float, default=0.01)

    # scheduler
    ap.add_argument("--scheduler", choices=["linear","cosine"], default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.10)

    # balanced & weights
    ap.add_argument("--balanced_sampler", type=lambda x: str(x).lower() != "false", default=False)

    # export/publish
    ap.add_argument("--best_metric", type=str, default="eval_macro_f1_cat_gold_super")
    ap.add_argument("--publish_hub", action="store_true")
    ap.add_argument("--hub_repo", type=str, default=None)
    ap.add_argument("--hub_private", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # ---- resume
    resume_ckpt = pick_latest_checkpoint(args.out_dir)

    # ---- label maps & ids
    s_name2id, c_name2id, s_id2name, c_id2name = load_label_maps(args.label_maps)
    num_super = max(s_name2id.values()) + 1
    num_cat   = max(c_name2id.values()) + 1
    nd_id     = c_name2id.get("#N/D", None)

    # ---- mask da ontologia (opzionale)
    mask_matrix, report = build_mask_from_ontology(args.ontology, s_name2id, c_name2id)
    if mask_matrix is None:
        S = num_super; C = num_cat
        mask_matrix = np.ones((S, C), dtype=np.float32)  # fallback: nessun vincolo
        report = {"note": "no mask; using full-ones"}
    with open(os.path.join(args.out_dir, "mask_report.json"), "w", encoding="utf-8") as w:
        json.dump(report, w, ensure_ascii=False, indent=2)

    # ---- dataset mapping (NOMI → ID; ignora righe senza mapping)
    def load_df(jsonl_path):
        df0 = load_jsonl_to_df(jsonl_path)
        keep = []
        miss_super = miss_cat = 0
        for _, r in df0.iterrows():
            txt = r["text"]; sname = r.get("super"); cname = r.get("cat")
            sid = s_name2id.get(sname, None); cid = c_name2id.get(cname, None)
            if sid is None: miss_super += 1; continue
            if cid is None: miss_cat   += 1; continue
            keep.append({"text": txt, "super_label": int(sid), "cat_label": int(cid)})
        if miss_super or miss_cat:
            print(f"[DATA] drop righe senza mapping: super={miss_super} cat={miss_cat}")
        df = pd.DataFrame(keep)
        return df

    train_df = load_df(args.train_jsonl)
    val_df   = load_df(args.val_jsonl)

    # ---- tokenizer
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
    eval_ds  = to_hfds(val_df).map(tokenize, batched=True, remove_columns=["text"])
    do_eval = len(eval_ds) > 0

    # ---- label texts (per inizializzare emb)
    def build_label_texts(id2name: dict[int,str], template="{name}."):
        return [template.format(name=id2name[i]) for i in sorted(id2name.keys())]
    super_texts = build_label_texts(s_id2name, "{name}.")
    cat_texts   = build_label_texts(c_id2name, "{name}.")

    # ---- config/backbone
    bb_src = args.backbone_src or args.base_model
    cfg = AutoConfig.from_pretrained(bb_src)
    cfg.proj_dim = int(args.proj_dim)
    cfg.use_mean_pool = bool(args.use_mean_pool)

    mask_tensor = torch.tensor(mask_matrix, dtype=torch.float32)

    # ---- modello (init_from opzionale)
    if args.init_from:
        print(f"[INIT] Carico interamente da: {args.init_from}")
        model = LabelEmbedModel.from_pretrained(
            args.init_from, config=cfg,
            num_super=num_super, num_cat=num_cat,
            label_texts_super=super_texts, label_texts_cat=cat_texts,
            tokenizer=tok, backbone_src=bb_src,
            proj_dim=args.proj_dim, temperature=args.temperature,
            use_mean_pool=args.use_mean_pool, mask_matrix=mask_tensor,
            nd_id=nd_id, ban_nd_in_eval=True,
            freeze_encoder=args.freeze_encoder, train_label_emb=args.train_label_emb,
        )
    else:
        print(f"[INIT] Inizializzo da backbone HF: {bb_src}")
        model = LabelEmbedModel(
            config=cfg, num_super=num_super, num_cat=num_cat,
            label_texts_super=super_texts, label_texts_cat=cat_texts,
            tokenizer=tok, backbone_src=bb_src,
            proj_dim=args.proj_dim, temperature=args.temperature,
            use_mean_pool=args.use_mean_pool, mask_matrix=mask_tensor,
            nd_id=nd_id, ban_nd_in_eval=True,
            freeze_encoder=args.freeze_encoder, train_label_emb=args.train_label_emb,
        )

    # ---- freeze/unfreeze ultimi N layer
    bb = model.backbone
    for p in bb.parameters(): p.requires_grad = False
    if args.unfreeze_last > 0 and hasattr(bb, "encoder") and hasattr(bb.encoder, "layer"):
        for layer in bb.encoder.layer[-args.unfreeze_last:]:
            for p in layer.parameters(): p.requires_grad = True
    print(f"[INFO] sbloccati gli ultimi {args.unfreeze_last} layer dell'encoder." if args.unfreeze_last > 0 else "[INFO] encoder congelato")

    # ---- optimizer (param groups)
    head_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    enc_params  = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    optimizer = AdamW([
        {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": enc_params,  "lr": args.lr_encoder, "weight_decay": args.weight_decay},
    ])

    # ---- collator & wrapper per Trainer (concat logits come ontologico)
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

    # sampler bilanciato (per classi cat sbilanciate)
    train_sampler = None
    if args.balanced_sampler:
        labels = train_df["cat_label"].values
        by_cat = {}
        for i, c in enumerate(labels): by_cat.setdefault(int(c), []).append(i)
        pools = [idxs for idxs in by_cat.values() if len(idxs) >= 2]
        if pools:
            class BalancedSampler(torch.utils.data.Sampler):
                def __init__(self, pools, batch_size, n_samples):
                    self.pools = [np.array(p) for p in pools]; self.batch = batch_size; self.n = n_samples
                def __iter__(self):
                    rng = np.random.default_rng(); buf = []; produced = 0
                    while produced < self.n:
                        rng.shuffle(self.pools)
                        for pool in self.pools:
                            if len(pool) < 2: continue
                            pick = rng.choice(pool, size=min(2, len(pool)), replace=False).tolist()
                            buf.extend(pick)
                            if len(buf) >= self.batch:
                                batch = buf[:self.batch]; buf = buf[self.batch:]
                                for i in batch:
                                    yield i; produced += 1
                                    if produced >= self.n: return
                    for i in buf: yield i
                def __len__(self): return self.n
            train_sampler = BalancedSampler(pools, args.batch_size, len(train_wrapped))
        else:
            print("[WARN] balanced_sampler richiesto ma nessuna cat con >=2 esempi. Uso default.")

    # wrapper per concatenare logits come nel trainer ontologico
    class Wrap(nn.Module):
        def __init__(self, core): super().__init__(); self.core = core
        def forward(self, **batch):
            out = self.core(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                token_type_ids=batch.get("token_type_ids", None),
                super_labels=batch.get("super_labels"),
                cat_labels=batch.get("cat_labels"),
                return_dict=True,
            )
            ls = out["logits_super"]; lc_pred = out["logits_cat_pred_masked"]; lc_gold = out["logits_cat_gold_masked"]
            logits_concat = torch.cat([ls, lc_pred, lc_gold], dim=-1)
            return (out["loss"], logits_concat) if out["loss"] is not None else logits_concat

    wrap = Wrap(model)

    # ---- training args
    tr_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(64, args.batch_size),
        num_train_epochs=args.epochs,
        learning_rate=args.lr_head,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=50,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        fp16=False, bf16=False,
        report_to=["none"],
        seed=args.seed,
        remove_unused_columns=False,
        save_total_limit=2,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=True,
        label_names=["super_labels", "cat_labels"],
    )

    trainer = Trainer(
        model=wrap,
        args=tr_args,
        train_dataset=train_wrapped,
        eval_dataset=eval_wrapped,
        compute_metrics=make_compute_metrics(num_super, num_cat),
        data_collator=collator,
        tokenizer=tok,
        optimizers=(optimizer, None),
        callbacks=[SanitizeGrads()],
    )

    if train_sampler is not None:
        trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
            train_wrapped, batch_size=tr_args.per_device_train_batch_size,
            sampler=train_sampler, collate_fn=collator
        )

    # ---------- Train ----------
    if resume_ckpt:
        try:
            print(f"[RESUME] riprendo da {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        except ValueError as e:
            if "vulnerability issue" in str(e) or "upgrade torch to at least v2.6" in str(e):
                # elimina stati optimizer/scheduler e riprova
                for pat in ("optimizer.*", "scheduler.*"):
                    for p in glob.glob(os.path.join(resume_ckpt, pat)):
                        try: os.remove(p)
                        except: pass
                print("[RESUME] optimizer/scheduler rimossi per compatibilità Torch<2.6; riprendo solo coi pesi del modello.")
                trainer.train(resume_from_checkpoint=resume_ckpt)
            else:
                raise
    else:
        trainer.train()

    # ---- save checkpoint finale + core pulito (safetensors)
    trainer.save_model(args.out_dir); tok.save_pretrained(args.out_dir)
    clean_dir = os.path.join(args.out_dir, "model_clean"); os.makedirs(clean_dir, exist_ok=True)
    model.save_pretrained(clean_dir, safe_serialization=True)

    # ---- eval finale
    final_metrics = trainer.evaluate(eval_dataset=eval_wrapped)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as w:
        json.dump(final_metrics, w, ensure_ascii=False, indent=2)

    # ---- export deploy-ready
    export_dir = os.path.join(args.out_dir, "export"); os.makedirs(export_dir, exist_ok=True)
    # 1) ricava super2id / cat2id robustamente
    try:
        # se le hai già calcolate prima nel file (ontologia_utils.load_label_maps)
        super2id = {k: int(v) for k, v in super_name_to_id.items()}
        cat2id   = {k: int(v) for k, v in cat_name_to_id.items()}
    except NameError:
        # fallback: da config o da label_maps.json
        super2id = getattr(cfg, "super2id", None)
        cat2id   = getattr(cfg, "cat2id",   None)
    
        if not (isinstance(super2id, dict) and isinstance(cat2id, dict) and super2id and cat2id):
            with open(args.label_maps, "r", encoding="utf-8") as f:
                lm = json.load(f)
            if "super2id" in lm and "cat2id" in lm:
                super2id = {k: int(v) for k, v in lm["super2id"].items()}
                cat2id   = {k: int(v) for k, v in lm["cat2id"].items()}
            elif "id2super" in lm and "id2cat" in lm:
                super2id = {v: int(k) for k, v in lm["id2super"].items()}
                cat2id   = {v: int(k) for k, v in lm["id2cat"].items()}
            else:
                raise SystemExit("[EXPORT] Impossibile ricostruire le mappe (né super2id/cat2id né id2* in label_maps.json).")
    
    # 2) derivati id2*
    id2super = {int(v): k for k, v in super2id.items()}
    id2cat   = {int(v): k for k, v in cat2id.items()}
    
    # 3) mask_matrix: usa quella del training o ricostruiscila da ontology.json
    mask_export = None
    try:
        # se hai 'mask_matrix' (lista o np.array) già calcolata nel training
        mask_export = np.array(mask_matrix, dtype=np.float32)
    except NameError:
        # prova a leggere da config
        mm = getattr(cfg, "mask_matrix", None)
        if mm is not None:
            mask_export = np.array(mm, dtype=np.float32)
    
    if mask_export is None:
        onto_path = os.path.join(args.out_dir, "export", "ontology.json")
        if not os.path.isfile(onto_path):
            # se non è ancora copiato in export, usa quello originale passato a --ontology
            onto_path = args.ontology if (args.ontology and os.path.isfile(args.ontology)) else None
        if onto_path is None:
            raise SystemExit("[EXPORT] mask_matrix assente e ontology.json non trovato per ricostruirla.")
    
        onto = json.load(open(onto_path, "r", encoding="utf-8"))
        S = max(id2super.keys()) + 1
        C = max(id2cat.keys()) + 1
        mask_export = np.zeros((S, C), dtype=np.float32)
        for sname, cats in onto.items():
            sid = super2id.get(sname)
            if sid is None: 
                continue
            for cname in cats:
                cid = cat2id.get(cname)
                if cid is not None:
                    mask_export[int(sid), int(cid)] = 1.0
    
    # 4) scrivi tutto nel config PRIMA del salvataggio
    cfg = model.config
    cfg.backbone_name = model.backbone.__class__.__name__
    cfg.backbone_src  = args.backbone_src or args.base_model
    cfg.num_labels_super = int(getattr(cfg, "num_labels_super", len(id2super)) or len(id2super))
    cfg.num_labels_cat   = int(getattr(cfg, "num_labels_cat",   len(id2cat))   or len(id2cat))
    cfg.nd_id            = None if (locals().get("nd_id", None) is None) else int(nd_id)
    cfg.super2id = {k: int(v) for k, v in super2id.items()}
    cfg.cat2id   = {k: int(v) for k, v in cat2id.items()}
    cfg.id2super = {int(k): v for k, v in id2super.items()}
    cfg.id2cat   = {int(k): v for k, v in id2cat.items()}
    cfg.mask_matrix = mask_export.tolist()

    # salvataggi
    model.save_pretrained(export_dir, safe_serialization=True)
    tok.save_pretrained(export_dir)

    for src, name in [
        (args.label_maps, "label_maps.json"),
        (args.ontology, "ontology.json"),
        (os.path.join(args.out_dir, "mask_report.json"), "mask_report.json"),
    ]:
        try:
            if src and os.path.isfile(src): shutil.copy2(src, os.path.join(export_dir, name))
        except Exception as e:
            print(f"[WARN] copia {name} fallita: {e}")
    
    # README
    readme = os.path.join(export_dir, "README.md")
    md = f"""# {os.path.basename(args.out_dir)} — Label-Embedding (IT AEC)

Model label-similarity con maschera ontologica opzionale.

## Config
- Backbone: `{args.backbone_src or args.base_model}`
- Embedding dim: {cfg.proj_dim}
- Temperature: {cfg.temperature}
- Super: {num_super} — Cat: {num_cat}

## Files
- `model.safetensors`, `config.json`, tokenizer
- `label_maps.json`, `mask_report.json`
- `predict.py` (placeholder)
"""
    with open(readme, "w", encoding="utf-8") as w:
        w.write(md)

    print(f"[EXPORT] Pacchetto salvato in: {export_dir}")

    # ---- publish (opzionale)
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
