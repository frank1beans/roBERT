#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TAPT/MLM su corpus edilizio con XLM-R (o altro) – GPU-first.

Feature:
- Whole-Word Masking (--wwm) opzionale, altrimenti MLM standard.
- LLRD (--llrd) opzionale con AdamW e LR decrescente verso i layer bassi.
- Freeze/Unfreeze: congela i primi N layer e sblocca a epoca K.
- Early stopping + best model: stop se eval_loss non migliora.
- Resume robusto (--resume_from_checkpoint) + --safe_resume (ignora optimizer/scheduler .pt).
- BF16 su 4090, gradient checkpointing, TF32 on.
- Collator con pad_to_multiple_of=8 per efficienza su GPU.

Uso tipico:
python tapt_mlm.py data/corpus \
  --model xlm-roberta-base \
  --block_size 512 --epochs 4 \
  --batch_size 8 --grad_accum 8 --lr 1e-5 \
  --output_dir ./mlm_out2 --grad_ckpt --wwm \
  --freeze_layers 6 --unfreeze_at_epoch 1 \
  --resume_from_checkpoint ./mlm_out2/checkpoint-1539 --safe_resume
"""

import os, argparse, random, math
from pathlib import Path
from typing import List, Optional

# Solo safetensors (evita torch.load su .bin)
os.environ["TRANSFORMERS_USE_SAFETENSORS"] = "1"
# Evita blocco parallelism dopo fork
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# (opzionale) meno barre di progresso di datasets
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

import numpy as np
import torch
from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    DataCollatorForLanguageModeling, DataCollatorForWholeWordMask,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

# ------------------ UTILS ------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def assert_cuda_ready(force_cpu: bool = False):
    if force_cpu:
        print("[WARN] CPU mode attivo: training senza GPU (più lento).")
        return
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA non disponibile. Verifica driver NVIDIA sull'host, avvio container con --gpus all, "
            "NVIDIA_VISIBLE_DEVICES, e PyTorch build CUDA valida."
        )
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    cc_major, cc_minor = torch.cuda.get_device_capability(dev)
    cuda_ver = getattr(torch.version, "cuda", "unknown")
    print(f"[GPU] {name} | SM {cc_major}.{cc_minor} | torch CUDA {cuda_ver}")
    # TF32 su Ampere/Ada
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def load_text_files(paths: List[str], min_len: int = 5, dedup: bool = True) -> Dataset:
    ds = load_dataset("text", data_files=paths, split="train", keep_linebreaks=True)
    def _clean(ex): return {"text": ex["text"].strip()}
    ds = ds.map(_clean, remove_columns=["text"], desc="strip")
    ds = ds.filter(lambda t: len(t["text"]) >= min_len, desc=f"filter len>={min_len}")
    if dedup:
        seen = set()
        def _dedup():
            for x in ds:
                t = x["text"]
                if t not in seen:
                    seen.add(t)
                    yield x
        ds = Dataset.from_generator(_dedup)
    return ds

def prepare_mlm_blocks(ds: Dataset, tokenizer: AutoTokenizer, block_size: int) -> Dataset:
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            return_attention_mask=False,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )
    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"], desc="tokenize")

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // block_size) * block_size
        if total_len == 0:
            return {k: [] for k in concatenated.keys()}
        return {
            k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
    return tokenized.map(group_texts, batched=True, desc=f"group to blocks of {block_size}")

def freeze_bottom_n(model, n: int):
    if n <= 0: return
    if not hasattr(model, "roberta") or not hasattr(model.roberta, "encoder"):
        return
    for i, layer in enumerate(model.roberta.encoder.layer):
        req = (i < n)
        for p in layer.parameters():
            p.requires_grad = not req
    if n > 0:
        print(f"[INFO] Freezati i primi {n} layer encoder.")

def make_llrd_param_groups(model, base_lr=1e-5, decay=0.9, wd=0.01):
    """
    Layer-Wise LR Decay per XLM-R:
    - dedup per id(param) per evitare "param in più gruppi"
    - no_decay per bias/LayerNorm
    - salta lm_head.decoder.weight se è tied con embeddings
    """
    no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = []
    seen = set()

    def add_named_params(named_params, lr):
        # split decay / no_decay e dedup
        decay, nodecay = [], []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if any(k in name for k in no_decay_keys):
                nodecay.append(p)
            else:
                decay.append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd})
        if nodecay:
            groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})

    # 1) Encoder layers: LR decrescente verso il basso
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        encoder = model.roberta.encoder.layer
        lr = base_lr
        for i in reversed(range(len(encoder))):  # layer alti → lr più alta
            layer = encoder[i]
            named = [(f"encoder.layer.{i}.{n}", p) for n, p in layer.named_parameters()]
            add_named_params(named, lr)
            lr *= decay

        # 2) Embeddings (LR finale post-decay)
        emb_named = [(f"embeddings.{n}", p) for n, p in model.roberta.embeddings.named_parameters()]
        add_named_params(emb_named, lr)

    # 3) lm_head: attenzione al weight tying (decoder.weight == embeddings.weight)
    if hasattr(model, "lm_head"):
        lm_named = []
        decoder_w = None
        emb_w = None

        if hasattr(model.lm_head, "decoder") and hasattr(model.lm_head.decoder, "weight"):
            decoder_w = model.lm_head.decoder.weight
        if hasattr(model, "roberta") and hasattr(model.roberta, "embeddings"):
            emb_w = model.roberta.embeddings.word_embeddings.weight

        for n, p in model.lm_head.named_parameters():
            # se weight tied, evita di aggiungere il decoder.weight (già visto)
            if decoder_w is not None and emb_w is not None and p.data_ptr() == emb_w.data_ptr():
                continue
            lm_named.append((f"lm_head.{n}", p))

        add_named_params(lm_named, base_lr)

    return groups


class UnfreezeCallback(EarlyStoppingCallback):
    """EarlyStopping + sblocco layer a una certa epoca."""
    def __init__(self, early_stopping_patience=1, unfreeze_epoch: Optional[int]=None, unfreeze_fn=None):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_done = False
        self.unfreeze_fn = unfreeze_fn

    def on_epoch_begin(self, args, state, control, **kwargs):
        if (self.unfreeze_epoch is not None
            and not self.unfreeze_done
            and state.epoch is not None
            and int(state.epoch) >= int(self.unfreeze_epoch)):
            if self.unfreeze_fn:
                self.unfreeze_fn()
            self.unfreeze_done = True
        return super().on_epoch_begin(args, state, control, **kwargs)

# ------------------ MAIN ------------------
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="TAPT/MLM su corpus (GPU-first, XLM-R consigliato).")
    ap.add_argument("txt_dir", help="Cartella con file *-clean-MLM.txt")
    ap.add_argument("--model", default="xlm-roberta-base", help="Checkpoint di partenza (es. xlm-roberta-base)")
    ap.add_argument("--block_size", type=int, default=512, help="Lunghezza blocco token (<= max seq len)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_len", type=int, default=5, help="Filtro righe corte (caratteri)")
    ap.add_argument("--no_dedup", action="store_true", help="Disabilita dedup")
    ap.add_argument("--output_dir", default="./mlm_out")
    ap.add_argument("--grad_ckpt", action="store_true", help="Gradient checkpointing (meno VRAM)")
    ap.add_argument("--wwm", action="store_true", help="Whole-Word Masking invece del MLM standard")
    ap.add_argument("--llrd", action="store_true", help="Layer-Wise LR Decay (ottimo per stabilità)")
    ap.add_argument("--freeze_layers", type=int, default=0, help="Congela i primi N layer encoder all'inizio")
    ap.add_argument("--unfreeze_at_epoch", type=int, default=None, help="Sblocca i layer a questa epoca (int)")
    ap.add_argument("--resume_from_checkpoint", default=None, help="Cartella checkpoint per resume")
    ap.add_argument("--safe_resume", action="store_true",
                    help="Prima del resume, rimuove optimizer/scheduler/scaler .pt per evitare torch.load")
    ap.add_argument("--cpu", action="store_true", help="Forza training su CPU (skip check CUDA)")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    assert_cuda_ready(force_cpu=args.cpu)

    txt_dir = Path(args.txt_dir)
    files = sorted(str(p) for p in txt_dir.glob("*-clean-MLM.txt"))
    if not files:
        raise SystemExit(f"Nessun file *-clean-MLM.txt in {txt_dir}")

    print(f"[INFO] Files: {len(files)} | Esempio: {files[0]}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, use_safetensors=True)

    print("[INFO] Carico e pulisco testi…")
    ds = load_text_files(files, min_len=args.min_len, dedup=not args.no_dedup)

    print("[INFO] Creo blocchi MLM…")
    lm_ds = prepare_mlm_blocks(ds, tok, block_size=args.block_size)
    if len(lm_ds) == 0:
        raise SystemExit("Dopo il raggruppamento non ci sono blocchi: aumenta corpus o riduci --block_size")

    split = lm_ds.train_test_split(test_size=0.02, seed=args.seed)
    train_ds, val_ds = split["train"], split["test"]

    # Collator
    if args.wwm:
        collator = DataCollatorForWholeWordMask(tokenizer=tok, mlm_probability=0.15)
    else:
        collator = DataCollatorForLanguageModeling(
            tokenizer=tok, mlm=True, mlm_probability=0.15, pad_to_multiple_of=8
        )

    print("[INFO] Carico modello…")
    bf16_ok = (not args.cpu) and (torch.cuda.get_device_capability(0)[0] >= 8)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        use_safetensors=True,
        dtype=(torch.bfloat16 if bf16_ok else torch.float32),
        low_cpu_mem_usage=True,
    )
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # Freeze iniziale
    if args.freeze_layers > 0:
        freeze_bottom_n(model, args.freeze_layers)

    # TrainingArguments (moderni)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        bf16=(bf16_ok and not args.cpu),
        fp16=False,
        gradient_checkpointing=args.grad_ckpt,
        max_grad_norm=1.0,

        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        eval_accumulation_steps=64,
        seed=args.seed,
        report_to=["none"],
    )

    # Callback EarlyStopping + Unfreeze
    def _do_unfreeze():
        print(f"[INFO] Sblocco layer congelati (epoch >= {args.unfreeze_at_epoch})…")
        freeze_bottom_n(model, 0)

    callbacks = [UnfreezeCallback(
        early_stopping_patience=1,
        unfreeze_epoch=args.unfreeze_at_epoch,
        unfreeze_fn=_do_unfreeze if args.unfreeze_at_epoch is not None else None
    )]

    # Trainer (processing_class per evitare warning deprec)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
        callbacks=callbacks,
    )



    # LLRD: sostituisce l'optimizer del Trainer con param groups custom
    if args.llrd:
        from torch.optim import AdamW
    
        def _create_opt():
            groups = make_llrd_param_groups(model, base_lr=args.lr, decay=0.9, wd=0.01)
            # difensivo: rimuovi eventuali gruppi vuoti
            groups = [g for g in groups if g.get("params")]
            if not groups:
                raise RuntimeError("LLRD: nessun parametro gradiente trovato nei param groups.")
            return AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    
        # disabilita la creazione automatica e inietta il nostro optimizer
        trainer.create_optimizer = lambda: None
        trainer.optimizer = _create_opt()

    # Safe resume: rimuovi .pt potenzialmente bloccati dal CVE
    resume_path = args.resume_from_checkpoint
    if resume_path and args.safe_resume:
        for fn in ("optimizer.pt", "scheduler.pt", "scaler.pt"):
            p = Path(resume_path) / fn
            if p.exists():
                try:
                    p.unlink()
                    print(f"[SAFE_RESUME] rimosso {p.name}")
                except Exception as e:
                    print(f"[SAFE_RESUME] non riesco a rimuovere {p.name}: {e}")

    print("[INFO] Avvio training MLM…")
    trainer.train(resume_from_checkpoint=resume_path)

    print("[INFO] Valutazione finale…")
    metrics = trainer.evaluate()
    try:
        print("[VAL] loss:", metrics.get("eval_loss"))
        if "eval_loss" in metrics:
            print("[VAL] ppl:", math.exp(metrics["eval_loss"]))
    except Exception:
        pass

    print("[INFO] Salvataggio finale…")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[OK] Salvato in: {args.output_dir}")

if __name__ == "__main__":
    main()
