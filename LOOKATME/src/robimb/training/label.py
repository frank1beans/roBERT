
from __future__ import annotations
import json, numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from .utils import set_seed, classification_metrics, build_llrd_optimizer

@dataclass
class LabelConfig:
    model_name: str
    train_path: str
    val_path: str
    label_index: str
    save_dir: str = "runs/label"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 5
    lr: float = 3e-5
    seed: int = 42
    llrd_decay: float = 0.9
    freeze_n_layers: int = 0  # freeze first N encoder layers

def _load_config(path: str) -> LabelConfig:
    with open(path, "r", encoding="utf-8") as f: js=json.load(f)
    return LabelConfig(**js)

def main(config_path: str|None=None):
    assert config_path, "Passa --config al trainer label"
    cfg = _load_config(config_path)
    set_seed(cfg.seed)

    # datasets from jsonl
    ds = load_dataset("json", data_files={"train": cfg.train_path, "validation": cfg.val_path})
    with open(cfg.label_index,"r",encoding="utf-8") as f:
        lid = json.load(f)["label2id"]
    num_labels = len(lid)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.max_length)
    ds_tok = ds.map(tok_fn, batched=True)
    collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)

    # optional freeze of bottom layers
    if cfg.freeze_n_layers > 0:
        for name, param in model.named_parameters():
            # freeze embeddings
            if "embeddings" in name:
                param.requires_grad = False
            # freeze first N encoder layers
            import re
            m = re.search(r"encoder\.layer\.(\d+)", name)
            if m and int(m.group(1)) < cfg.freeze_n_layers:
                param.requires_grad = False

    args = TrainingArguments(
        output_dir=cfg.save_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        bf16=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return classification_metrics(preds, labels)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(build_llrd_optimizer(model, base_lr=cfg.lr, lr_decay=cfg.llrd_decay), None)
    )

    trainer.train()
    trainer.save_model(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    eval_res = trainer.evaluate()
    print(json.dumps({"eval": eval_res}, indent=2))
