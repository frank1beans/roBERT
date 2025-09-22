
from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from .utils import set_seed

@dataclass
class TAPTConfig:
    model_name: str
    train_corpus: str
    val_corpus: Optional[str] = None
    save_dir: str = "runs/tapt"
    block_size: int = 128
    mlm_probability: float = 0.15
    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    seed: int = 42

def _load_config(path: str) -> TAPTConfig:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    return TAPTConfig(**js)

def main(config_path: str|None=None):
    assert config_path, "Passa --config al trainer TAPT"
    cfg = _load_config(config_path)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model_name)

    data_files = {"train": cfg.train_corpus}
    if cfg.val_corpus:
        data_files["validation"] = cfg.val_corpus
    ds = load_dataset("text", data_files=data_files)

    def tok_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.block_size)
    tokenized = ds.map(tok_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.mlm_probability)

    args = TrainingArguments(
        output_dir=cfg.save_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        evaluation_strategy="epoch" if "validation" in tokenized else "no",
        save_strategy="epoch" if "validation" in tokenized else "no",
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True if "validation" in tokenized else False,
        bf16=True if hasattr(tokenizer, "is_fast") else False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"],
                      eval_dataset=tokenized.get("validation"), data_collator=collator)

    trainer.train()
    trainer.save_model(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    # compute final perplexity if eval exists
    if "validation" in tokenized:
        eval_res = trainer.evaluate()
        try:
            ppl = math.exp(eval_res["eval_loss"])
        except Exception:
            ppl = None
        print(json.dumps({"eval": eval_res, "perplexity": ppl}, indent=2))
