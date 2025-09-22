
from __future__ import annotations
import json, torch, numpy as np
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from torch import nn
from .utils import set_seed, classification_metrics, build_llrd_optimizer

@dataclass
class HierConfig:
    model_name: str
    train_path: str
    val_path: str
    super_index: str
    cat_index: str
    save_dir: str = "runs/hier"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 5
    lr: float = 3e-5
    seed: int = 42
    llrd_decay: float = 0.9
    freeze_n_layers: int = 0
    alpha_super: float = 0.3
    alpha_cat: float = 0.7

class HierModel(PreTrainedModel):
    def __init__(self, config, base_model_name: str, num_super: int, num_cat: int, freeze_n_layers: int=0):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(base_model_name, config=config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        self.super_head = nn.Linear(hidden, num_super)
        self.cat_head = nn.Linear(hidden, num_cat)

        if freeze_n_layers > 0:
            for name, param in self.backbone.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = False
                import re
                m = re.search(r"encoder\.layer\.(\d+)", name)
                if m and int(m.group(1)) < freeze_n_layers:
                    param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels_super=None, labels_cat=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = out.last_hidden_state[:,0,:]
        cls = self.dropout(cls)
        logits_super = self.super_head(cls)
        logits_cat = self.cat_head(cls)
        loss = None
        if labels_super is not None and labels_cat is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_super, labels_super) + loss_fct(logits_cat, labels_cat)
        return {"loss": loss, "logits_super": logits_super, "logits_cat": logits_cat}

def _load_config(path: str) -> HierConfig:
    with open(path, "r", encoding="utf-8") as f: js=json.load(f)
    return HierConfig(**js)

def main(config_path: str|None=None):
    assert config_path, "Passa --config al trainer hier"
    cfg = _load_config(config_path)
    set_seed(cfg.seed)

    ds = load_dataset("json", data_files={"train": cfg.train_path, "validation": cfg.val_path})

    with open(cfg.super_index,"r",encoding="utf-8") as f: sup2id=json.load(f)["label2id"]
    with open(cfg.cat_index,"r",encoding="utf-8") as f: cat2id=json.load(f)["label2id"]
    num_super, num_cat = len(sup2id), len(cat2id)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    def tok_fn(examples):
        tok = tokenizer(examples["text"], truncation=True, max_length=cfg.max_length)
        tok["labels_super"] = examples["super_id"]
        tok["labels_cat"] = examples["cat_id"]
        return tok
    ds_tok = ds.map(tok_fn, batched=True)

    collator = DataCollatorWithPadding(tokenizer)

    base_cfg = AutoConfig.from_pretrained(cfg.model_name)
    model = HierModel(base_cfg, base_model_name=cfg.model_name, num_super=num_super, num_cat=num_cat, freeze_n_layers=cfg.freeze_n_layers)

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
        metric_for_best_model="eval_f1_macro_cat",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        bf16=True,
    )

    def compute_metrics(eval_pred):
        # eval_pred is (predictions, labels) but with custom outputs; easier: override in Trainer?
        # We'll hook into predictions by adapting Trainer to expect 'logits' key; we pack cat logits
        (logits, labels) = eval_pred
        # not used
        return {}

    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels_super = inputs.pop("labels_super")
            labels_cat = inputs.pop("labels_cat")
            outputs = model(**inputs, labels_super=labels_super, labels_cat=labels_cat)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            labels_super = inputs.pop("labels_super")
            labels_cat = inputs.pop("labels_cat")
            with torch.no_grad():
                outputs = model(**inputs, labels_super=labels_super, labels_cat=labels_cat)
                loss = outputs["loss"]
            return (loss, (outputs["logits_super"], outputs["logits_cat"]), (labels_super, labels_cat))

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        optimizers=(build_llrd_optimizer(model, base_lr=cfg.lr, lr_decay=cfg.llrd_decay), None),
    )

    trainer.train()
    trainer.save_model(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    # Evaluate: compute metrics manually from predictions
    import numpy as np
    preds = trainer.predict(ds_tok["validation"])
    logits_super, logits_cat = preds.predictions
    y_super, y_cat = preds.label_ids
    yhat_super = np.argmax(logits_super, axis=-1)
    yhat_cat = np.argmax(logits_cat, axis=-1)
    from .utils import classification_metrics
    ms = classification_metrics(yhat_super, y_super)
    mc = classification_metrics(yhat_cat, y_cat)
    out = {"eval_super": ms, "eval_cat": mc}
    print(json.dumps(out, indent=2))
