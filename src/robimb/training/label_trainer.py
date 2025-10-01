"""Training utilities for the label embedding classifier."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from ..models.label_model import LabelEmbedModel, load_label_embed_model
from ..training.property_utils import (
    PropertyMetadata,
    build_property_metadata,
    build_property_targets,
)
from ..utils.dataset_prep import LabelMaps, build_mask_and_report
from ..utils.metrics_utils import make_compute_metrics
from ..utils.ontology_utils import load_label_maps
from ..utils.sampling import load_jsonl_to_df

__all__ = ["LabelTrainingArgs", "train_label_model", "main"]


@dataclass
class LabelTrainingArgs:
    base_model: str
    train_jsonl: str
    val_jsonl: str
    label_maps: str
    ontology: Optional[str]
    out_dir: str
    label_texts_super: Optional[str] = None
    label_texts_cat: Optional[str] = None
    backbone_src: Optional[str] = None
    tokenizer_src: Optional[str] = None
    init_from: Optional[str] = None
    unfreeze_last: int = 0
    seed: int = 42
    gradient_accumulation_steps: int = 1
    proj_dim: int = 256
    temperature: float = 0.07
    use_mean_pool: bool = True
    freeze_encoder: bool = False
    train_label_emb: bool = True
    epochs: int = 5
    batch_size: int = 64
    max_length: int = 256
    lr_head: float = 2e-4
    lr_encoder: float = 1e-5
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    balanced_sampler: bool = False
    best_metric: str = "eval_macro_f1_cat_gold_super"
    publish_hub: bool = False
    hub_repo: Optional[str] = None
    hub_private: bool = False
    property_presence_weight: float = 1.0
    property_regression_weight: float = 1.0


class SanitizeGrads(TrainerCallback):
    def on_after_backward(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        fixed = 0
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    fixed += 1
        if fixed:
            print(f"[SAN] Riparati {fixed} gradienti NaN/Inf")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_label_texts(path: Optional[str], fallback: Iterable[str]) -> List[str]:
    if not path:
        return [str(name) for name in fallback]
    texts: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("super_id") or record.get("cat_id") or record.get("label")
            text = record.get("text") or record.get("description") or name
            if name:
                texts[str(name)] = str(text)
    output: List[str] = []
    for name in fallback:
        output.append(texts.get(name, str(name)))
    return output


def _build_dataset(
    df: pd.DataFrame,
    max_length: int,
    tokenizer,
    property_meta: Optional[PropertyMetadata],
) -> Dataset:
    if not {"text", "super_label", "cat_label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text', 'super_label' and 'cat_label' columns.")

    def _tokenize(batch: Dict[str, List]):
        tokens = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokens["super_labels"] = batch["super_label"]
        tokens["cat_labels"] = batch["cat_label"]
        if property_meta is not None and property_meta.has_properties():
            properties = batch.get("properties", [{} for _ in batch["cat_label"]])
            mask, presence, reg_targets, reg_mask = build_property_targets(
                properties,
                batch["cat_label"],
                property_meta,
            )
            tokens["property_slot_mask"] = mask.tolist()
            tokens["property_presence_labels"] = presence.tolist()
            tokens["property_regression_targets"] = reg_targets.tolist()
            tokens["property_regression_mask"] = reg_mask.tolist()
        else:
            empty = [[0.0] * 0 for _ in batch["cat_label"]]
            tokens["property_slot_mask"] = empty
            tokens["property_presence_labels"] = empty
            tokens["property_regression_targets"] = empty
            tokens["property_regression_mask"] = empty
        return tokens

    dataset = Dataset.from_pandas(df, preserve_index=False)
    keep_cols = {"properties", "property_schema"}
    remove_cols = [col for col in df.columns.tolist() if col not in keep_cols]
    return dataset.map(_tokenize, batched=True, remove_columns=remove_cols)


def _param_groups(model: LabelEmbedModel, lr_head: float, lr_encoder: float, weight_decay: float):
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay},
    ]
    return groups


def _build_sampler(dataset: Dataset) -> Optional[WeightedRandomSampler]:
    cat_labels = np.array(dataset["cat_labels"])
    unique, counts = np.unique(cat_labels, return_counts=True)
    freq = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.array([1.0 / freq[label] for label in cat_labels], dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_label_model(args: LabelTrainingArgs) -> Tuple[Trainer, Dict[str, float]]:
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_name2id, c_name2id, s_id2name, c_id2name = load_label_maps(args.label_maps)
    label_maps = LabelMaps(
        super_name_to_id=s_name2id,
        cat_name_to_id=c_name2id,
        super_id_to_name=s_id2name,
        cat_id_to_name=c_id2name,
    )
    num_super = max(s_name2id.values()) + 1
    num_cat = max(c_name2id.values()) + 1
    nd_id = c_name2id.get("#N/D")

    mask_matrix, mask_report = build_mask_and_report(args.ontology, label_maps)

    tokenizer_src = args.tokenizer_src or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    label_texts_super = _load_label_texts(args.label_texts_super, [s_id2name[i] for i in range(num_super)])
    label_texts_cat = _load_label_texts(args.label_texts_cat, [c_id2name[i] for i in range(num_cat)])

    config = AutoConfig.from_pretrained(
        args.base_model,
        num_labels_super=num_super,
        num_labels_cat=num_cat,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        use_mean_pool=args.use_mean_pool,
        l2_normalize_emb=True,
        mask_matrix=mask_matrix.tolist(),
        ban_nd_in_eval=True,
        nd_id=nd_id,
    )
    config.label_texts_super = label_texts_super
    config.label_texts_cat = label_texts_cat
    config.backbone_src = args.base_model if args.backbone_src is None else args.backbone_src

    train_df = load_jsonl_to_df(args.train_jsonl)
    eval_df = load_jsonl_to_df(args.val_jsonl)

    property_meta = build_property_metadata((train_df, eval_df), num_cat)
    if property_meta.has_properties():
        config.num_properties = property_meta.num_slots
        config.property_slot_names = list(property_meta.slot_names)
        config.property_numeric_mask = property_meta.numeric_mask.astype(int).tolist()
        config.property_cat_mask = property_meta.cat_property_mask.tolist()
        config.property_presence_weight = float(args.property_presence_weight)
        config.property_regression_weight = float(args.property_regression_weight)
    else:
        config.num_properties = 0
        config.property_slot_names = []
        config.property_numeric_mask = []
        config.property_cat_mask = []
        config.property_presence_weight = float(args.property_presence_weight)
        config.property_regression_weight = float(args.property_regression_weight)

    property_cat_mask_tensor = (
        torch.tensor(property_meta.cat_property_mask, dtype=torch.bool)
        if property_meta.has_properties()
        else None
    )
    property_numeric_tensor = (
        torch.tensor(property_meta.numeric_mask, dtype=torch.bool)
        if property_meta.has_properties()
        else None
    )

    if args.init_from:
        print(f"[INFO] Ripartenza da {args.init_from}")
        model = load_label_embed_model(
            args.init_from,
            backbone_src=args.base_model if args.backbone_src is None else args.backbone_src,
            tokenizer=tokenizer,
            config_overrides={
                "num_labels_super": num_super,
                "num_labels_cat": num_cat,
                "mask_matrix": mask_matrix.tolist(),
                "nd_id": nd_id,
                "num_properties": config.num_properties,
                "property_slot_names": config.property_slot_names,
                "property_numeric_mask": config.property_numeric_mask,
                "property_cat_mask": config.property_cat_mask,
                "property_presence_weight": float(args.property_presence_weight),
                "property_regression_weight": float(args.property_regression_weight),
            },
        )
    else:
        model = LabelEmbedModel(
            config=config,
            num_super=num_super,
            num_cat=num_cat,
            label_texts_super=label_texts_super,
            label_texts_cat=label_texts_cat,
            tokenizer=tokenizer,
            backbone_src=args.base_model if args.backbone_src is None else args.backbone_src,
            proj_dim=args.proj_dim,
            temperature=args.temperature,
            mask_matrix=torch.tensor(mask_matrix, dtype=torch.float32),
            ban_nd_in_eval=True,
            nd_id=nd_id,
            freeze_encoder=args.freeze_encoder,
            train_label_emb=args.train_label_emb,
            num_properties=config.num_properties,
            property_cat_mask=property_cat_mask_tensor,
            property_numeric_mask=property_numeric_tensor,
            property_presence_weight=args.property_presence_weight,
            property_regression_weight=args.property_regression_weight,
        )

    if args.unfreeze_last and not args.freeze_encoder:
        encoder_layers = getattr(model.backbone, "encoder", None) or getattr(model.backbone, "encoder", None)
        if encoder_layers is not None and hasattr(encoder_layers, "layer"):
            total = len(encoder_layers.layer)
            to_unfreeze = max(0, min(total, args.unfreeze_last))
            for i in range(total - to_unfreeze):
                for param in encoder_layers.layer[i].parameters():
                    param.requires_grad = False

    train_dataset = _build_dataset(train_df, args.max_length, tokenizer, property_meta)
    eval_dataset = _build_dataset(eval_df, args.max_length, tokenizer, property_meta)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr_head,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=True,
        push_to_hub=args.publish_hub,
        hub_model_id=args.hub_repo,
        hub_private_repo=args.hub_private,
        report_to=["tensorboard"],
        save_total_limit=3,
        label_names=[
            "super_labels",
            "cat_labels",
            "property_slot_mask",
            "property_presence_labels",
            "property_regression_targets",
            "property_regression_mask",
        ]
        if property_meta.has_properties()
        else ["super_labels", "cat_labels"],
    )

    optimizer = AdamW(_param_groups(model, args.lr_head, args.lr_encoder, args.weight_decay))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        compute_metrics=make_compute_metrics(num_super, num_cat, property_meta),
        callbacks=[SanitizeGrads()],
    )

    if args.balanced_sampler:
        sampler = _build_sampler(train_dataset)
        trainer.get_train_dataloader = lambda: torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_args.train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
        )

    trainer.train()

    metrics = trainer.evaluate()

    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    with open(export_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    property_metrics = {
        key: value for key, value in metrics.items() if key.startswith("eval_prop_")
    }
    if property_metrics:
        with open(export_dir / "property_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(property_metrics, handle, indent=2)
    with open(out_dir / "mask_report.json", "w", encoding="utf-8") as handle:
        json.dump(mask_report, handle, indent=2)
    shutil.copyfile(args.label_maps, export_dir / "label_maps.json")
    if args.ontology:
        shutil.copyfile(args.ontology, export_dir / "ontology.json")

    return trainer, metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the label embedding classifier")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--label_maps", required=True)
    parser.add_argument("--ontology", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label_texts_super", default=None)
    parser.add_argument("--label_texts_cat", default=None)
    parser.add_argument("--backbone_src", default=None)
    parser.add_argument("--tokenizer_src", default=None)
    parser.add_argument("--init_from", default=None)
    parser.add_argument("--unfreeze_last", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--use_mean_pool", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--train_label_emb", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--lr_encoder", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--best_metric", default="eval_macro_f1_cat_gold_super")
    parser.add_argument("--publish_hub", action="store_true")
    parser.add_argument("--hub_repo", default=None)
    parser.add_argument("--hub_private", action="store_true")
    parser.add_argument("--property_presence_weight", type=float, default=1.0)
    parser.add_argument("--property_regression_weight", type=float, default=1.0)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    args = LabelTrainingArgs(**vars(parsed))
    train_label_model(args)


if __name__ == "__main__":
    main()
