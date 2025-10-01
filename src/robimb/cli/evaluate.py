"""Evaluate an exported model on a labelled dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import typer

from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding

from ..models.label_model import load_label_embed_model
from ..models.masked_model import load_masked_model

from ..reporting import generate_prediction_reports

from ..utils.sampling import load_jsonl_to_df
from ..utils.metrics_utils import make_compute_metrics
from ..utils.ontology_utils import load_label_maps

__all__ = [
    "EvaluationConfig",
    "evaluate_model",
    "evaluate_command",
    "build_arg_parser",
    "main",
    "ValidationConfig",
    "validate_model",
]


@dataclass(frozen=True)
class EvaluationConfig:
    model_dir: Path
    test_file: Path
    label_maps: Path
    ontology: Optional[Path] = None
    batch_size: int = 64
    max_length: int = 256
    output: Optional[Path] = None
    predictions: Optional[Path] = None
    report_dir: Optional[Path] = None


# Backwards compatibility aliases -------------------------------------------------
ValidationConfig = EvaluationConfig


def _build_dataset(path: Path, max_length: int, tokenizer) -> Dataset:
    df = load_jsonl_to_df(path)
    if not {"text", "super_label", "cat_label"}.issubset(df.columns):
        raise ValueError("Test file must contain text, super_label and cat_label columns")

    def _tokenize(batch: Dict[str, List]):
        tokens = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokens["super_labels"] = batch["super_label"]
        tokens["cat_labels"] = batch["cat_label"]
        return tokens

    dataset = Dataset.from_pandas(df, preserve_index=False)
    return dataset.map(_tokenize, batched=True, remove_columns=df.columns.tolist())


def evaluate_model(config: EvaluationConfig) -> Mapping[str, float]:
    import torch
    from torch.utils.data import DataLoader

    model_dir = config.model_dir
    label_maps = config.label_maps
    s_name2id, c_name2id, s_id2name, c_id2name = load_label_maps(label_maps)
    num_super = max(s_name2id.values()) + 1
    num_cat = max(c_name2id.values()) + 1
    model_config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if hasattr(model_config, "label_texts_super"):
        model = load_label_embed_model(
            model_dir,
            backbone_src=getattr(model_config, "backbone_src", None),
            tokenizer=tokenizer,
        )
    else:
        model = load_masked_model(model_dir)

    dataset = _build_dataset(config.test_file, config.max_length, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # manual evaluation loop to avoid Trainer dependency at inference time
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    predictions_super: List[np.ndarray] = []
    predictions_cat_pred: List[np.ndarray] = []
    predictions_cat_gold: List[np.ndarray] = []
    labels_super: List[np.ndarray] = []
    labels_cat: List[np.ndarray] = []

    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=data_collator)

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                super_labels=batch.get("super_labels"),
                cat_labels=batch.get("cat_labels"),
            )
            logits = outputs.get("logits")
            if logits is None:
                logits_super = outputs["logits_super"].detach().cpu().numpy()
                logits_cat_pred = outputs["logits_cat_pred_masked"].detach().cpu().numpy()
                logits_cat_gold = outputs["logits_cat_gold_masked"].detach().cpu().numpy()
                logits = np.concatenate([logits_super, logits_cat_pred, logits_cat_gold], axis=-1)
            else:
                logits = logits.detach().cpu().numpy()
            predictions_super.append(logits[:, :num_super])
            predictions_cat_pred.append(logits[:, num_super : num_super + num_cat])
            predictions_cat_gold.append(logits[:, num_super + num_cat :])
            labels_super.append(batch["super_labels"].detach().cpu().numpy())
            labels_cat.append(batch["cat_labels"].detach().cpu().numpy())

    logits_super = np.concatenate(predictions_super)
    logits_cat_pred = np.concatenate(predictions_cat_pred)
    logits_cat_gold = np.concatenate(predictions_cat_gold)
    labels_super_arr = np.concatenate(labels_super)
    labels_cat_arr = np.concatenate(labels_cat)

    logits = np.concatenate([logits_super, logits_cat_pred, logits_cat_gold], axis=-1)
    metrics = make_compute_metrics(num_super, num_cat)(
        {
            "predictions": logits,
            "label_ids": {
                "super_labels": labels_super_arr,
                "cat_labels": labels_cat_arr,
            },
        }
    )

    if config.output:
        with open(config.output, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    pred_super = logits_super.argmax(axis=-1)
    pred_cat = logits_cat_pred.argmax(axis=-1)
    if config.report_dir is not None:
        generate_prediction_reports(
            pred_super=pred_super,
            pred_cat=pred_cat,
            gold_super=labels_super_arr,
            gold_cat=labels_cat_arr,
            super_id_to_name=s_id2name,
            cat_id_to_name=c_id2name,
            output_dir=config.report_dir,
            prefix="evaluation",
        )
    if config.predictions:
        preds = []
        for idx in range(len(pred_super)):
            preds.append(
                {
                    "super_pred": s_id2name[int(pred_super[idx])],
                    "cat_pred": c_id2name[int(pred_cat[idx])],
                    "super_gold": s_id2name[int(labels_super_arr[idx])],
                    "cat_gold": c_id2name[int(labels_cat_arr[idx])],
                }
            )
        with open(config.predictions, "w", encoding="utf-8") as handle:
            for row in preds:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def validate_model(config: EvaluationConfig) -> Mapping[str, float]:
    """Compatibility wrapper for the legacy name used by external scripts."""

    return evaluate_model(config)


def evaluate_command(
    model_dir: Path = typer.Option(..., "--model-dir", exists=True, file_okay=True, dir_okay=True),
    test_file: Path = typer.Option(..., "--test-file", exists=True, dir_okay=False),
    label_maps: Path = typer.Option(..., "--label-maps", exists=True, dir_okay=False),
    ontology: Optional[Path] = typer.Option(None, "--ontology", dir_okay=False, help="Optional ontology for reporting"),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size for evaluation"),
    max_length: int = typer.Option(256, "--max-length", help="Tokenizer max length"),
    output: Optional[Path] = typer.Option(None, "--output", help="Path where metrics JSON should be saved"),
    predictions: Optional[Path] = typer.Option(None, "--predictions", help="Optional JSONL with detailed predictions"),
    report_dir: Optional[Path] = typer.Option(
        None,
        "--report-dir",
        help="Directory that will host confusion matrices and analytics",
    ),
) -> None:
    """Typer entrypoint delegating to :func:`evaluate_model`."""

    metrics = evaluate_model(
        EvaluationConfig(
            model_dir=model_dir,
            test_file=test_file,
            label_maps=label_maps,
            ontology=ontology,
            batch_size=batch_size,
            max_length=max_length,
            output=output,
            predictions=predictions,
            report_dir=report_dir,
        )
    )
    if output is None:
        typer.echo(json.dumps(metrics, indent=2, ensure_ascii=False))


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    metrics = evaluate_model(
        EvaluationConfig(
            model_dir=Path(args.model_dir),
            test_file=Path(args.test_file),
            label_maps=Path(args.label_maps),
            ontology=Path(args.ontology) if args.ontology else None,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output=Path(args.output) if args.output else None,
            predictions=Path(args.predictions) if args.predictions else None,
            report_dir=Path(args.report_dir) if args.report_dir else None,
        )
    )

    if args.output is None:
        print(json.dumps(metrics, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained BIM model")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--label-maps", required=True)
    parser.add_argument("--ontology", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", default=None, help="Optional metrics output path")
    parser.add_argument("--predictions", default=None, help="Optional path to save predictions")
    parser.add_argument("--report-dir", default=None, help="Directory for plots and evaluation reports")
    return parser


if __name__ == "__main__":
    main()
