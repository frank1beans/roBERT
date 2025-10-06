"""Property-aware extractive QA utilities built on top of transformer encoders."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from .schema_registry import load_registry as load_schema_registry

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QAExample:
    """Single QA training/prediction example."""

    qid: str
    context: str
    question: str
    answers: List[str]
    answer_starts: List[int]
    property_id: str

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "QAExample":
        answers_payload = payload.get("answers") or []
        answer_texts = [item["text"] for item in answers_payload if "text" in item]
        answer_starts = [int(item["start"]) for item in answers_payload if "start" in item]
        return cls(
            qid=str(payload.get("id") or payload.get("qid") or payload.get("property_id")),
            context=str(payload["context"]),
            question=str(payload["question"]),
            answers=answer_texts,
            answer_starts=answer_starts,
            property_id=str(payload.get("property_id") or payload.get("qid") or ""),
        )


class PropertyQADataset(Dataset):
    """Torch dataset converting :class:`QAExample` objects into QA features."""

    def __init__(
        self,
        examples: Sequence[QAExample],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 64,
        is_train: bool = True,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.is_train = is_train
        self.features: List[Dict[str, Any]] = []
        self._build_features()

    def _build_features(self) -> None:
        if not self.examples:
            return

        questions = [example.question for example in self.examples]
        contexts = [example.context for example in self.examples]

        encoded: BatchEncoding = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=False,
        )

        sample_mapping = encoded.pop("overflow_to_sample_mapping")
        offset_mapping = encoded.pop("offset_mapping")

        for feature_index in range(len(encoded["input_ids"])):
            example_index = int(sample_mapping[feature_index])
            example = self.examples[example_index]

            input_ids = encoded["input_ids"][feature_index]
            attention_mask = encoded["attention_mask"][feature_index]

            sequence_ids = encoded.sequence_ids(feature_index)
            offsets = offset_mapping[feature_index]

            # Keep offsets only for the context tokens (sequence id == 1).
            context_offsets: List[Optional[Tuple[int, int]]] = []
            for idx, offset in enumerate(offsets):
                if sequence_ids[idx] != 1:
                    context_offsets.append(None)
                else:
                    context_offsets.append((offset[0], offset[1]))

            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            feature: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": context_offsets,
                "cls_index": cls_index,
                "example_id": example_index,
            }

            if self.is_train:
                start_position, end_position = self._compute_span_positions(example, context_offsets, cls_index)
                feature["start_positions"] = start_position
                feature["end_positions"] = end_position

            self.features.append(feature)

    def _compute_span_positions(
        self,
        example: QAExample,
        offsets: Sequence[Optional[Tuple[int, int]]],
        cls_index: int,
    ) -> Tuple[int, int]:
        if not example.answers or not example.answer_starts:
            return cls_index, cls_index

        start_char = example.answer_starts[0]
        end_char = start_char + len(example.answers[0])

        start_token = None
        end_token = None

        for idx, offset in enumerate(offsets):
            if offset is None:
                continue
            start, end = offset
            if start_token is None and start <= start_char < end:
                start_token = idx
            if end_token is None and start < end_char <= end:
                end_token = idx
            if start_token is not None and end_token is not None:
                break

        if start_token is None or end_token is None:
            return cls_index, cls_index

        return start_token, end_token

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        feature = self.features[index]
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
        }
        if self.is_train:
            item["start_positions"] = torch.tensor(feature["start_positions"], dtype=torch.long)
            item["end_positions"] = torch.tensor(feature["end_positions"], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def load_registry(registry_path: Path | str):
    """Convenience wrapper returning the schema registry object."""

    return load_schema_registry(registry_path)


def default_prompt_for(category_name: str, property_title: str) -> str:
    """Return a default Italian QA prompt for the given property."""

    return f"Nel seguente testo di capitolato per {category_name} indica '{property_title}'."


def build_properties_for_category(category_id: str, registry_path: Path | str) -> List[Tuple[str, str]]:
    """Return list of tuples ``(property_id, question)`` for a category."""

    registry = load_registry(registry_path)
    category = registry.get(category_id)
    if category is None:
        raise ValueError(f"Categoria '{category_id}' non presente nel registry")
    return [
        (prop.id, default_prompt_for(category.name, prop.title))
        for prop in category.properties
    ]


def prepare_qa_dataset(
    train_path: Path | str,
    val_path: Path | str | None,
    property_registry_path: Path | str,
    label_maps: Any,
    val_split: float = 0.2,
) -> Tuple[Any, Any]:
    """Prepare QA dataset from JSONL with extraction.

    Returns:
        Tuple of (train_df, val_df) as pandas DataFrames
    """
    import pandas as pd
    from ..utils.sampling import load_jsonl_to_df

    registry = load_registry(property_registry_path)
    train_df = load_jsonl_to_df(train_path)

    qa_records = []

    for idx, row in train_df.iterrows():
        text = row.get("text", "")
        cat = row.get("cat") or row.get("super")

        # Skip if cat is NaN, None, or empty
        if not text or cat is None or (isinstance(cat, float) and pd.isna(cat)):
            continue

        cat = str(cat).strip()
        if not cat:
            continue

        category = registry.get(cat)
        if not category:
            continue

        # For each property, create a QA example
        for prop in category.properties:
            qa_record = {
                "id": f"{idx}:{prop.id}",
                "context": text,
                "question": default_prompt_for(category.name, prop.title),
                "answers": [],  # Will be filled during training
                "property_id": prop.id,
            }
            qa_records.append(qa_record)

    qa_df = pd.DataFrame(qa_records)

    # Split
    if val_path:
        val_df = load_jsonl_to_df(val_path)
        val_qa_records = []
        for idx, row in val_df.iterrows():
            text = row.get("text", "")
            cat = row.get("cat") or row.get("super")

            # Skip if cat is NaN, None, or empty
            if not text or cat is None or (isinstance(cat, float) and pd.isna(cat)):
                continue

            cat = str(cat).strip()
            if not cat:
                continue

            category = registry.get(cat)
            if not category:
                continue
            for prop in category.properties:
                val_qa_records.append({
                    "id": f"{idx}:{prop.id}",
                    "context": text,
                    "question": default_prompt_for(category.name, prop.title),
                    "answers": [],
                    "property_id": prop.id,
                })
        val_qa_df = pd.DataFrame(val_qa_records)
    else:
        qa_df = qa_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(qa_df) * (1.0 - val_split))
        val_qa_df = qa_df.iloc[split_idx:].reset_index(drop=True)
        qa_df = qa_df.iloc[:split_idx].reset_index(drop=True)

    return qa_df, val_qa_df


def make_jsonl_from_rule_outputs(
    rule_output_path: Path | str,
    registry_path: Path | str,
    destination: Path | str,
) -> None:
    """Create QA training data from rule-based extraction outputs."""

    registry = load_registry(registry_path)
    rule_path = Path(rule_output_path)
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with rule_path.open("r", encoding="utf-8") as src, dest_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if not line.strip():
                continue
            payload = json.loads(line)
            context = payload.get("text") or ""
            categoria = payload.get("categoria") or payload.get("category")
            if not categoria:
                continue
            category = registry.get(categoria)
            if category is None:
                continue
            properties = payload.get("properties", {})
            for prop in category.properties:
                entry = properties.get(prop.id)
                if not entry:
                    continue
                span = entry.get("span")
                raw_value = entry.get("raw") or entry.get("value")
                if not raw_value or not span:
                    continue
                start, end = int(span[0]), int(span[1])
                qa_payload = {
                    "id": f"{idx}:{prop.id}",
                    "context": context,
                    "question": default_prompt_for(category.name, prop.title),
                    "answers": [
                        {
                            "text": context[start:end],
                            "start": start,
                        }
                    ],
                    "property_id": prop.id,
                }
                dst.write(json.dumps(qa_payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Training / prediction routines
# ---------------------------------------------------------------------------


def _load_examples(jsonl_path: Path | str) -> List[QAExample]:
    path = Path(jsonl_path)
    examples: List[QAExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(QAExample.from_json(payload))
    return examples


def train_property_qa(
    model_name: str,
    train_jsonl: Path | str,
    *,
    eval_jsonl: Path | str | None = None,
    out_dir: Path | str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 384,
    doc_stride: int = 128,
    seed: int = 42,
) -> None:
    """Fine-tune a QA encoder on property-level examples."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    train_examples = _load_examples(train_jsonl)
    train_dataset = PropertyQADataset(
        train_examples,
        tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        is_train=True,
    )

    eval_dataset = None
    if eval_jsonl:
        eval_examples = _load_examples(eval_jsonl)
        eval_dataset = PropertyQADataset(
            eval_examples,
            tokenizer,
            max_length=max_length,
            doc_stride=doc_stride,
            is_train=True,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="steps" if eval_dataset else "no",
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def predict_with_encoder(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[QAExample],
    *,
    max_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 64,
    null_threshold: float = 0.25,
    batch_size: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """Run extractive QA and return spans per property id."""

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = PropertyQADataset(
        examples,
        tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
        is_train=False,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)

    start_logits_list: List[torch.Tensor] = []
    end_logits_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits_list.extend(outputs.start_logits.cpu())
            end_logits_list.extend(outputs.end_logits.cpu())

    feature_to_logits = zip(dataset.features, start_logits_list, end_logits_list)
    best_predictions: Dict[int, Dict[str, Any]] = {}

    for feature, start_logits, end_logits in feature_to_logits:
        example_id = feature["example_id"]
        offsets = feature["offset_mapping"]
        cls_index = feature["cls_index"]

        null_score = (start_logits[cls_index] + end_logits[cls_index]).item()
        best_score = float("-inf")
        best_span: Optional[Tuple[int, int, int, int]] = None

        for start_idx, offset in enumerate(offsets):
            if offset is None:
                continue
            for end_idx in range(start_idx, min(start_idx + dataset.max_answer_length, len(offsets))):
                end_offset = offsets[end_idx]
                if end_offset is None:
                    break
                if end_offset[1] < offset[0]:  # invalid ordering
                    continue
                score = (start_logits[start_idx] + end_logits[end_idx]).item()
                if score > best_score:
                    best_score = score
                    best_span = (start_idx, end_idx, offset[0], end_offset[1])

        if best_span is None:
            continue

        score_delta = best_score - null_score
        previous = best_predictions.get(example_id)
        if previous is None or score_delta > previous["score"]:
            best_predictions[example_id] = {
                "score": score_delta,
                "span": best_span,
                "raw_score": best_score,
                "null_score": null_score,
            }

    results: Dict[str, Dict[str, Any]] = {}
    for example_id, prediction in best_predictions.items():
        example = dataset.examples[example_id]
        span = prediction["span"]
        if prediction["score"] < null_threshold:
            continue
        start_char, end_char = span[2], span[3]
        results[example.property_id] = {
            "span": example.context[start_char:end_char],
            "start": start_char,
            "end": end_char,
            "score": float(prediction["score"]),
        }

    return results


def answer_properties(
    model_dir: Path | str,
    text: str,
    category_id: str,
    registry_path: Path | str,
    *,
    null_threshold: float = 0.25,
    max_length: int = 384,
    doc_stride: int = 128,
    max_answer_length: int = 64,
) -> Dict[str, Dict[str, Any]]:
    """High-level helper creating prompts and returning property spans."""

    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    properties = build_properties_for_category(category_id, registry_path)
    examples = [
        QAExample(
            qid=f"{category_id}:{prop_id}",
            context=text,
            question=prompt,
            answers=[],
            answer_starts=[],
            property_id=prop_id,
        )
        for prop_id, prompt in properties
    ]

    return predict_with_encoder(
        model,
        tokenizer,
        examples,
        null_threshold=null_threshold,
        max_length=max_length,
        doc_stride=doc_stride,
        max_answer_length=max_answer_length,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property-aware extractive QA")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune a QA encoder")
    train_parser.add_argument("--model", required=True, help="Base encoder name or path")
    train_parser.add_argument("--train-jsonl", required=True, type=Path, help="Training QA JSONL")
    train_parser.add_argument("--eval-jsonl", type=Path, help="Evaluation QA JSONL")
    train_parser.add_argument("--out-dir", required=True, type=Path, help="Directory to store the fine-tuned model")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--max-length", type=int, default=384)
    train_parser.add_argument("--doc-stride", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict property spans for a text")
    predict_parser.add_argument("--model-dir", required=True, type=Path, help="Directory with fine-tuned QA model")
    predict_parser.add_argument("--text", required=True, help="Input text to analyse")
    predict_parser.add_argument("--category", required=True, help="Category identifier")
    predict_parser.add_argument("--registry", required=True, type=Path, help="Schema registry path")
    predict_parser.add_argument("--null-th", type=float, default=0.25, help="No-answer threshold")
    predict_parser.add_argument("--max-length", type=int, default=384)
    predict_parser.add_argument("--doc-stride", type=int, default=128)
    predict_parser.add_argument("--max-answer-length", type=int, default=64)

    return parser


def _cmd_train(args: argparse.Namespace) -> None:
    train_property_qa(
        model_name=args.model,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        seed=args.seed,
    )


def _cmd_predict(args: argparse.Namespace) -> None:
    predictions = answer_properties(
        model_dir=args.model_dir,
        text=args.text,
        category_id=args.category,
        registry_path=args.registry,
        null_threshold=args.null_th,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        max_answer_length=args.max_answer_length,
    )
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        _cmd_train(args)
    elif args.command == "predict":
        _cmd_predict(args)
    else:  # pragma: no cover - defensive
        parser.error(f"Comando non supportato: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
