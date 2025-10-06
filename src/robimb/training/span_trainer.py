"""Training utilities for the span-based property extractor."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from ..models.span_extractor import PropertySpanExtractor

__all__ = ["SpanTrainingArgs", "train_span_model", "main"]


@dataclass
class SpanTrainingArgs:
    """Arguments for span extractor training."""

    backbone_name: str
    train_data: str
    output_dir: str
    property_map: Optional[str] = None
    val_split: float = 0.1
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    seed: int = 42
    device: Optional[str] = None


class PropertyQADataset(Dataset):
    """Dataset for property extraction QA pairs."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        property_id_map: Dict[str, int],
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.property_id_map = property_id_map
        self.max_length = max_length

        # Load data
        self.examples = []
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                prop_id = example.get("property_id")
                if prop_id in property_id_map:
                    self.examples.append(example)

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        text = example["context"]
        prop_id = example["property_id"]
        answer_start_char = example["answers"]["answer_start"][0]
        answer_text = example["answers"]["text"][0]
        answer_end_char = answer_start_char + len(answer_text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        # Find token positions for answer span
        offset_mapping = encoding["offset_mapping"]
        start_token = None
        end_token = None

        for token_idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens
            if start == end == 0:
                continue

            # Find start token
            if start_token is None and start <= answer_start_char < end:
                start_token = token_idx

            # Find end token
            if end_token is None and start < answer_end_char <= end:
                end_token = token_idx
                break

        # Fallback if not found
        if start_token is None:
            start_token = 0
        if end_token is None:
            end_token = start_token

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "property_id": torch.tensor(self.property_id_map[prop_id], dtype=torch.long),
            "start_position": torch.tensor(start_token, dtype=torch.long),
            "end_position": torch.tensor(end_token, dtype=torch.long),
        }


def train_epoch(
    model: PropertySpanExtractor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        property_ids = batch["property_id"].to(device)
        start_positions = batch["start_position"].to(device)
        end_positions = batch["end_position"].to(device)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            property_ids=property_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        loss = outputs["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(
    model: PropertySpanExtractor,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    exact_matches = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            property_ids = batch["property_id"].to(device)
            start_positions = batch["start_position"].to(device)
            end_positions = batch["end_position"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                property_ids=property_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            # Calculate exact match
            start_preds = outputs["start_logits"].argmax(dim=-1)
            end_preds = outputs["end_logits"].argmax(dim=-1)

            exact_matches += ((start_preds == start_positions) & (end_preds == end_positions)).sum().item()
            total_examples += input_ids.size(0)

    avg_loss = total_loss / len(dataloader)
    exact_match = exact_matches / total_examples

    return {
        "loss": avg_loss,
        "exact_match": exact_match,
    }


def train_span_model(args: SpanTrainingArgs) -> None:
    """Train the span extractor model."""
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Using backbone: {args.backbone_name}")

    # Load or create property map
    if args.property_map and Path(args.property_map).exists():
        with open(args.property_map, "r") as f:
            property_id_map = json.load(f)
        print(f"Loaded property map with {len(property_id_map)} properties")
    else:
        # Default property map
        property_id_map = {
            "marchio": 0,
            "materiale": 1,
            "dimensione_lunghezza": 2,
            "dimensione_larghezza": 3,
            "dimensione_altezza": 4,
            "tipologia_installazione": 5,
            "portata_l_min": 6,
            "normativa_riferimento": 7,
            "classe_ei": 8,
            "classe_reazione_al_fuoco": 9,
            "presenza_isolante": 10,
            "stratigrafia_lastre": 11,
            "spessore_mm": 12,
            "materiale_struttura": 13,
            "formato": 14,
            "spessore_pannello_mm": 15,
            "trasmittanza_termica": 16,
            "isolamento_acustico_db": 17,
            "colore_ral": 18,
            "coefficiente_fonoassorbimento": 19,
        }
        print(f"Using default property map with {len(property_id_map)} properties")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save property map
    with (output_dir / "property_id_map.json").open("w") as f:
        json.dump(property_id_map, f, indent=2)

    # Load HF token if needed
    hf_token = None
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
    except:
        pass

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_name, token=hf_token)

    # Dataset
    train_data_path = Path(args.train_data)
    dataset = PropertyQADataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        property_id_map=property_id_map,
        max_length=args.max_length,
    )

    # Split train/val
    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Model
    model = PropertySpanExtractor(
        backbone_name=args.backbone_name,
        num_properties=len(property_id_map),
        dropout=args.dropout,
        hf_token=hf_token,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    best_exact_match = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val exact match: {val_metrics['exact_match']:.4f}")

        # Save best model
        if val_metrics["exact_match"] > best_exact_match:
            best_exact_match = val_metrics["exact_match"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "property_id_map": property_id_map,
                    "exact_match": best_exact_match,
                    "config": {
                        "backbone_name": args.backbone_name,
                        "num_properties": len(property_id_map),
                        "dropout": args.dropout,
                    },
                },
                output_dir / "best_model.pt",
            )
            print(f"Saved best model (exact match: {best_exact_match:.4f})")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "property_id_map": property_id_map,
            "config": {
                "backbone_name": args.backbone_name,
                "num_properties": len(property_id_map),
                "dropout": args.dropout,
            },
        },
        output_dir / "final_model.pt",
    )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete! Model saved to {output_dir}")
    print(f"Best exact match: {best_exact_match:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for span extractor training."""
    parser = argparse.ArgumentParser(
        description="Train span-based property extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--backbone-name",
        type=str,
        default="dbmdz/bert-base-italian-xxl-cased",
        help="Pretrained model to use as backbone",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--property-map",
        type=str,
        help="Path to property ID mapping JSON file",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use for training",
    )

    return parser


def main(argv=None) -> None:
    """Main entry point for span extractor training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_span_model(SpanTrainingArgs(**vars(args)))


if __name__ == "__main__":
    main()
