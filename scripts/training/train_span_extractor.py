"""Training script for span-based property extractor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from robimb.models.span_extractor import PropertySpanExtractor


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
                example = json.loads(line)
                prop_id = example["property_id"]
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


def main():
    # Config
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "outputs" / "qa_dataset" / "property_extraction_qa.jsonl"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "span_extractor_model"
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load HF token for atipiqal/BOB
    from dotenv import load_dotenv
    import os
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Use TAPT domain-specific model
    BACKBONE = "atipiqal/BOB"  # TAPT on BIM/construction domain (XLM-RoBERTa)
    BATCH_SIZE = 4 if not torch.cuda.is_available() else 8  # Smaller for CPU
    LEARNING_RATE = 2e-5  # Lower LR for fine-tuned model
    EPOCHS = 3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")
    print(f"Using backbone: {BACKBONE} (TAPT domain-specific)")

    # Property ID mapping
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

    # Save property map
    with (OUTPUT_DIR / "property_id_map.json").open("w") as f:
        json.dump(property_id_map, f, indent=2)

    # Tokenizer (with token for private model)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, token=HF_TOKEN)

    # Dataset
    dataset = PropertyQADataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        property_id_map=property_id_map,
        max_length=512,
    )

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model (with HF token for private backbone)
    model = PropertySpanExtractor(
        backbone_name=BACKBONE,
        num_properties=len(property_id_map),
        dropout=0.1,
        hf_token=HF_TOKEN,
    )
    model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    best_exact_match = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, DEVICE)
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
                },
                OUTPUT_DIR / "best_model.pt",
            )
            print(f"Saved best model (exact match: {best_exact_match:.4f})")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "property_id_map": property_id_map,
        },
        OUTPUT_DIR / "final_model.pt",
    )

    # Save tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print(f"Best exact match: {best_exact_match:.4f}")


if __name__ == "__main__":
    main()
