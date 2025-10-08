"""Training utilities for the price regression model."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from safetensors.torch import save_file as save_safetensors
from tqdm import tqdm

from ..models.price_regressor import PriceRegressor, get_unit_id, get_price_unit_id

__all__ = ["PriceTrainingArgs", "train_price_model", "main"]


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@dataclass
class PriceTrainingArgs:
    """Arguments for price regressor training."""

    backbone_name: str
    train_data: str
    output_dir: str
    property_map: Optional[str] = None
    property_unit_map: Optional[str] = None
    use_properties: bool = True
    property_dim: int = 64
    unit_dim: int = 32
    hidden_dims: Optional[str] = "512,256"
    val_split: float = 0.1
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 10
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    seed: int = 42
    device: Optional[str] = None


class PriceDataset(Dataset):
    """Dataset for price regression."""

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        property_id_map: Dict[str, int],
        property_unit_map: Dict[str, str],
        max_length: int = 512,
        use_properties: bool = True,
    ):
        self.tokenizer = tokenizer
        self.property_id_map = property_id_map
        self.property_unit_map = property_unit_map
        self.max_length = max_length
        self.use_properties = use_properties

        # Load data
        self.examples = []
        filtered_count = 0
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)

                # Require text and price
                if "text" in example and "price" in example:
                    price = example["price"]
                    # Filter invalid prices and extreme outliers
                    # Remove: price <= 0, or price > 99th percentile threshold (50k)
                    if price > 0 and price <= 50000:
                        self.examples.append(example)
                    else:
                        filtered_count += 1

        print(f"Loaded {len(self.examples)} examples from {data_path}")
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} examples (invalid or extreme outliers)")

        # Calculate statistics for normalization
        self._compute_normalizers()

    def _compute_normalizers(self):
        """Compute mean and std for price and properties."""
        prices = [ex["price"] for ex in self.examples]
        self.price_mean = np.mean(prices)
        self.price_std = np.std(prices)

        # Log transform prices for better distribution
        log_prices = [np.log(p) for p in prices if p > 0]
        self.log_price_mean = np.mean(log_prices)
        self.log_price_std = np.std(log_prices)

        print(f"Price stats: mean={self.price_mean:.2f}, std={self.price_std:.2f}")
        print(f"Log-price stats: mean={self.log_price_mean:.4f}, std={self.log_price_std:.4f}")

        # Compute property normalizers
        self.property_normalizers = {}
        if self.use_properties:
            for prop_name in self.property_id_map.keys():
                values = []
                for ex in self.examples:
                    props = ex.get("properties", {})
                    if prop_name in props and isinstance(props[prop_name], (int, float)):
                        values.append(float(props[prop_name]))

                if values:
                    self.property_normalizers[prop_name] = (
                        np.mean(values),
                        np.std(values)
                    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        text = example["text"]
        price = example["price"]
        price_unit = example.get("price_unit", "cad")  # NEW: Load price_unit

        # Log-transform price (target)
        log_price = np.log(price) if price > 0 else 0.0

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "target": torch.tensor(log_price, dtype=torch.float),
            "price_unit_id": torch.tensor(get_price_unit_id(price_unit), dtype=torch.long),  # NEW
        }

        # Encode properties
        if self.use_properties:
            properties = example.get("properties", {})
            max_props = len(self.property_id_map)

            prop_ids = []
            prop_vals = []
            prop_units = []
            prop_mask = []

            for prop_name, prop_value in properties.items():
                if prop_name in self.property_id_map and isinstance(prop_value, (int, float)):
                    prop_ids.append(self.property_id_map[prop_name])

                    # Normalize
                    if prop_name in self.property_normalizers:
                        mean, std = self.property_normalizers[prop_name]
                        normalized_value = (prop_value - mean) / (std + 1e-8)
                    else:
                        normalized_value = prop_value

                    prop_vals.append(normalized_value)

                    # Get unit for this property
                    unit_str = self.property_unit_map.get(prop_name)
                    prop_units.append(get_unit_id(unit_str))

                    prop_mask.append(1.0)

            # Pad to max_props
            while len(prop_ids) < max_props:
                prop_ids.append(0)
                prop_vals.append(0.0)
                prop_units.append(0)
                prop_mask.append(0.0)

            item.update({
                "property_ids": torch.tensor(prop_ids, dtype=torch.long),
                "property_values": torch.tensor(prop_vals, dtype=torch.float),
                "property_units": torch.tensor(prop_units, dtype=torch.long),
                "property_mask": torch.tensor(prop_mask, dtype=torch.float),
            })

        return item


def train_epoch(
    model: PriceRegressor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    use_properties: bool,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        price_unit_ids = batch["price_unit_id"].to(device)  # NEW

        property_ids = None
        property_values = None
        property_units = None
        property_mask = None

        if use_properties:
            property_ids = batch["property_ids"].to(device)
            property_values = batch["property_values"].to(device)
            property_units = batch["property_units"].to(device)
            property_mask = batch["property_mask"].to(device)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            price_unit_ids=price_unit_ids,  # NEW
            property_ids=property_ids,
            property_values=property_values,
            property_units=property_units,
            property_mask=property_mask,
            targets=targets,
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
    model: PriceRegressor,
    dataloader: DataLoader,
    device: str,
    use_properties: bool,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            price_unit_ids = batch["price_unit_id"].to(device)  # NEW

            property_ids = None
            property_values = None
            property_units = None
            property_mask = None

            if use_properties:
                property_ids = batch["property_ids"].to(device)
                property_values = batch["property_values"].to(device)
                property_units = batch["property_units"].to(device)
                property_mask = batch["property_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                price_unit_ids=price_unit_ids,  # NEW
                property_ids=property_ids,
                property_values=property_values,
                property_units=property_units,
                property_mask=property_mask,
                targets=targets,
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            predictions.extend(outputs["predictions"].cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    # Calculate metrics
    predictions = np.array(predictions)
    targets_list = np.array(targets_list)

    # Metrics in log-space (more stable)
    mse_log = np.mean((predictions - targets_list) ** 2)
    mae_log = np.mean(np.abs(predictions - targets_list))
    rmse_log = np.sqrt(mse_log)

    # Convert to actual prices for interpretability
    pred_prices = np.exp(predictions)
    true_prices = np.exp(targets_list)

    # Clip individual errors to prevent extreme outliers from dominating MAPE
    percentage_errors = np.abs((true_prices - pred_prices) / (true_prices + 1e-8))
    percentage_errors_clipped = np.clip(percentage_errors, 0, 5.0)  # Cap at 500%
    mape = np.mean(percentage_errors_clipped) * 100

    # Additional metrics
    mae_actual = np.mean(np.abs(true_prices - pred_prices))
    rmse_actual = np.sqrt(np.mean((true_prices - pred_prices) ** 2))

    # Median metrics (more robust to outliers)
    median_ape = np.median(percentage_errors_clipped) * 100

    return {
        "loss": total_loss / len(dataloader),
        "rmse": rmse_log,  # RMSE in log-space
        "mae": mae_log,    # MAE in log-space
        "mape": mape,      # Clipped MAPE in actual space
        "median_ape": median_ape,  # Median APE (more robust)
        "mae_actual": mae_actual,   # MAE in actual euros
        "rmse_actual": rmse_actual, # RMSE in actual euros
    }


def train_price_model(args: PriceTrainingArgs) -> None:
    """Train the price regressor model."""
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
            "spessore_mm": 7,
            "formato": 8,
            "spessore_pannello_mm": 9,
        }
        print(f"Using default property map with {len(property_id_map)} properties")

    # Load or create property unit map
    if args.property_unit_map and Path(args.property_unit_map).exists():
        with open(args.property_unit_map, "r") as f:
            property_unit_map = json.load(f)
        print(f"Loaded property unit map with {len(property_unit_map)} properties")
    else:
        # Default property unit map
        property_unit_map = {
            "marchio": "none",
            "materiale": "none",
            "dimensione_lunghezza": "mm",
            "dimensione_larghezza": "mm",
            "dimensione_altezza": "mm",
            "tipologia_installazione": "none",
            "portata_l_min": "l/min",
            "spessore_mm": "mm",
            "formato": "none",
            "spessore_pannello_mm": "mm",
        }
        print(f"Using default property unit map")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save property map and unit map
    with (output_dir / "property_id_map.json").open("w") as f:
        json.dump(property_id_map, f, indent=2)

    with (output_dir / "property_unit_map.json").open("w") as f:
        json.dump(property_unit_map, f, indent=2)

    # Parse hidden dims
    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

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
    dataset = PriceDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        property_id_map=property_id_map,
        property_unit_map=property_unit_map,
        max_length=args.max_length,
        use_properties=args.use_properties,
    )

    # Save normalizers
    normalizers = {
        "price_mean": dataset.price_mean,
        "price_std": dataset.price_std,
        "log_price_mean": dataset.log_price_mean,
        "log_price_std": dataset.log_price_std,
        "property_normalizers": dataset.property_normalizers,
    }
    with (output_dir / "normalizers.json").open("w") as f:
        json.dump(normalizers, f, indent=2)

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
    from ..models.price_regressor import UNIT_MAP, PRICE_UNIT_MAP
    model = PriceRegressor(
        backbone_name=args.backbone_name,
        num_properties=len(property_id_map),
        num_units=len(UNIT_MAP),
        num_price_units=len(PRICE_UNIT_MAP),  # NEW
        dropout=args.dropout,
        use_properties=args.use_properties,
        property_dim=args.property_dim,
        unit_dim=args.unit_dim,
        price_unit_dim=16,  # NEW
        hidden_dims=hidden_dims,
        hf_token=hf_token,
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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
    best_mape = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.use_properties
        )
        print(f"Train loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader, device, args.use_properties)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val RMSE (log): {val_metrics['rmse']:.4f}")
        print(f"Val MAE (log): {val_metrics['mae']:.4f}")
        print(f"Val MAPE (clipped): {val_metrics['mape']:.2f}%")
        print(f"Val Median APE: {val_metrics['median_ape']:.2f}%")
        print(f"Val MAE (actual): €{val_metrics['mae_actual']:.2f}")
        print(f"Val RMSE (actual): €{val_metrics['rmse_actual']:.2f}")

        # Save best model (based on MAPE)
        if val_metrics["mape"] < best_mape:
            best_mape = val_metrics["mape"]

            # Save model weights in SafeTensors format
            save_safetensors(model.state_dict(), str(output_dir / "best_model.safetensors"))

            # Save metadata and config as JSON
            config_data = {
                "property_id_map": property_id_map,
                "metrics": _convert_to_serializable(val_metrics),
                "config": {
                    "backbone_name": args.backbone_name,
                    "num_properties": len(property_id_map),
                    "num_units": len(UNIT_MAP),
                    "num_price_units": len(PRICE_UNIT_MAP),
                    "dropout": args.dropout,
                    "use_properties": args.use_properties,
                    "property_dim": args.property_dim,
                    "unit_dim": args.unit_dim,
                    "price_unit_dim": 16,
                    "hidden_dims": hidden_dims,
                },
            }
            with open(output_dir / "best_model_config.json", "w") as f:
                json.dump(config_data, f, indent=2)

            print(f"Saved best model (MAPE: {best_mape:.2f}%)")

    # Save final model in SafeTensors format
    save_safetensors(model.state_dict(), str(output_dir / "final_model.safetensors"))

    # Save metadata and config as JSON
    config_data = {
        "property_id_map": property_id_map,
        "config": {
            "backbone_name": args.backbone_name,
            "num_properties": len(property_id_map),
            "num_units": len(UNIT_MAP),
            "num_price_units": len(PRICE_UNIT_MAP),
            "dropout": args.dropout,
            "use_properties": args.use_properties,
            "property_dim": args.property_dim,
            "unit_dim": args.unit_dim,
            "price_unit_dim": 16,
            "hidden_dims": hidden_dims,
        },
    }
    with open(output_dir / "final_model_config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete! Model saved to {output_dir}")
    print(f"Best MAPE: {best_mape:.2f}%")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for price regressor training."""
    parser = argparse.ArgumentParser(
        description="Train price regression model",
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
        help="Path to training data (JSONL format with text, price, properties fields)",
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
        "--use-properties",
        action="store_true",
        default=True,
        help="Use extracted properties as features",
    )
    parser.add_argument(
        "--property-dim",
        type=int,
        default=64,
        help="Dimension for property embeddings",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="512,256",
        help="Comma-separated hidden layer dimensions",
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
        default=16,
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
        default=10,
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
    """Main entry point for price regressor training."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_price_model(PriceTrainingArgs(**vars(args)))


if __name__ == "__main__":
    main()
