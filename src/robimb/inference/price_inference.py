"""Inference utilities for price prediction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file as load_safetensors

from ..models.price_regressor import PriceRegressor, PricePredictionPipeline

__all__ = ["PriceInference"]


class PriceInference:
    """Price prediction inference wrapper."""

    def __init__(
        self,
        model_dir: Path,
        device: Optional[str] = None,
    ):
        """Initialize price inference.

        Args:
            model_dir: Directory containing trained model
            device: Device to run on (cpu/cuda)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load property map
        property_map_path = model_dir / "property_id_map.json"
        if not property_map_path.exists():
            raise FileNotFoundError(f"Property map not found at {property_map_path}")

        with property_map_path.open() as f:
            property_id_map = json.load(f)

        # Load property unit map
        property_unit_map_path = model_dir / "property_unit_map.json"
        property_unit_map = None
        if property_unit_map_path.exists():
            with property_unit_map_path.open() as f:
                property_unit_map = json.load(f)

        # Load normalizers
        normalizers_path = model_dir / "normalizers.json"
        property_normalizers = None
        if normalizers_path.exists():
            with normalizers_path.open() as f:
                normalizers_data = json.load(f)
                property_normalizers = normalizers_data.get("property_normalizers", {})

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Try to load SafeTensors first, fallback to PyTorch
        safetensors_path = model_dir / "best_model.safetensors"
        config_path = model_dir / "best_model_config.json"

        if not safetensors_path.exists():
            safetensors_path = model_dir / "final_model.safetensors"
            config_path = model_dir / "final_model_config.json"

        # Load config
        if config_path.exists():
            with config_path.open() as f:
                checkpoint_data = json.load(f)
            config = checkpoint_data.get("config", {})
        else:
            # Fallback to old .pt format
            checkpoint_path = model_dir / "best_model.pt"
            if not checkpoint_path.exists():
                checkpoint_path = model_dir / "final_model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found in {model_dir}")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            config = checkpoint.get("config", {})
        backbone_name = config.get("backbone_name", "dbmdz/bert-base-italian-xxl-cased")
        num_properties = config.get("num_properties", len(property_id_map))
        dropout = config.get("dropout", 0.1)
        use_properties = config.get("use_properties", True)
        property_dim = config.get("property_dim", 64)
        unit_dim = config.get("unit_dim", 32)
        hidden_dims = config.get("hidden_dims", [512, 256])

        # Get num_units from config or UNIT_MAP
        from ..models.price_regressor import UNIT_MAP, PRICE_UNIT_MAP
        num_units = config.get("num_units", len(UNIT_MAP))
        num_price_units = config.get("num_price_units", len(PRICE_UNIT_MAP))  # NEW
        price_unit_dim = config.get("price_unit_dim", 16)  # NEW

        # Load HF token if needed
        hf_token = None
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
        except:
            pass

        # Initialize model
        model = PriceRegressor(
            backbone_name=backbone_name,
            num_properties=num_properties,
            num_units=num_units,
            num_price_units=num_price_units,  # NEW
            dropout=dropout,
            use_properties=use_properties,
            property_dim=property_dim,
            unit_dim=unit_dim,
            price_unit_dim=price_unit_dim,  # NEW
            hidden_dims=hidden_dims,
            hf_token=hf_token,
        )

        # Load model weights
        if safetensors_path.exists():
            # Load from SafeTensors
            state_dict = load_safetensors(str(safetensors_path))
            model.load_state_dict(state_dict)
        else:
            # Load from old .pt format
            model.load_state_dict(checkpoint["model_state_dict"])

        # Create pipeline
        self.pipeline = PricePredictionPipeline(
            model=model,
            tokenizer=tokenizer,
            property_id_map=property_id_map,
            property_normalizers=property_normalizers,
            property_unit_map=property_unit_map,
            device=device,
        )

        self.property_id_map = property_id_map
        self.model = model
        self.tokenizer = tokenizer

    def predict(
        self,
        text: str,
        properties: Optional[Dict[str, float]] = None,
        price_unit: str = "cad",  # NEW: Default to cadauno
    ) -> Dict[str, any]:
        """Predict price for a product.

        Args:
            text: Product description
            properties: Dict of extracted properties (e.g., {"dimensione_lunghezza": 200.0})
            price_unit: Price unit (e.g., "m2", "cad", "m")

        Returns:
            Dict with:
                - price: predicted price (EUR)
                - log_price: predicted log-price
                - currency: EUR
        """
        return self.pipeline.predict(text, properties, price_unit)

    def predict_batch(
        self,
        texts: List[str],
        properties_list: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, any]]:
        """Predict prices for multiple products.

        Args:
            texts: List of product descriptions
            properties_list: List of property dicts (same length as texts)

        Returns:
            List of prediction dicts
        """
        return self.pipeline.predict_batch(texts, properties_list)
