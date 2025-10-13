"""Price regression model for construction products.

This model predicts average prices for BIM/construction products based on
their descriptions and extracted properties.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class PriceRegressor(nn.Module):
    """Regression model for predicting product prices.

    Uses a transformer backbone to encode product descriptions,
    optionally conditioning on extracted properties with unit awareness.
    """

    def __init__(
        self,
        backbone_name: str = "dbmdz/bert-base-italian-xxl-cased",
        num_properties: int = 20,
        num_units: int = 18,           # Property units
        num_price_units: int = 14,     # Price units
        dropout: float = 0.1,
        use_properties: bool = True,
        property_dim: int = 64,
        unit_dim: int = 32,
        price_unit_dim: int = 16,      # Price unit embedding dimension
        hidden_dims: List[int] = [512, 256],
        hf_token: Optional[str] = None,
    ):
        """Initialize price regressor.

        Args:
            backbone_name: Pretrained transformer model
            num_properties: Number of property types (for embeddings)
            num_units: Number of unit types (mm, cm, m, kg, etc.)
            num_price_units: Number of price unit types (€/m², €/cad, etc.)
            dropout: Dropout rate
            use_properties: Whether to condition on extracted properties
            property_dim: Dimension for property embeddings
            unit_dim: Dimension for unit embeddings
            price_unit_dim: Dimension for price unit embeddings
            hidden_dims: Hidden layer dimensions for regression head
            hf_token: HuggingFace token for private models
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(backbone_name, token=hf_token)
        self.backbone = AutoModel.from_pretrained(backbone_name, token=hf_token)

        hidden_size = self.backbone.config.hidden_size
        self.use_properties = use_properties

        # Price unit embedding (CRITICAL!)
        self.price_unit_embedding = nn.Embedding(num_price_units, price_unit_dim)

        # Property embeddings (learnable representations)
        if use_properties:
            self.property_embeddings = nn.Embedding(num_properties, property_dim)
            # Unit embeddings (e.g., mm, cm, m, kg, l/min, etc.)
            self.unit_embeddings = nn.Embedding(num_units, unit_dim)
            # Property value encoder (for numeric properties)
            self.property_value_encoder = nn.Linear(1, property_dim)
            # Combine property, unit, and value embeddings
            combined_prop_dim = property_dim + unit_dim
            self.property_combiner = nn.Linear(combined_prop_dim, property_dim)
            input_dim = hidden_size + property_dim + price_unit_dim
        else:
            input_dim = hidden_size + price_unit_dim

        self.num_properties = num_properties
        self.num_units = num_units
        self.num_price_units = num_price_units

        # Regression head (multi-layer MLP)
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final prediction layer (log-price)
        layers.append(nn.Linear(prev_dim, 1))

        self.regression_head = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.num_properties = num_properties

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price_unit_ids: torch.Tensor,  # NEW: Required parameter
        property_ids: Optional[torch.Tensor] = None,
        property_values: Optional[torch.Tensor] = None,
        property_units: Optional[torch.Tensor] = None,
        property_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            price_unit_ids: [batch_size] - price unit IDs (€/m², €/cad, etc.)
            property_ids: [batch_size, max_properties] - property type IDs
            property_values: [batch_size, max_properties] - normalized property values
            property_units: [batch_size, max_properties] - unit IDs for each property
            property_mask: [batch_size, max_properties] - mask for valid properties
            targets: [batch_size] - ground truth log-prices (for training)

        Returns:
            Dict with:
                - predictions: [batch_size] - predicted log-prices
                - loss: scalar (if targets provided)
        """
        # Encode text
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)

        # Encode price unit (CRITICAL!)
        price_unit_embeds = self.price_unit_embedding(price_unit_ids)  # [batch_size, price_unit_dim]

        # Encode properties (if provided)
        if self.use_properties:
            if property_ids is not None:
                # Get property type embeddings
                prop_embeds = self.property_embeddings(property_ids)  # [batch_size, max_props, prop_dim]

                # Encode property values
                if property_values is not None:
                    value_embeds = self.property_value_encoder(
                        property_values.unsqueeze(-1)
                    )  # [batch_size, max_props, prop_dim]
                    # Combine type and value embeddings
                    prop_embeds = prop_embeds + value_embeds

                # Encode units
                if property_units is not None:
                    unit_embeds = self.unit_embeddings(property_units)  # [batch_size, max_props, unit_dim]
                    # Concatenate property and unit embeddings
                    prop_embeds = torch.cat([prop_embeds, unit_embeds], dim=-1)  # [batch_size, max_props, prop_dim + unit_dim]

                # Apply property mask and pool
                if property_mask is not None:
                    prop_embeds = prop_embeds * property_mask.unsqueeze(-1)
                    # Average pooling over valid properties
                    prop_count = property_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    prop_pooled = prop_embeds.sum(dim=1) / prop_count  # [batch_size, prop_dim + unit_dim]
                else:
                    # Simple average pooling
                    prop_pooled = prop_embeds.mean(dim=1)  # [batch_size, prop_dim + unit_dim]

                prop_pooled = self.property_combiner(prop_pooled)
            else:
                # No properties provided - use zero embeddings
                batch_size = cls_output.size(0)
                prop_pooled = torch.zeros(
                    batch_size,
                    self.property_embeddings.embedding_dim,
                    device=cls_output.device,
                    dtype=cls_output.dtype
                )

            # Concatenate text, price unit, and property representations
            combined = torch.cat([cls_output, price_unit_embeds, prop_pooled], dim=-1)
        else:
            # Concatenate text and price unit only
            combined = torch.cat([cls_output, price_unit_embeds], dim=-1)

        # Predict log-price
        predictions = self.regression_head(combined).squeeze(-1)  # [batch_size]

        output = {"predictions": predictions}

        # Calculate loss if targets provided
        if targets is not None:
            # Use MSE loss on log-prices
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, targets)
            output["loss"] = loss

        return output

    def predict_price(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        price_unit_ids: torch.Tensor,
        property_ids: Optional[torch.Tensor] = None,
        property_values: Optional[torch.Tensor] = None,
        property_units: Optional[torch.Tensor] = None,
        property_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict prices (in original scale, not log).

        Returns:
            Tensor of predicted prices [batch_size]
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            price_unit_ids=price_unit_ids,
            property_ids=property_ids,
            property_values=property_values,
            property_units=property_units,
            property_mask=property_mask,
        )

        log_prices = outputs["predictions"]
        # Convert from log-price to price
        prices = torch.exp(log_prices)

        return prices


# Standard unit mappings (for property values)
UNIT_MAP = {
    "none": 0,  # No unit / categorical
    "mm": 1,
    "cm": 2,
    "m": 3,
    "mm2": 4,
    "cm2": 5,
    "m2": 6,
    "mm3": 7,
    "cm3": 8,
    "m3": 9,
    "kg": 10,
    "g": 11,
    "l": 12,
    "l/min": 13,
    "db": 14,
    "w/m2k": 15,  # trasmittanza termica
    "ral": 16,  # colore RAL
    "percentage": 17,
}


def get_unit_id(unit: Optional[str]) -> int:
    """Get unit ID from unit string."""
    if not unit:
        return UNIT_MAP["none"]
    unit_lower = unit.lower().strip()
    return UNIT_MAP.get(unit_lower, UNIT_MAP["none"])


# Price unit mappings (for price per unit)
PRICE_UNIT_MAP = {
    "cad": 0,      # € per cadauno
    "m": 1,        # € per metro lineare
    "m2": 2,       # € per metro quadrato
    "m3": 3,       # € per metro cubo
    "kg": 4,       # € per kilogrammo
    "l": 5,        # € per litro
    "h": 6,        # € per ora
    "giorno": 7,   # € per giorno
    "set": 8,      # € per set/kit
    "a_corpo": 9,  # € forfait
    "t": 10,       # € per tonnellata
    "q": 11,       # € per quintale
    "mese": 12,    # € per mese
    "g": 13,       # € per grammo
}


def get_price_unit_id(price_unit: Optional[str]) -> int:
    """Get price unit ID from price unit string."""
    if not price_unit:
        return PRICE_UNIT_MAP["cad"]
    unit_lower = price_unit.lower().strip()
    return PRICE_UNIT_MAP.get(unit_lower, PRICE_UNIT_MAP["cad"])


class PricePredictionPipeline:
    """End-to-end pipeline for price prediction."""

    def __init__(
        self,
        model: PriceRegressor,
        tokenizer,
        property_id_map: Dict[str, int],
        property_normalizers: Optional[Dict[str, Tuple[float, float]]] = None,
        property_unit_map: Optional[Dict[str, str]] = None,
        device: str = "cpu",
    ):
        """Initialize pipeline.

        Args:
            model: Trained price regressor
            tokenizer: Tokenizer for text encoding
            property_id_map: Mapping from property names to IDs
            property_normalizers: Dict mapping property names to (mean, std) for normalization
            property_unit_map: Dict mapping property names to their units
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.property_id_map = property_id_map
        self.property_normalizers = property_normalizers or {}
        self.property_unit_map = property_unit_map or {}
        self.device = device

        # Reverse map
        self.id_to_property = {v: k for k, v in property_id_map.items()}

    def predict(
        self,
        text: str,
        properties: Optional[Dict[str, float]] = None,
        price_unit: str = "cad",  # NEW
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
                - confidence: prediction confidence (optional)
        """
        self.model.eval()

        with torch.no_grad():
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode price_unit (NEW)
            price_unit_id = torch.tensor(
                [get_price_unit_id(price_unit)],
                dtype=torch.long,
                device=self.device
            )

            # Encode properties
            property_ids = None
            property_values = None
            property_units = None
            property_mask = None

            if properties and self.model.use_properties:
                max_props = len(self.property_id_map)
                prop_ids = []
                prop_vals = []
                prop_units_list = []
                prop_mask_list = []

                for prop_name, prop_value in properties.items():
                    if prop_name in self.property_id_map:
                        prop_ids.append(self.property_id_map[prop_name])

                        # Normalize property value
                        if prop_name in self.property_normalizers:
                            mean, std = self.property_normalizers[prop_name]
                            normalized_value = (prop_value - mean) / (std + 1e-8)
                        else:
                            normalized_value = prop_value

                        prop_vals.append(normalized_value)

                        # Get unit for this property
                        unit_str = self.property_unit_map.get(prop_name)
                        prop_units_list.append(get_unit_id(unit_str))

                        prop_mask_list.append(1.0)

                # Pad to max_props
                while len(prop_ids) < max_props:
                    prop_ids.append(0)
                    prop_vals.append(0.0)
                    prop_units_list.append(0)
                    prop_mask_list.append(0.0)

                property_ids = torch.tensor([prop_ids], dtype=torch.long, device=self.device)
                property_values = torch.tensor([prop_vals], dtype=torch.float, device=self.device)
                property_units = torch.tensor([prop_units_list], dtype=torch.long, device=self.device)
                property_mask = torch.tensor([prop_mask_list], dtype=torch.float, device=self.device)

            # Predict
            outputs = self.model.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                price_unit_ids=price_unit_id,  # NEW
                property_ids=property_ids,
                property_values=property_values,
                property_units=property_units,
                property_mask=property_mask,
            )

            log_price = outputs["predictions"][0].item()
            price = torch.exp(outputs["predictions"][0]).item()

            return {
                "price": round(price, 2),
                "log_price": round(log_price, 4),
                "currency": "EUR",
            }

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
        if properties_list is None:
            properties_list = [None] * len(texts)

        return [
            self.predict(text, properties)
            for text, properties in zip(texts, properties_list)
        ]


__all__ = [
    "PriceRegressor",
    "PricePredictionPipeline",
    "UNIT_MAP",
    "get_unit_id",
    "PRICE_UNIT_MAP",
    "get_price_unit_id",
]
