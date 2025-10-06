"""Inference utilities for span-based property extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from ..models.span_extractor import PropertySpanExtractor, PropertyExtractorPipeline
from ..extraction.parsers.dimensions import parse_dimensions
from ..extraction.parsers.numbers import extract_numbers

__all__ = ["SpanInference", "apply_parser_to_span"]


def apply_parser_to_span(
    raw_text: str,
    property_id: str,
    full_text: str,
) -> Optional[Dict[str, any]]:
    """Apply appropriate parser/regex to extracted span.

    Args:
        raw_text: The text span extracted by the model
        property_id: Which property we're extracting
        full_text: Full context (for better parsing)

    Returns:
        Dict with parsed value, unit, confidence, etc.
    """
    # Dimensioni
    if property_id in ["dimensione_lunghezza", "dimensione_larghezza", "dimensione_altezza"]:
        matches = list(parse_dimensions(raw_text))
        if matches:
            result = matches[0]
            return {
                "value": result.values_mm[0] if result.values_mm else None,
                "unit": "mm",
                "confidence": 0.9,
                "raw": raw_text,
            }

    # Portata
    elif property_id == "portata_l_min":
        numbers = list(extract_numbers(raw_text))
        if numbers and "l/min" in raw_text.lower():
            return {
                "value": numbers[0].value,
                "unit": "l/min",
                "confidence": 0.9,
                "raw": raw_text,
            }

    # Marchio
    elif property_id == "marchio":
        # Clean up common words
        brand = raw_text.strip()
        if brand and len(brand) > 1:
            return {
                "value": brand,
                "unit": None,
                "confidence": 0.85,
                "raw": raw_text,
            }

    # Materiale
    elif property_id == "materiale":
        # Use the extracted span directly
        return {
            "value": raw_text.strip().lower(),
            "unit": None,
            "confidence": 0.8,
            "raw": raw_text,
        }

    # Tipologia installazione
    elif property_id == "tipologia_installazione":
        value = raw_text.strip().lower().replace(" ", "_")
        return {
            "value": value,
            "unit": None,
            "confidence": 0.85,
            "raw": raw_text,
        }

    # Spessore
    elif property_id in ["spessore_mm", "spessore_pannello_mm"]:
        numbers = list(extract_numbers(raw_text))
        if numbers:
            return {
                "value": numbers[0].value,
                "unit": "mm",
                "confidence": 0.9,
                "raw": raw_text,
            }

    # Default: return raw text
    return {
        "value": raw_text.strip(),
        "unit": None,
        "confidence": 0.7,
        "raw": raw_text,
    }


class SpanInference:
    """Combines span extraction model with domain-specific parsers."""

    def __init__(
        self,
        model_dir: Path,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load property map
        property_map_path = model_dir / "property_id_map.json"
        if not property_map_path.exists():
            raise FileNotFoundError(f"Property map not found at {property_map_path}")

        with property_map_path.open() as f:
            property_id_map = json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Load model checkpoint
        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / "final_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found in {model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model config
        config = checkpoint.get("config", {})
        backbone_name = config.get("backbone_name", "dbmdz/bert-base-italian-xxl-cased")
        num_properties = config.get("num_properties", len(property_id_map))
        dropout = config.get("dropout", 0.1)

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
        model = PropertySpanExtractor(
            backbone_name=backbone_name,
            num_properties=num_properties,
            dropout=dropout,
            hf_token=hf_token,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create pipeline
        self.pipeline = PropertyExtractorPipeline(
            model=model,
            tokenizer=tokenizer,
            property_id_map=property_id_map,
            device=device,
        )

        self.property_id_map = property_id_map
        self.model = model
        self.tokenizer = tokenizer

    def extract_properties(
        self,
        text: str,
        property_ids: Optional[List[str]] = None,
        apply_parsers: bool = True,
    ) -> Dict[str, Dict[str, any]]:
        """Extract properties from text.

        Args:
            text: Product description
            property_ids: List of properties to extract (None = all)
            apply_parsers: Whether to apply domain-specific parsers

        Returns:
            Dict mapping property_id to extraction:
            {
                "marchio": {
                    "value": "Grohe",
                    "raw": "Grohe",
                    "span": (47, 52),
                    "confidence": 0.95,
                    "unit": None,
                },
                ...
            }
        """
        if property_ids is None:
            property_ids = list(self.property_id_map.keys())

        # Step 1: Extract spans using model
        span_results = self.pipeline.extract(text, property_ids)

        # Step 2: Apply parsers to spans (optional)
        final_results = {}

        for prop_id, span_result in span_results.items():
            raw_text = span_result["raw_text"]

            # Skip empty spans
            if not raw_text or not raw_text.strip():
                continue

            if apply_parsers:
                # Apply parser
                parsed = apply_parser_to_span(raw_text, prop_id, text)

                if parsed:
                    final_results[prop_id] = {
                        "value": parsed["value"],
                        "raw": raw_text,
                        "span": span_result["span"],
                        "confidence": min(span_result["confidence"], parsed["confidence"]),
                        "unit": parsed.get("unit"),
                        "source": "span_extractor",
                    }
            else:
                # Return raw span without parsing
                final_results[prop_id] = {
                    "value": raw_text.strip(),
                    "raw": raw_text,
                    "span": span_result["span"],
                    "confidence": span_result["confidence"],
                    "unit": None,
                    "source": "span_extractor",
                }

        return final_results

    def extract_batch(
        self,
        texts: List[str],
        property_ids: Optional[List[str]] = None,
        apply_parsers: bool = True,
    ) -> List[Dict[str, Dict[str, any]]]:
        """Extract properties from multiple texts.

        Args:
            texts: List of product descriptions
            property_ids: List of properties to extract (None = all)
            apply_parsers: Whether to apply domain-specific parsers

        Returns:
            List of extraction results, one per input text
        """
        return [
            self.extract_properties(text, property_ids, apply_parsers)
            for text in texts
        ]
