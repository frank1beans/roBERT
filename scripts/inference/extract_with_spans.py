"""Inference script: span extractor + regex parser pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from robimb.models.span_extractor import PropertySpanExtractor, PropertyExtractorPipeline
from robimb.extraction.parsers.dimensions import parse_dimension_pattern
from robimb.extraction.parsers.units import parse_number_with_unit
from robimb.extraction.matchers.brands import match_brand


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
        result = parse_dimension_pattern(raw_text)
        if result:
            return {
                "value": result.get("value_mm"),
                "unit": "mm",
                "confidence": 0.9,
                "raw": raw_text,
            }

    # Portata
    elif property_id == "portata_l_min":
        result = parse_number_with_unit(raw_text)
        if result and "l/min" in raw_text.lower():
            return {
                "value": result["value"],
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

    # Default: return raw text
    return {
        "value": raw_text.strip(),
        "unit": None,
        "confidence": 0.7,
        "raw": raw_text,
    }


class SmartPropertyExtractor:
    """Combines span extraction model with domain-specific parsers."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
    ):
        self.device = device

        # Load property map
        with (model_dir / "property_id_map.json").open() as f:
            property_id_map = json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Load model
        checkpoint = torch.load(
            model_dir / "best_model.pt",
            map_location=device,
        )

        model = PropertySpanExtractor(
            backbone_name=str(model_dir),
            num_properties=len(property_id_map),
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

    def extract_properties(
        self,
        text: str,
        property_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, any]]:
        """Extract properties from text.

        Args:
            text: Product description
            property_ids: List of properties to extract (None = all)

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

        # Step 2: Apply parsers to spans
        final_results = {}

        for prop_id, span_result in span_results.items():
            raw_text = span_result["raw_text"]

            # Skip empty spans
            if not raw_text or not raw_text.strip():
                continue

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

        return final_results


def main():
    """Demo inference."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_DIR = PROJECT_ROOT / "outputs" / "span_extractor_model"

    if not MODEL_DIR.exists():
        print(f"Model not found at {MODEL_DIR}")
        print("Please train the model first using: python scripts/training/train_span_extractor.py")
        return

    # Initialize extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SmartPropertyExtractor(MODEL_DIR, device=device)

    # Test examples
    examples = [
        "PX15 - Fornitura e posa in opera di griglia in acciaio per scarico a pavimento tipo AISI 304, dim. 20x20 cm.",
        "Miscelatore monocomando per lavabo tipo Grohe, serie Essence, taglia S",
        "Box doccia in cristallo, Tipo ARTICA apertura angolare scorrevole. Box 90x70 h190 sp.6mm.",
    ]

    print("Smart Property Extraction Demo\n")

    for i, text in enumerate(examples, 1):
        print(f"Example {i}: {text[:80]}...")
        print("-" * 80)

        results = extractor.extract_properties(text)

        for prop_id, result in results.items():
            print(f"  {prop_id:30s}: {result['value']}")
            print(f"    Raw: \"{result['raw']}\"")
            print(f"    Span: {result['span']}")
            print(f"    Confidence: {result['confidence']:.3f}")

        print()


if __name__ == "__main__":
    main()
