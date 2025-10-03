"""End-to-end pipeline: Classification + Span Extraction + Parsing.

This pipeline combines:
1. roBERTino: BIM category classification
2. Span Extractor: Find relevant text spans for properties
3. Parsers/Regex: Extract structured values from spans
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoTokenizer

from robimb.models.label_model import load_label_embed_model
from robimb.models.span_extractor import PropertyExtractorPipeline, PropertySpanExtractor
from robimb.extraction.parsers.dimensions import parse_dimension_pattern
from robimb.extraction.parsers.units import parse_number_with_unit


class SmartExtractionPipeline:
    """Complete extraction pipeline: classify → find spans → parse values."""

    def __init__(
        self,
        classifier_model_path: Path,
        span_extractor_model_path: Path,
        device: str = "cpu",
        hf_token: Optional[str] = None,
    ):
        """Initialize pipeline with both models.

        Args:
            classifier_model_path: Path to roBERTino (classification model)
            span_extractor_model_path: Path to trained span extractor
            device: "cpu" or "cuda"
            hf_token: HuggingFace token for private models
        """
        self.device = device
        self.hf_token = hf_token

        # 1. Load classifier (roBERTino)
        print("Loading classifier (roBERTino)...")
        self.classifier = load_label_embed_model(
            str(classifier_model_path),
            backbone_src=classifier_model_path,
        )
        self.classifier.eval()
        self.classifier.to(device)

        self.classifier_tokenizer = AutoTokenizer.from_pretrained(
            str(classifier_model_path),
            token=hf_token,
        )

        # Load category labels
        config = self.classifier.config
        self.super_labels = getattr(config, "label_texts_super", [])
        self.cat_labels = getattr(config, "label_texts_cat", [])

        # 2. Load span extractor
        print("Loading span extractor...")
        import json

        with (span_extractor_model_path / "property_id_map.json").open() as f:
            property_id_map = json.load(f)

        checkpoint = torch.load(
            span_extractor_model_path / "best_model.pt",
            map_location=device,
        )

        span_model = PropertySpanExtractor(
            backbone_name=str(span_extractor_model_path),
            num_properties=len(property_id_map),
            hf_token=hf_token,
        )
        span_model.load_state_dict(checkpoint["model_state_dict"])

        span_tokenizer = AutoTokenizer.from_pretrained(
            str(span_extractor_model_path),
            token=hf_token,
        )

        self.span_pipeline = PropertyExtractorPipeline(
            model=span_model,
            tokenizer=span_tokenizer,
            property_id_map=property_id_map,
            device=device,
        )

        self.property_id_map = property_id_map

        print("Pipeline ready!")

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text into BIM categories.

        Args:
            text: Product description

        Returns:
            Dict with:
                - supercategory: str
                - category: str
                - confidence_super: float
                - confidence_cat: float
        """
        # Tokenize
        inputs = self.classifier_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward
        with torch.no_grad():
            outputs = self.classifier(**inputs)

        # Get predictions
        super_logits = outputs["logits_super"][0]
        cat_logits = outputs["logits_cat_pred_masked"][0]

        super_pred = super_logits.argmax().item()
        cat_pred = cat_logits.argmax().item()

        # Confidence (softmax)
        super_conf = torch.softmax(super_logits, dim=0)[super_pred].item()
        cat_conf = torch.softmax(cat_logits, dim=0)[cat_pred].item()

        return {
            "supercategory": self.super_labels[super_pred] if super_pred < len(self.super_labels) else "Unknown",
            "category": self.cat_labels[cat_pred] if cat_pred < len(self.cat_labels) else "Unknown",
            "confidence_super": super_conf,
            "confidence_cat": cat_conf,
            "super_id": super_pred,
            "cat_id": cat_pred,
        }

    def extract_properties(
        self,
        text: str,
        property_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract property spans and parse values.

        Args:
            text: Product description
            property_ids: Which properties to extract (None = all)

        Returns:
            Dict mapping property_id to:
                - raw_text: str (extracted span)
                - span: (start, end)
                - value: parsed value
                - unit: str or None
                - confidence: float
        """
        # Get spans from model
        span_results = self.span_pipeline.extract(text, property_ids)

        # Parse spans to values
        final_results = {}

        for prop_id, span_result in span_results.items():
            raw_text = span_result["raw_text"]

            if not raw_text or not raw_text.strip():
                continue

            # Apply domain-specific parsers
            parsed = self._parse_span(raw_text, prop_id, text)

            if parsed:
                final_results[prop_id] = {
                    "raw_text": raw_text,
                    "span": span_result["span"],
                    "value": parsed["value"],
                    "unit": parsed.get("unit"),
                    "confidence": min(span_result["confidence"], parsed["confidence"]),
                    "source": "span_extractor",
                }

        return final_results

    def _parse_span(
        self,
        raw_text: str,
        property_id: str,
        full_text: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply appropriate parser to extracted span."""

        # Dimensioni
        if property_id in ["dimensione_lunghezza", "dimensione_larghezza", "dimensione_altezza"]:
            result = parse_dimension_pattern(raw_text)
            if result:
                return {
                    "value": result.get("value_mm"),
                    "unit": "mm",
                    "confidence": 0.9,
                }

        # Portata
        elif property_id == "portata_l_min":
            result = parse_number_with_unit(raw_text)
            if result and "l/min" in raw_text.lower():
                return {
                    "value": result["value"],
                    "unit": "l/min",
                    "confidence": 0.9,
                }

        # Marchio
        elif property_id == "marchio":
            brand = raw_text.strip()
            if brand and len(brand) > 1:
                return {
                    "value": brand,
                    "unit": None,
                    "confidence": 0.85,
                }

        # Materiale
        elif property_id == "materiale":
            return {
                "value": raw_text.strip().lower(),
                "unit": None,
                "confidence": 0.8,
            }

        # Tipologia installazione
        elif property_id == "tipologia_installazione":
            value = raw_text.strip().lower().replace(" ", "_")
            return {
                "value": value,
                "unit": None,
                "confidence": 0.85,
            }

        # Spessore
        elif property_id == "spessore_mm":
            result = parse_number_with_unit(raw_text)
            if result:
                return {
                    "value": result["value"],
                    "unit": "mm",
                    "confidence": 0.9,
                }

        # Default: return raw text
        return {
            "value": raw_text.strip(),
            "unit": None,
            "confidence": 0.7,
        }

    def process(
        self,
        text: str,
        property_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Complete pipeline: classify + extract + parse.

        Args:
            text: Product description
            property_ids: Which properties to extract (None = all)

        Returns:
            Dict with:
                - classification: category info
                - properties: extracted properties
        """
        # Step 1: Classify
        classification = self.classify(text)

        # Step 2: Extract properties
        properties = self.extract_properties(text, property_ids)

        return {
            "text": text,
            "classification": classification,
            "properties": properties,
        }


def demo():
    """Demo of complete pipeline."""
    from pathlib import Path
    import os
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Load HF token
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    hf_token = os.getenv("HF_TOKEN")

    # Paths (TODO: update when models are trained)
    classifier_path = Path("path/to/roBERTino")  # Your roBERTino checkpoint
    span_extractor_path = PROJECT_ROOT / "outputs" / "span_extractor_model"

    # Initialize pipeline
    pipeline = SmartExtractionPipeline(
        classifier_model_path=classifier_path,
        span_extractor_model_path=span_extractor_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hf_token=hf_token,
    )

    # Test examples
    examples = [
        "Pavimento in gres porcellanato tipo Florim, dimensioni 120x280 cm, spessore 6 mm",
        "Miscelatore monocomando per lavabo Grohe Essence, portata 5.7 l/min",
        "Box doccia in cristallo tipo ARTICA, apertura angolare, 90x70x190 cm",
    ]

    print("\n=== SMART EXTRACTION PIPELINE ===\n")

    for i, text in enumerate(examples, 1):
        print(f"Example {i}: {text[:60]}...")
        result = pipeline.process(text)

        print(f"  Category: {result['classification']['category']}")
        print(f"  Confidence: {result['classification']['confidence_cat']:.2f}")
        print("  Properties:")
        for prop_id, prop_data in result["properties"].items():
            print(f"    {prop_id}: {prop_data['value']} (conf: {prop_data['confidence']:.2f})")
        print()


if __name__ == "__main__":
    demo()
