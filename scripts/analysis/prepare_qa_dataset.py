"""Prepare QA-style dataset for property extraction from estrazione_cartongesso.jsonl.

This script:
1. Filters out false positives (e.g., "compensato" from "compreso e compensato")
2. Creates QA pairs: (text, property_name) -> (start_pos, end_pos, value)
3. Exports clean dataset for fine-tuning a span-extraction model
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# False positive patterns to filter out
FALSE_POSITIVE_PATTERNS = {
    "materiale": [
        # "compensato" from "compreso e compensato"
        (r"compreso\s+e\s+compensato", "compensato"),
        # "autolivellante" when main material is different
        (r"pavimento\s+in\s+\w+.*autolivellante", "autolivellante"),
        (r"linoleum.*autolivellante", "autolivellante"),
        (r"pvc.*autolivellante", "autolivellante"),
        # Secondary materials
        (r"sottofondo|massetto", "sottofondo"),
        (r"sottofondo|massetto", "massetto"),
    ],
}

def is_false_positive(
    text: str,
    prop_id: str,
    raw: str,
    span: Optional[Tuple[int, int]]
) -> bool:
    """Check if extraction is a false positive."""
    if not span or not raw:
        return False

    start, end = span

    # General false positive patterns
    if prop_id in FALSE_POSITIVE_PATTERNS:
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        context = text[context_start:context_end].lower()

        for pattern, target_raw in FALSE_POSITIVE_PATTERNS[prop_id]:
            if target_raw.lower() == raw.lower() and re.search(pattern, context, re.IGNORECASE):
                return True

    # Special validation for materiale
    if prop_id == "materiale":
        # Check 1: Partial word (not word boundary)
        if start > 0 and text[start-1].isalnum():
            return True  # Partial word like "mma" from "gomma"

        # Check 2: From "effetto X" (aesthetic, not material)
        context_before = text[max(0, start-30):start].lower()
        if "effetto" in context_before and len(raw) < 15:
            return True

        # Check 3: Main material mentioned earlier (search wider context)
        context_window = text[max(0, start-500):start].lower()
        main_materials = [
            r"pavimento\s+in\s+linoleum",
            r"pavimento\s+in\s+pvc",
            r"pavimento\s+in\s+vinilico",
            r"pavimento\s+in\s+gomma",
            r"pavimento\s+in\s+gres",
            r"pavimento\s+in\s+ceramica",
        ]
        for mat_pattern in main_materials:
            match = re.search(mat_pattern, context_window)
            if match:
                main_mat = match.group(0).split()[-1]
                # If found main material and current answer is NOT it -> filter out
                if main_mat not in raw.lower() and raw.lower() not in ["linoleum", "pvc", "vinilico", "gomma", "gres", "ceramica"]:
                    return True  # Different from main material

    return False


def extract_qa_pairs(input_path: Path) -> List[Dict]:
    """Extract QA pairs from dataset.

    Returns:
        List of dicts with:
        - text: str
        - property_id: str
        - property_name: str (human-readable)
        - start_char: int
        - end_char: int
        - raw_text: str
        - value: any
        - confidence: float
        - source: str
    """
    qa_pairs = []

    property_names = {
        "marchio": "marchio",
        "materiale": "materiale",
        "dimensione_lunghezza": "lunghezza",
        "dimensione_larghezza": "larghezza",
        "dimensione_altezza": "altezza",
        "tipologia_installazione": "tipo di installazione",
        "portata_l_min": "portata",
        "normativa_riferimento": "normativa di riferimento",
        "classe_ei": "classe di resistenza al fuoco",
        "classe_reazione_al_fuoco": "classe di reazione al fuoco",
        "presenza_isolante": "presenza di isolante",
        "stratigrafia_lastre": "stratigrafia delle lastre",
        "spessore_mm": "spessore",
        "materiale_struttura": "materiale della struttura",
        "formato": "formato",
        "spessore_pannello_mm": "spessore del pannello",
        "trasmittanza_termica": "trasmittanza termica",
        "isolamento_acustico_db": "isolamento acustico",
        "colore_ral": "colore",
        "coefficiente_fonoassorbimento": "coefficiente di fonoassorbimento",
    }

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON at line {line_num}")
                continue

            text = data.get("text", "")
            if not text:
                continue

            for prop_id, prop_data in data.get("properties", {}).items():
                # Skip if no value extracted
                if prop_data.get("value") is None:
                    continue

                # Skip if no span/raw text
                span = prop_data.get("span")
                raw = prop_data.get("raw")
                if not span or not raw:
                    continue

                # Filter false positives
                if is_false_positive(text, prop_id, raw, span):
                    continue

                # Skip low confidence extractions
                confidence = prop_data.get("confidence", 0.0)
                if confidence < 0.5:
                    continue

                # Skip heuristic sources (we want learned patterns)
                source = prop_data.get("source", "")
                if "heuristic" in source:
                    continue

                start, end = span

                qa_pairs.append({
                    "text": text,
                    "property_id": prop_id,
                    "property_name": property_names.get(prop_id, prop_id),
                    "start_char": start,
                    "end_char": end,
                    "raw_text": raw,
                    "value": prop_data["value"],
                    "confidence": confidence,
                    "source": source,
                })

    return qa_pairs


def create_training_examples(qa_pairs: List[Dict]) -> List[Dict]:
    """Convert QA pairs to training examples for span extraction.

    Format similar to SQuAD:
    {
        "context": text,
        "question": "What is the {property_name}?",
        "answers": {
            "text": [raw_text],
            "answer_start": [start_char]
        }
    }
    """
    examples = []

    for qa in qa_pairs:
        # Determine article based on property name (Italian grammar)
        property_name = qa['property_name']
        if property_name in ['lunghezza', 'larghezza', 'altezza', 'portata',
                              'normativa di riferimento', 'classe di resistenza al fuoco',
                              'classe di reazione al fuoco', 'presenza di isolante',
                              'stratigrafia delle lastre', 'trasmittanza termica']:
            article = "la"
        else:
            article = "il"

        example = {
            "context": qa["text"],
            "question": f"Qual è {article} {property_name}?",
            "property_id": qa["property_id"],
            "answers": {
                "text": [qa["raw_text"]],
                "answer_start": [qa["start_char"]],
            },
            "value": qa["value"],
            "confidence": qa["confidence"],
            "source": qa["source"],
        }
        examples.append(example)

    return examples


def main():
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "outputs" / "estrazione_cartongesso.jsonl"
    output_dir = project_root / "outputs" / "qa_dataset"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Reading from: {input_path}")
    qa_pairs = extract_qa_pairs(input_path)
    print(f"Extracted {len(qa_pairs)} QA pairs")

    # Stats by property
    from collections import Counter
    prop_counts = Counter(qa["property_id"] for qa in qa_pairs)
    print("\nQA pairs by property:")
    for prop_id, count in prop_counts.most_common():
        print(f"  {prop_id:30s}: {count:4d}")

    # Create training examples
    examples = create_training_examples(qa_pairs)

    # Save as JSONL
    output_path = output_dir / "property_extraction_qa.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(examples)} training examples to: {output_path}")

    # Also save a sample for inspection
    sample_path = output_dir / "sample_qa_pairs.json"
    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(examples[:10], f, indent=2, ensure_ascii=False)

    print(f"Saved sample to: {sample_path}")


if __name__ == "__main__":
    main()
