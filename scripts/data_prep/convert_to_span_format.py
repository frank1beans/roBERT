"""Convert extracted properties to span training format.

Input: train_extracted.jsonl (from `robimb extract properties`)
Output: span training format with answer spans for QA-style training

Format:
{
    "text": "...",
    "properties": [
        {
            "id": "marchio",
            "question": "Qual è il marchio?",
            "answer": {
                "text": "Knauf",
                "start": 290,
                "end": 295
            },
            "confidence": 0.85
        },
        ...
    ]
}
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


# Property to question mapping
PROPERTY_QUESTIONS = {
    "marchio": "Qual è il marchio?",
    "materiale": "Qual è il materiale?",
    "spessore_mm": "Qual è lo spessore in millimetri?",
    "classe_reazione_al_fuoco": "Qual è la classe di reazione al fuoco?",
    "classe_ei": "Qual è la classe EI di resistenza al fuoco?",
    "tipologia_lastra": "Qual è la tipologia di lastra?",
    "presenza_isolante": "È presente isolante?",
    "dimensione_lunghezza": "Qual è la lunghezza?",
    "dimensione_larghezza": "Qual è la larghezza?",
    "dimensione_altezza": "Qual è l'altezza?",
    "formato": "Qual è il formato?",
    "finitura": "Qual è la finitura?",
    "posa": "Qual è il tipo di posa?",
    "colore_ral": "Qual è il colore RAL?",
    "portata_l_min": "Qual è la portata in litri al minuto?",
    "tipologia_installazione": "Qual è la tipologia di installazione?",
    "spessore_pannello_mm": "Qual è lo spessore del pannello in millimetri?",
    "coefficiente_fonoassorbimento": "Qual è il coefficiente di fonoassorbimento?",
}


def convert_property_to_span(prop_id: str, prop_data: Dict) -> Optional[Dict]:
    """Convert a single property extraction to span format.

    Args:
        prop_id: Property identifier
        prop_data: Property data with value, span, confidence, etc.

    Returns:
        Span format dict or None if no valid span
    """
    # Skip if no value
    if prop_data.get("value") is None:
        return None

    # Skip if no span (can't train on it)
    span = prop_data.get("span")
    if not span:
        # Try to get from start/end
        start = prop_data.get("start")
        end = prop_data.get("end")
        if start is not None and end is not None:
            span = [start, end]
        else:
            return None

    # Get question for this property
    question = PROPERTY_QUESTIONS.get(prop_id, f"Qual è {prop_id}?")

    # Get answer text
    answer_text = prop_data.get("raw") or str(prop_data.get("value", ""))

    # Handle complex values (like stratigrafia_lastre)
    if isinstance(prop_data.get("value"), dict):
        # Skip complex nested structures for now
        # TODO: could extract sub-properties
        return None

    return {
        "id": prop_id,
        "question": question,
        "answer": {
            "text": answer_text,
            "start": span[0],
            "end": span[1],
        },
        "confidence": prop_data.get("confidence", 0.5),
        "source": prop_data.get("source", "unknown"),
    }


def convert_record(record: Dict) -> Optional[Dict]:
    """Convert a single extracted record to span training format.

    Args:
        record: Record from train_extracted.jsonl

    Returns:
        Span training format dict or None if no valid spans
    """
    text = record.get("text", "")
    properties = record.get("properties", {})

    # Convert each property to span format
    span_properties = []
    for prop_id, prop_data in properties.items():
        span_prop = convert_property_to_span(prop_id, prop_data)
        if span_prop:
            span_properties.append(span_prop)

    # Skip if no valid spans
    if not span_properties:
        return None

    return {
        "text": text,
        "category": record.get("cat", ""),
        "super_category": record.get("super", ""),
        "properties": span_properties,
        "num_properties": len(span_properties),
    }


def main():
    """Main conversion script."""
    base_dir = Path(__file__).parent.parent.parent

    # Paths
    input_file = base_dir / "resources/data/train/span/train_extracted.jsonl"
    output_file = base_dir / "resources/data/train/span/train_span_format.jsonl"

    print("=" * 60)
    print("CONVERTING TO SPAN TRAINING FORMAT")
    print("=" * 60)

    stats = defaultdict(int)
    property_stats = defaultdict(int)

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            record = json.loads(line)
            stats["total"] += 1

            # Convert to span format
            span_record = convert_record(record)

            if span_record:
                stats["with_spans"] += 1
                stats["total_spans"] += span_record["num_properties"]

                # Track property counts
                for prop in span_record["properties"]:
                    property_stats[prop["id"]] += 1

                # Write to output
                f_out.write(json.dumps(span_record, ensure_ascii=False) + "\n")
            else:
                stats["no_spans"] += 1

    # Print statistics
    print(f"\nProcessed {stats['total']} records")
    print(f"  Records with spans: {stats['with_spans']}")
    print(f"  Records without spans: {stats['no_spans']}")
    print(f"  Total spans: {stats['total_spans']}")
    print(f"  Average spans per record: {stats['total_spans']/stats['with_spans']:.2f}")

    print(f"\nTop 15 properties by span count:")
    for prop, count in sorted(property_stats.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {prop}: {count}")

    print(f"\nOutput written to: {output_file}")
    print("=" * 60)

    # Print sample record
    print("\nSample record:")
    with open(output_file, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:1000] + "...")


if __name__ == "__main__":
    main()
