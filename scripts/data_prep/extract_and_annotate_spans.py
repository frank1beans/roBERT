"""Extract properties and annotate spans for training data.

This script:
1. Loads filtered price data with property_ids
2. Extracts properties using deterministic parsers/matchers
3. Finds spans (start/end positions) in the text
4. Generates span annotation format for training
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Import extraction utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.robimb.extraction.parsers.dimensions import parse_dimensions
from src.robimb.extraction.parsers.numbers import extract_numbers
from src.robimb.extraction.cartongesso import extract_cartongesso_features


def find_span_in_text(text: str, value_str: str, property_id: str) -> Optional[Tuple[int, int]]:
    """Find the span (start, end) of a value in text.

    Args:
        text: Full text to search
        value_str: Value string to find
        property_id: Property type (for context-aware search)

    Returns:
        (start, end) tuple or None if not found
    """
    # Clean value string
    value_clean = str(value_str).strip()

    # Try exact match first (case-insensitive)
    pattern = re.escape(value_clean)
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return (match.start(), match.end())

    # Try fuzzy match for numbers (with different separators)
    if property_id in ["spessore_mm", "dimensione_lunghezza", "dimensione_larghezza"]:
        # Try variations: "12.5", "12,5", "12.5mm", etc.
        number_part = re.sub(r'[^\d.,]', '', value_clean)
        if number_part:
            # Look for number with optional unit nearby
            pattern = rf'\b{re.escape(number_part)}\s*mm\b'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (match.start(), match.end())

            # Just the number
            pattern = rf'\b{re.escape(number_part)}\b'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return (match.start(), match.end())

    # Try partial match for brands/strings
    if len(value_clean) > 3:
        # Look for the value as a whole word
        pattern = rf'\b{re.escape(value_clean)}\b'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return (match.start(), match.end())

    return None


def extract_property_value(text: str, property_id: str, category: str = "") -> Optional[str]:
    """Extract property value using deterministic parsers.

    Args:
        text: Product description
        property_id: Property to extract
        category: Category name for context-aware extraction

    Returns:
        Extracted value as string or None
    """
    text_lower = text.lower()

    # Use cartongesso extractor for drywall categories
    if "cartongesso" in category.lower() or "controsoffit" in category.lower():
        features = extract_cartongesso_features(text)
        if features:
            if property_id == "spessore_mm" and features.layers:
                return f"{features.layers[0].thickness_mm:.1f} mm"
            elif property_id == "tipologia_lastra" and features.layers:
                return features.layers[0].type
            elif property_id == "classe_reazione_al_fuoco" and features.reaction_class:
                return features.reaction_class
            elif property_id == "classe_ei" and features.rei_class:
                return features.rei_class
            elif property_id == "presenza_isolante":
                return "sì" if features.insulation_material else "no"
            elif property_id == "stratigrafia_lastre" and features.layers:
                return " + ".join([f"{l.type} {l.thickness_mm}mm" for l in features.layers])

    # Marchio (brand) extraction
    if property_id == "marchio":
        # Common brands in construction
        brands = [
            "Knauf", "Gyproc", "Rigips", "Siniat", "Saint-Gobain",
            "Mapei", "Sika", "Weber", "Kerakoll", "Fassa Bortolo",
            "Grohe", "Hansgrohe", "Geberit", "Ideal Standard", "Roca",
            "Atlas Concorde", "Marazzi", "Ceramiche Refin", "Mirage",
            "Caparol", "Metecno", "Lecablocco"
        ]
        for brand in brands:
            if brand.lower() in text_lower:
                # Find exact occurrence
                match = re.search(rf'\b{re.escape(brand)}\b', text, re.IGNORECASE)
                if match:
                    return text[match.start():match.end()]
        return None

    # Spessore (thickness) extraction
    elif property_id == "spessore_mm":
        # Look for patterns like "sp. 12.5 mm", "spessore 40mm", etc.
        patterns = [
            r'sp\.?\s*(\d+(?:[.,]\d+)?)\s*mm',
            r'spessore\s*(?:di)?\s*(\d+(?:[.,]\d+)?)\s*mm',
            r'(?:sp|spessore)\.?\s*(\d+(?:[.,]\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1).replace(',', '.')
                return f"{value} mm"
        return None

    # Tipologia lastra (board type)
    elif property_id == "tipologia_lastra":
        if "idrofug" in text_lower or "idrorepellent" in text_lower:
            return "idrofuga"
        elif "ignifug" in text_lower or "resistente al fuoco" in text_lower:
            return "ignifuga"
        elif "acustic" in text_lower or "fono" in text_lower:
            return "acustica"
        elif "standard" in text_lower:
            return "standard"
        return None

    # Classe reazione al fuoco
    elif property_id == "classe_reazione_al_fuoco":
        # Look for patterns like "Classe A1", "Euroclasse A2-s1,d0"
        patterns = [
            r'classe\s+([A-F]\d?(?:-s\d+)?(?:,d\d+)?)',
            r'euroclasse\s+([A-F]\d?(?:-s\d+)?(?:,d\d+)?)',
            r'\b([A-F]\d)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    # Classe EI (fire resistance)
    elif property_id == "classe_ei":
        patterns = [
            r'((?:REI|EI)\s*\d+)',
            r'resistenza al fuoco[:\s]+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    # Default: try to find with parsers
    else:
        # Use dimension parser
        if "dimension" in property_id or "lunghezza" in property_id:
            dims = list(parse_dimensions(text))
            if dims:
                return f"{dims[0].values_mm[0]:.1f} mm"

        # Use number extractor
        numbers = list(extract_numbers(text))
        if numbers and len(numbers) > 0:
            return str(numbers[0].value)

    return None


def process_record(record: Dict) -> Optional[Dict]:
    """Process a single record to extract properties and find spans.

    Args:
        record: Input record with text and property_ids

    Returns:
        Record with span annotations or None if no spans found
    """
    text = record["text"]
    property_ids = record.get("property_ids", [])

    spans = {}
    extracted_values = {}

    category = record.get("cat", "")
    super_category = record.get("super", "")

    for prop_id in property_ids:
        # Extract value
        value = extract_property_value(text, prop_id, category or super_category)
        if value:
            # Find span in text
            span = find_span_in_text(text, value, prop_id)
            if span:
                spans[prop_id] = {
                    "text": value,
                    "start": span[0],
                    "end": span[1],
                }
                extracted_values[prop_id] = value

    # Only return if we found at least one span
    if not spans:
        return None

    return {
        "text": text,
        "spans": spans,
        "properties": extracted_values,
        "category": record.get("cat"),
        "super_category": record.get("super"),
    }


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent.parent

    # Paths
    input_file = base_dir / "resources/data/train/span/train_filtered.jsonl"
    output_file = base_dir / "resources/data/train/span/train_spans.jsonl"

    print("=" * 60)
    print("EXTRACTING PROPERTIES AND ANNOTATING SPANS")
    print("=" * 60)

    # Process records
    stats = defaultdict(int)
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            record = json.loads(line)
            stats["total"] += 1

            # Process record
            result = process_record(record)

            if result:
                stats["with_spans"] += 1
                stats["total_spans"] += len(result["spans"])

                # Write to output
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                stats["no_spans"] += 1

    print(f"\nProcessed {stats['total']} records")
    print(f"  Records with spans: {stats['with_spans']}")
    print(f"  Records without spans: {stats['no_spans']}")
    print(f"  Total spans extracted: {stats['total_spans']}")
    print(f"  Average spans per record: {stats['total_spans']/stats['with_spans']:.2f}")
    print(f"\nOutput written to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
