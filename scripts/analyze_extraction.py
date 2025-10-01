#!/usr/bin/env python3
"""Analyze property extraction results."""
import json
from pathlib import Path
from collections import defaultdict, Counter

def analyze_extraction(jsonl_path: str):
    """Analyze extraction results from JSONL file."""
    path = Path(jsonl_path)

    if not path.exists():
        print(f"File not found: {jsonl_path}")
        return

    total_docs = 0
    total_properties = 0
    extracted_properties = 0
    validation_ok = 0
    validation_failed = 0

    confidence_scores = []
    sources = Counter()
    categories = Counter()
    property_extraction_rate = defaultdict(lambda: {"extracted": 0, "total": 0})

    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            doc = json.loads(line)
            total_docs += 1

            # Category
            cat = doc.get("categoria", "unknown")
            categories[cat] += 1

            # Validation
            validation = doc.get("validation", {})
            if validation.get("status") == "ok":
                validation_ok += 1
            else:
                validation_failed += 1

            # Confidence
            conf = doc.get("confidence_overall", 0.0)
            confidence_scores.append(conf)

            # Properties
            props = doc.get("properties", {})
            for prop_id, prop_data in props.items():
                total_properties += 1
                property_extraction_rate[prop_id]["total"] += 1

                value = prop_data.get("value")
                if value is not None and value != "":
                    extracted_properties += 1
                    property_extraction_rate[prop_id]["extracted"] += 1

                    source = prop_data.get("source")
                    if source:
                        sources[source] += 1

    # Calculate stats
    extraction_rate = (extracted_properties / total_properties * 100) if total_properties > 0 else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    validation_rate = (validation_ok / total_docs * 100) if total_docs > 0 else 0

    # Print report
    print("=" * 80)
    print(f"EXTRACTION ANALYSIS: {path.name}")
    print("=" * 80)
    print(f"\nOVERALL STATISTICS")
    print(f"  Total documents:        {total_docs}")
    print(f"  Total properties:       {total_properties}")
    print(f"  Extracted properties:   {extracted_properties} ({extraction_rate:.1f}%)")
    print(f"  Average confidence:     {avg_confidence:.3f}")
    print(f"  Validation OK:          {validation_ok} ({validation_rate:.1f}%)")
    print(f"  Validation FAILED:      {validation_failed} ({100-validation_rate:.1f}%)")

    print(f"\nEXTRACTION BY SOURCE")
    for source, count in sources.most_common():
        pct = count / extracted_properties * 100 if extracted_properties > 0 else 0
        print(f"  {source:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nCATEGORIES")
    for cat, count in categories.most_common():
        pct = count / total_docs * 100 if total_docs > 0 else 0
        print(f"  {cat:40s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nPROPERTY EXTRACTION RATES")
    sorted_props = sorted(
        property_extraction_rate.items(),
        key=lambda x: x[1]["extracted"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True
    )

    for prop_id, stats in sorted_props:
        rate = stats["extracted"] / stats["total"] * 100 if stats["total"] > 0 else 0
        bar_length = int(rate / 2)  # 50 chars max
        bar = "#" * bar_length + "." * (50 - bar_length)
        print(f"  {prop_id:30s} {bar} {rate:5.1f}% ({stats['extracted']}/{stats['total']})")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_extraction(sys.argv[1])
    else:
        analyze_extraction("outputs/rules_only.jsonl")
