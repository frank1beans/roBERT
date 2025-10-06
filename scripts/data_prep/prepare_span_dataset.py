"""Prepare span training dataset from price data.

This script:
1. Loads price training data
2. Filters records by categories that have property schemas in registry
3. Prepares data for span extraction training
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


def load_registry_categories(registry_path: Path) -> Dict[str, List[str]]:
    """Load categories with their properties from registry.

    Returns:
        Dict mapping "super::cat" to list of property IDs
    """
    with open(registry_path) as f:
        registry = json.load(f)

    categories_with_props = {}

    for category in registry.get("categories", []):
        cat_id = category.get("id")
        cat_name = category.get("name")
        properties = category.get("properties", [])

        if properties:
            # Extract property IDs
            prop_ids = [p["id"] for p in properties]
            categories_with_props[cat_name] = prop_ids

    return categories_with_props


def filter_price_data(
    price_jsonl: Path,
    categories_with_props: Dict[str, List[str]],
    output_jsonl: Path,
) -> None:
    """Filter price data to only categories with property schemas."""

    stats = defaultdict(int)
    filtered_records = []

    with open(price_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            stats["total"] += 1

            # Check if category has properties
            cat = record.get("cat")
            super_cat = record.get("super")

            # Try exact match first
            if cat in categories_with_props:
                record["property_ids"] = categories_with_props[cat]
                filtered_records.append(record)
                stats["matched"] += 1
                stats[f"cat:{cat}"] += 1
            elif super_cat in categories_with_props:
                record["property_ids"] = categories_with_props[super_cat]
                filtered_records.append(record)
                stats["matched"] += 1
                stats[f"super:{super_cat}"] += 1

    # Write filtered data
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in filtered_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return stats


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent.parent

    # Paths
    registry_path = base_dir / "resources/pack/current/registry.json"
    price_train = base_dir / "resources/data/train/price/train.jsonl"
    output_dir = base_dir / "resources/data/train/span"
    output_train = output_dir / "train_filtered.jsonl"

    print("=" * 60)
    print("PREPARING SPAN TRAINING DATASET")
    print("=" * 60)

    # Load registry
    print(f"\n1. Loading registry from: {registry_path}")
    categories_with_props = load_registry_categories(registry_path)
    print(f"   Found {len(categories_with_props)} categories with properties")

    for cat, props in list(categories_with_props.items())[:5]:
        print(f"     - {cat}: {len(props)} properties")

    # Filter price data
    print(f"\n2. Filtering price data from: {price_train}")
    stats = filter_price_data(price_train, categories_with_props, output_train)

    print(f"\n3. Results:")
    print(f"   Total records: {stats['total']}")
    print(f"   Matched records: {stats['matched']}")
    print(f"   Match rate: {stats['matched']/stats['total']*100:.1f}%")

    print(f"\n4. Output written to: {output_train}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)

    # Print category distribution
    print("\nCategory distribution:")
    cat_stats = {k: v for k, v in stats.items() if k.startswith("cat:") or k.startswith("super:")}
    for cat, count in sorted(cat_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
