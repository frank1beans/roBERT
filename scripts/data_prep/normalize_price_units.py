"""Normalize price_unit values in price dataset to standard format."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Mapping from various formats to standard price units
PRICE_UNIT_NORMALIZATION = {
    # A corpo / forfait
    "a corpo": "a_corpo",
    "corpo": "a_corpo",
    "forfait": "a_corpo",

    # Cadauno / pezzo
    "cadauno": "cad",
    "cad": "cad",
    "pezzo": "cad",
    "pz": "cad",
    "n": "cad",
    "nr": "cad",

    # Metri lineari
    "m": "m",
    "metro": "m",
    "ml": "m",
    "metro lineare": "m",

    # Metri quadrati
    "m²": "m2",
    "m2": "m2",
    "mq": "m2",
    "metro quadrato": "m2",
    "metri quadrati": "m2",

    # Metri cubi
    "m³": "m3",
    "m3": "m3",
    "mc": "m3",
    "metro cubo": "m3",
    "metri cubi": "m3",

    # Peso
    "kg": "kg",
    "chilogrammo": "kg",
    "t": "t",
    "tonnellata": "t",
    "q": "q",
    "quintale": "q",
    "g": "g",
    "grammo": "g",

    # Volume liquidi
    "l": "l",
    "litro": "l",
    "litri": "l",

    # Tempo
    "h": "h",
    "ora": "h",
    "ore": "h",
    "giorno": "giorno",
    "giorni": "giorno",
    "mese": "mese",
    "anno": "anno",

    # Set/Kit
    "set": "set",
    "kit": "set",

    # Unknown/Missing
    "#n/d": "cad",  # Default to cadauno
    "": "cad",
    "null": "cad",
}


def normalize_price_unit(raw_unit: str) -> str:
    """Normalize a price unit to standard format.

    Args:
        raw_unit: Raw price unit string from dataset

    Returns:
        Normalized price unit
    """
    if not raw_unit or raw_unit.strip() == "":
        return "cad"

    # Clean and lowercase
    cleaned = raw_unit.lower().strip()

    # Direct mapping
    if cleaned in PRICE_UNIT_NORMALIZATION:
        return PRICE_UNIT_NORMALIZATION[cleaned]

    # Try partial matches
    for key, value in PRICE_UNIT_NORMALIZATION.items():
        if key in cleaned or cleaned in key:
            return value

    # Default: return as-is but cleaned
    print(f"Warning: Unknown price unit '{raw_unit}' -> defaulting to 'cad'", file=sys.stderr)
    return "cad"


def normalize_csv_price_units(
    input_path: Path,
    output_path: Path,
    price_unit_column: str = "price_unit",
) -> Dict[str, int]:
    """Normalize price_unit values in a CSV file.

    Args:
        input_path: Input CSV file
        output_path: Output CSV file with normalized units
        price_unit_column: Name of the price_unit column

    Returns:
        Dict with statistics about normalization
    """
    stats = {
        "total_rows": 0,
        "normalized": {},
        "unknown": set(),
    }

    with input_path.open("r", encoding="utf-8-sig") as f_in, \
         output_path.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in, delimiter=";")
        fieldnames = reader.fieldnames

        if price_unit_column not in fieldnames:
            raise ValueError(f"Column '{price_unit_column}' not found in CSV")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for row in reader:
            stats["total_rows"] += 1

            raw_unit = row[price_unit_column]
            normalized_unit = normalize_price_unit(raw_unit)

            # Track statistics
            if raw_unit not in stats["normalized"]:
                stats["normalized"][raw_unit] = {
                    "normalized_to": normalized_unit,
                    "count": 0,
                }
            stats["normalized"][raw_unit]["count"] += 1

            # Update row
            row[price_unit_column] = normalized_unit
            writer.writerow(row)

    return stats


def convert_csv_to_jsonl(
    csv_path: Path,
    jsonl_path: Path,
    text_column: str = "text",
    price_column: str = "price",
    price_unit_column: str = "price_unit",
) -> int:
    """Convert CSV to JSONL format for training.

    Args:
        csv_path: Input CSV file
        jsonl_path: Output JSONL file
        text_column: Column name for text
        price_column: Column name for price
        price_unit_column: Column name for price_unit

    Returns:
        Number of records converted
    """
    count = 0

    with csv_path.open("r", encoding="utf-8-sig") as f_in, \
         jsonl_path.open("w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in, delimiter=";")

        for row in reader:
            # Skip rows without required fields
            if not row.get(text_column) or not row.get(price_column):
                continue

            # Parse price (handle comma as decimal separator)
            try:
                price_str = row[price_column].replace(",", ".")
                price = float(price_str)
            except (ValueError, AttributeError):
                print(f"Warning: Invalid price '{row.get(price_column)}', skipping row", file=sys.stderr)
                continue

            # Create record
            record = {
                "text": row[text_column].strip(),
                "price": price,
                "price_unit": row.get(price_unit_column, "cad"),
            }

            # Add optional fields if available (use original field names)
            if "super" in row and row["super"] and row["super"] != "#N/D":
                record["super"] = row["super"]
            if "cat" in row and row["cat"] and row["cat"] != "#N/D":
                record["cat"] = row["cat"]

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize price_unit values in price dataset"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file (default: input_normalized.csv)",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Also convert to JSONL format",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print normalization statistics",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_normalized.csv"

    # Normalize CSV
    print(f"Normalizing price units in {args.input}...")
    stats = normalize_csv_price_units(args.input, args.output)

    print(f"OK Processed {stats['total_rows']} rows")
    print(f"OK Output written to {args.output}")

    # Print statistics
    if args.stats:
        print("\nNormalization Statistics:")
        print("-" * 60)
        for raw_unit, info in sorted(
            stats["normalized"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        ):
            normalized = info["normalized_to"]
            count = info["count"]
            print(f"  {raw_unit:20s} -> {normalized:10s} ({count:5d} rows)")

    # Convert to JSONL
    if args.jsonl:
        print(f"\nConverting to JSONL format...")
        count = convert_csv_to_jsonl(args.output, args.jsonl)
        print(f"OK Converted {count} records to {args.jsonl}")


if __name__ == "__main__":
    main()
