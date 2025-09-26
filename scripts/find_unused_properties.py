#!/usr/bin/env python3
"""Utility to cross property extractors with processed classification data."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set


def load_property_patterns(path: Path) -> Dict[str, dict]:
    with path.open() as f:
        data = json.load(f)
    patterns = data.get("patterns", [])
    property_patterns: Dict[str, dict] = {}
    for entry in patterns:
        property_id = entry.get("property_id")
        if not property_id:
            continue
        property_patterns[property_id] = entry
    return property_patterns


def iter_processed_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Failed to parse JSON from {path}: {exc}\nLine: {line[:200]}")


def collect_schema_slots(records: Iterable[dict]) -> Set[str]:
    property_ids: Set[str] = set()
    for record in records:
        slots = record.get("property_schema", {}).get("slots", {})
        property_ids.update(slots.keys())
    return property_ids


def collect_observed_properties(records: Iterable[dict]) -> Counter:
    counter: Counter = Counter()
    for record in records:
        props = record.get("properties", {})
        counter.update(k for k, v in props.items() if v not in (None, ""))
    return counter


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extractors",
        type=Path,
        default=Path("data/properties/extractors.json"),
        help="Path to the extractors JSON file.",
    )
    parser.add_argument(
        "--processed-glob",
        default="data/train/classification/processed/*_processed.jsonl",
        help="Glob used to discover processed dataset files.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path where to dump the unused property ids as JSON.",
    )
    args = parser.parse_args(argv)

    property_patterns = load_property_patterns(args.extractors)

    processed_paths = sorted(Path().glob(args.processed_glob))
    if not processed_paths:
        raise SystemExit(f"No processed dataset files matched {args.processed_glob!r}")

    records = list(iter_processed_records(processed_paths))
    schema_property_ids = collect_schema_slots(records)
    observed_counts = collect_observed_properties(records)

    unused_property_ids = sorted(set(property_patterns) - schema_property_ids)

    by_super: Dict[str, List[str]] = defaultdict(list)
    for property_id in unused_property_ids:
        super_id = property_id.split(".", 1)[0]
        by_super[super_id].append(property_id)

    print(f"Total property patterns: {len(property_patterns)}")
    print(f"Property ids referenced by schema: {len(schema_property_ids)}")
    print(f"Unused property ids: {len(unused_property_ids)}\n")

    print("Top unused properties by super (count >= 1):")
    for super_id, property_ids in sorted(by_super.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"- {super_id}: {len(property_ids)}")

    print("\nSample unused property ids:")
    for property_id in unused_property_ids[:20]:
        print(f"  - {property_id}")

    if args.json:
        payload = {
            "unused_property_ids": unused_property_ids,
            "by_super": {k: v for k, v in sorted(by_super.items())},
            "property_patterns_total": len(property_patterns),
            "schema_property_ids_total": len(schema_property_ids),
            "observed_counts": dict(observed_counts),
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nWrote JSON report to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
