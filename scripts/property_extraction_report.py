"""Generate a confidence-aware report for property extraction outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence

from robimb.extraction.resources import load_default
from robimb.features.extractors import extract_properties_with_confidences


@dataclass
class RecordReport:
    """Structured representation of extracted properties for a record."""

    index: int
    super_name: str
    category_name: str
    text_excerpt: str
    properties: Sequence[Dict[str, object]]


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prepare_report(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    extractors_pack = load_default()

    record_reports: List[RecordReport] = []
    properties_per_record: List[int] = []
    confidence_values: List[float] = []

    expected_counts: Counter[str] = Counter()
    extracted_counts: Counter[str] = Counter()
    confidence_by_property: Dict[str, List[float]] = defaultdict(list)

    for idx, record in enumerate(records):
        text = str(record.get("text", ""))
        property_schema = record.get("property_schema") or {}
        slots = property_schema.get("slots") if isinstance(property_schema, dict) else None
        allowed_properties: Iterable[str] | None = None
        slot_ids: Sequence[str] = ()
        if isinstance(slots, dict):
            slot_ids = tuple(str(key) for key in slots.keys())
            expected_counts.update(slot_ids)
            allowed_properties = slot_ids

        extraction = extract_properties_with_confidences(
            text,
            extractors_pack,
            allowed_properties=allowed_properties,
        )

        items: List[Dict[str, object]] = []
        for prop_id, value in extraction.values.items():
            confidence = extraction.confidences.get(prop_id)
            if confidence is not None:
                confidence_values.append(float(confidence))
                confidence_by_property[prop_id].append(float(confidence))
            extracted_counts[prop_id] += 1
            items.append({"id": prop_id, "value": value, "confidence": confidence})

        properties_per_record.append(len(items))

        record_reports.append(
            RecordReport(
                index=idx,
                super_name=str(record.get("super", "")),
                category_name=str(record.get("cat", "")),
                text_excerpt=text[:160],
                properties=tuple(items),
            )
        )

    extracted_records = [report for report in record_reports if report.properties]
    flat_properties = [item for report in record_reports for item in report.properties]

    summary: Dict[str, object] = {
        "num_records": len(record_reports),
        "records_with_properties": len(extracted_records),
        "total_properties": len(flat_properties),
        "properties_per_record": {
            "avg": mean(properties_per_record) if properties_per_record else 0.0,
            "median": median(properties_per_record) if properties_per_record else 0.0,
            "max": max(properties_per_record) if properties_per_record else 0,
        },
    }

    if confidence_values:
        summary["confidence_stats"] = {
            "avg": mean(confidence_values),
            "median": median(confidence_values),
            "min": min(confidence_values),
            "max": max(confidence_values),
        }
    else:
        summary["confidence_stats"] = {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    property_stats: List[Dict[str, object]] = []
    for prop_id, expected in sorted(expected_counts.items()):
        extracted = extracted_counts.get(prop_id, 0)
        confidences = confidence_by_property.get(prop_id, [])
        avg_conf = mean(confidences) if confidences else None
        property_stats.append(
            {
                "property_id": prop_id,
                "expected": expected,
                "extracted": extracted,
                "coverage_ratio": float(extracted) / expected if expected else 0.0,
                "avg_confidence": avg_conf,
            }
        )

    property_stats.sort(key=lambda item: (item["coverage_ratio"], item["expected"]))

    summary["property_stats"] = property_stats
    summary["low_coverage_properties"] = property_stats[:25]

    lowest_confidence = [
        item
        for item in property_stats
        if item["avg_confidence"] is not None and item["extracted"] >= 2
    ]
    lowest_confidence.sort(key=lambda item: item["avg_confidence"])
    summary["lowest_confidence_properties"] = lowest_confidence[:25]

    summary["records"] = [
        {
            "index": report.index,
            "super": report.super_name,
            "category": report.category_name,
            "text_excerpt": report.text_excerpt,
            "num_properties": len(report.properties),
            "properties": list(report.properties),
        }
        for report in record_reports
    ]

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL dataset enriched with property schema")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path where the JSON report will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_jsonl(args.dataset)
    report = _prepare_report(records)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
