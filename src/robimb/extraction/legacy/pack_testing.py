"""Utilities to validate extractor regexes against real datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from robimb.registry import RegistryLoader
from robimb.registry.schemas import build_category_key

from .engine import dry_run

__all__ = ["PackTestArtifacts", "run_pack_dataset_evaluation"]


@dataclass(frozen=True)
class PackTestArtifacts:
    """Paths generated while evaluating regex extractors."""

    summary: Path
    matched_examples: Path
    unmatched_examples: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "summary": str(self.summary),
            "matched_examples": str(self.matched_examples),
            "unmatched_examples": str(self.unmatched_examples),
        }


def _normalise_dataset_record(record: Mapping[str, Any], text_field: str) -> Optional[Dict[str, Any]]:
    """Return a normalised record or ``None`` when mandatory fields are missing."""

    text = record.get(text_field)
    super_label = record.get("super")
    cat_label = record.get("cat")
    if not isinstance(text, str) or not isinstance(super_label, str) or not isinstance(cat_label, str):
        return None
    return {
        "text": text,
        "super": super_label,
        "cat": cat_label,
        "uid": record.get("uid"),
    }


def _prepare_property_map(loader: RegistryLoader) -> Dict[str, List[str]]:
    """Build a dictionary keying ``super|cat`` to allowed property identifiers."""

    categories = loader.load_registry()
    property_map: Dict[str, List[str]] = {}
    for key, definition in categories.items():
        property_ids = [prop_id for prop_id in definition.property_ids() if prop_id]
        if property_ids:
            property_map[key] = property_ids
    return property_map


def _record_preview(text: str, limit: int = 240) -> str:
    """Return a short preview of ``text`` suitable for JSON reports."""

    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def run_pack_dataset_evaluation(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    pack_path: Optional[Path | str] = None,
    limit: Optional[int] = None,
    text_field: str = "text",
    sample_size: int = 20,
) -> Dict[str, Any]:
    """Execute regex extraction on a dataset returning aggregated metrics."""

    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    loader = RegistryLoader(Path(pack_path) if pack_path is not None else None)
    bundle = loader.bundle()
    extractors_pack = bundle.extractors or {}
    patterns: Iterable[Mapping[str, Any]] = extractors_pack.get("patterns", []) if isinstance(extractors_pack, Mapping) else []
    if not patterns:
        raise ValueError("Il pack selezionato non contiene pattern di estrazione.")

    property_map = _prepare_property_map(loader)
    if not property_map:
        raise ValueError("Nessuna categoria valida trovata nel registry del pack.")

    summary_file = output / "summary.json"
    matched_file = output / "matched_examples.jsonl"
    unmatched_file = output / "unmatched_examples.jsonl"

    matched_examples: List[Dict[str, Any]] = []
    unmatched_examples: List[Dict[str, Any]] = []

    property_hits: Counter[str] = Counter()
    regex_match_counter: Counter[str] = Counter()
    category_stats: MutableMapping[str, MutableMapping[str, int]] = defaultdict(lambda: defaultdict(int))
    skip_reasons: Counter[str] = Counter()

    total_records = 0
    processed_records = 0
    matched_records = 0
    unmatched_records = 0

    def _append_example(bucket: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
        if len(bucket) < sample_size:
            bucket.append(payload)

    with dataset.open("r", encoding="utf-8") as handle, matched_file.open("w", encoding="utf-8") as matched_out, unmatched_file.open(
        "w", encoding="utf-8"
    ) as unmatched_out:
        for idx, line in enumerate(handle):
            if limit is not None and total_records >= limit:
                break
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skip_reasons["invalid_json"] += 1
                continue

            normalised = _normalise_dataset_record(record, text_field)
            if normalised is None:
                skip_reasons["missing_fields"] += 1
                continue

            key = build_category_key(normalised["super"], normalised["cat"])
            allowed_properties = property_map.get(key)
            if not allowed_properties:
                skip_reasons["outside_pack"] += 1
                continue

            processed_records += 1
            category_stats[key]["processed"] += 1

            result = dry_run(normalised["text"], extractors_pack, allowed_properties=allowed_properties)
            matches = result.get("matches", [])
            extracted = result.get("extracted", {})

            for match in matches:
                property_id = match.get("property_id")
                if isinstance(property_id, str) and property_id:
                    regex_match_counter[property_id] += 1

            if extracted:
                matched_records += 1
                category_stats[key]["matched"] += 1
                for prop_id in extracted.keys():
                    property_hits[prop_id] += 1
                example_payload = {
                    "index": idx,
                    "super": normalised["super"],
                    "cat": normalised["cat"],
                    "uid": normalised.get("uid"),
                    "properties": extracted,
                    "match_count": len(matches),
                    "text_preview": _record_preview(normalised["text"]),
                }
                _append_example(matched_examples, example_payload)
                matched_out.write(json.dumps(example_payload, ensure_ascii=False) + "\n")
            else:
                unmatched_records += 1
                category_stats[key]["unmatched"] += 1
                example_payload = {
                    "index": idx,
                    "super": normalised["super"],
                    "cat": normalised["cat"],
                    "uid": normalised.get("uid"),
                    "match_count": len(matches),
                    "text_preview": _record_preview(normalised["text"]),
                }
                _append_example(unmatched_examples, example_payload)
                unmatched_out.write(json.dumps(example_payload, ensure_ascii=False) + "\n")

    artifacts = PackTestArtifacts(summary=summary_file, matched_examples=matched_file, unmatched_examples=unmatched_file)

    summary_payload: Dict[str, Any] = {
        "pack_version": bundle.version,
        "generated_at": bundle.generated_at,
        "dataset": str(dataset),
        "total_records": total_records,
        "processed_records": processed_records,
        "matched_records": matched_records,
        "unmatched_records": unmatched_records,
        "skipped_records": total_records - processed_records,
        "skip_reasons": dict(skip_reasons),
        "regex_success": matched_records > 0,
        "property_hits": [
            {"property_id": prop_id, "records": count}
            for prop_id, count in sorted(property_hits.items(), key=lambda item: (-item[1], item[0]))
        ],
        "regex_matches": [
            {"property_id": prop_id, "matches": count}
            for prop_id, count in sorted(regex_match_counter.items(), key=lambda item: (-item[1], item[0]))
        ],
        "category_stats": {
            key: {
                "processed": stats.get("processed", 0),
                "matched": stats.get("matched", 0),
                "unmatched": stats.get("unmatched", 0),
            }
            for key, stats in sorted(category_stats.items())
        },
        "artifacts": artifacts.as_dict(),
        "matched_examples_sample": matched_examples,
        "unmatched_examples_sample": unmatched_examples,
    }

    summary_file.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary_payload

