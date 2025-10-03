#!/usr/bin/env python
"""Inspect cartongesso feature extraction and catalog matches."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

from robimb.extraction.cartongesso import (
    extract_cartongesso_features,
    summarize_cartongesso_features,
)


def load_jsonl(path: Path, category: Optional[str], limit: Optional[int]) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        count = 0
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if category and record.get("cat") != category:
                continue
            yield record
            count += 1
            if limit and count >= limit:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cartongesso descriptions")
    parser.add_argument("--jsonl", type=Path, required=True, help="JSONL file with descriptions")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records")
    args = parser.parse_args()

    for record in load_jsonl(args.jsonl, args.category, args.limit):
        features = extract_cartongesso_features(record.get("text", ""))
        if not features:
            print(json.dumps({"title": record.get("text", "")[:80], "note": "No layers extracted"}, ensure_ascii=False, indent=2))
            print("-" * 80)
            continue
        payload = asdict(features)
        payload["summary"] = summarize_cartongesso_features(features)
        payload["title"] = record.get("text", "").splitlines()[0][:120]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("-" * 80)


if __name__ == "__main__":
    main()