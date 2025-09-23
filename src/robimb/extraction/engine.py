"""Regex based property extraction engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .dsl import ExtractorsPack
from .normalizers import BUILTIN_NORMALIZERS, Normalizer, build_normalizer


@dataclass
class Pattern:
    """Compiled representation of a pattern entry in the pack."""

    property_id: str
    regex: Sequence[str]
    normalizers: Sequence[str]


def _compile_patterns(
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
) -> List[Pattern]:
    pats: List[Pattern] = []
    allowed: Optional[set[str]] = None
    if allowed_properties is not None:
        allowed = {p for p in allowed_properties if p}

    for item in extractors_pack.get("patterns", []):
        pid = item["property_id"]
        if allowed is not None and pid not in allowed:
            continue
        pats.append(
            Pattern(
                property_id=pid,
                regex=list(item.get("regex", [])),
                normalizers=list(item.get("normalizers", [])),
            )
        )
    return pats


def _apply_normalizers(
    value: Any,
    matched_text: str,
    normalizers: Sequence[str],
    extractors_pack: ExtractorsPack,
) -> Any:
    v = value
    for name in normalizers:
        fn: Normalizer = build_normalizer(name, extractors_pack)
        v = fn(v, matched_text)
    return v


def _coerce_capture(match: re.Match[str]) -> Any:
    # No groups: return entire matched text
    if match.lastindex is None:
        return match.group(0)
    # 1 group: return that group
    if match.lastindex == 1:
        return match.group(1)
    # 2+ groups: return tuple of groups
    return tuple(match.group(i) for i in range(1, match.lastindex + 1))


def extract_properties(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Apply patterns declared in ``extractors_pack`` to ``text``."""

    out: Dict[str, Any] = {}
    pats = _compile_patterns(extractors_pack, allowed_properties=allowed_properties)
    compiled: List[Tuple[Pattern, List[re.Pattern[str]]]] = [
        (p, [re.compile(rx, flags=re.IGNORECASE) for rx in p.regex]) for p in pats
    ]
    for pat, regs in compiled:
        has_collect = "collect_many" in pat.normalizers
        for rx in regs:
            for m in rx.finditer(text):
                cap = _coerce_capture(m)
                val = _apply_normalizers(cap, m.group(0), pat.normalizers, extractors_pack)
                if has_collect:
                    prev = out.get(pat.property_id, [])
                    if not isinstance(prev, list):
                        prev = [prev]
                    prev.append(val)
                    out[pat.property_id] = prev
                else:
                    if pat.property_id not in out:
                        out[pat.property_id] = val
            if pat.property_id in out and not has_collect:
                break
        if has_collect and pat.property_id in out and "unique_list" in pat.normalizers:
            out[pat.property_id] = BUILTIN_NORMALIZERS["unique_list"](out[pat.property_id], "")
    return out


def dry_run(
    text: str,
    extractors_pack: ExtractorsPack,
    allowed_properties: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return debug information with raw matches and normalized output."""

    details: List[Dict[str, Any]] = []
    pats = _compile_patterns(extractors_pack, allowed_properties=allowed_properties)
    for pat in pats:
        for rx in pat.regex:
            comp = re.compile(rx, re.IGNORECASE)
            for m in comp.finditer(text):
                details.append(
                    {
                        "property_id": pat.property_id,
                        "regex": rx,
                        "match": m.group(0),
                        "groups": m.groups(),
                    }
                )
    extracted = extract_properties(
        text,
        extractors_pack,
        allowed_properties=allowed_properties,
    )
    return {"matches": details, "extracted": extracted}


__all__ = ["Pattern", "extract_properties", "dry_run"]
