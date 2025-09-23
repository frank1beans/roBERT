"""Normalizers registry used by the extraction engine."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Protocol, Sequence, Set

from .dsl import ExtractorsPack


class Normalizer(Protocol):
    """Callable protocol for normalizer functions."""

    def __call__(self, value: Any, matched_text: str) -> Any:  # pragma: no cover - protocol definition
        ...


NormalizerFactory = Callable[[Any, str], Any]
"""Backward compatible alias for callables implementing :class:`Normalizer`."""


def _comma_to_dot(v: Any, m: str) -> Any:
    if isinstance(v, str):
        return v.replace(",", ".")
    if isinstance(v, list):
        return [str(x).replace(",", ".") for x in v]
    return v


def _dot_to_comma(v: Any, m: str) -> Any:
    if isinstance(v, str):
        return v.replace(".", ",")
    if isinstance(v, list):
        return [str(x).replace(".", ",") for x in v]
    return v


def _to_float(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return x

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _to_int(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        try:
            return int(re.sub(r"[^0-9-]", "", str(x)))
        except Exception:
            return x

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _lower(v: Any, m: str) -> Any:
    return v.lower() if isinstance(v, str) else v


def _upper(v: Any, m: str) -> Any:
    return v.upper() if isinstance(v, str) else v


def _strip(v: Any, m: str) -> Any:
    return v.strip() if isinstance(v, str) else v


_UNIT_VARIANTS: Dict[str, Sequence[str]] = {
    "m²": ["mq", "m2", "m²", "m^2", "metri quadri", "metri quadrati"],
    "m³": ["m3", "m³", "metri cubi", "metri cubici"],
    "m": ["m", "mt", "metri", "metro"],
    "cm": ["cm", "centimetri", "centimetro"],
    "mm": ["mm", "millimetri", "millimetro"],
    "kg": ["kg", "chilogrammo", "chilogrammi"],
    "kW": ["kw", "kilowatt", "kilowatts"],
    "kVA": ["kva", "kilovolt ampere", "kilovolt-ampere"],
}


def _normalize_unit_symbols(v: Any, m: str) -> Any:
    """Coerce textual unit variants to a canonical SI representation."""

    unit_lookup: Dict[str, str] = {}
    for canonical, variants in _UNIT_VARIANTS.items():
        for variant in variants:
            unit_lookup[variant.lower()] = canonical

    if not unit_lookup:
        return v

    pattern = re.compile(
        r"(?i)\b(" + "|".join(sorted((re.escape(k) for k in unit_lookup), key=len, reverse=True)) + r")\b"
    )

    def normalize_text(text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            token = match.group(0).lower()
            return unit_lookup.get(token, match.group(0))

        return pattern.sub(repl, text)

    if isinstance(v, list):
        return [normalize_text(str(x)) for x in v]
    if isinstance(v, str):
        return normalize_text(v)
    return v


def _ei_from_any(v: Any, m: str) -> Any:
    s = " ".join(v) if isinstance(v, (list, tuple)) else str(v)
    mnum = re.search(r"(15|30|45|60|90|120|180|240)", s)
    return f"EI {mnum.group(1)}" if mnum else s


def _dims_join(v: Any, m: str) -> Any:
    # accept tuple/list like ('60','60') -> '60×60'
    if isinstance(v, (list, tuple)):
        if len(v) == 2:
            return f"{v[0]}×{v[1]}"
        if len(v) > 2 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in v):
            return "; ".join([f"{a}×{b}" for a, b in v])
    return v


def _collect_many(v: Any, m: str) -> Any:
    # if list, keep list; else wrap into list
    if isinstance(v, list):
        return v
    return [v]


def _if_cm_to_mm(v: Any, m: str) -> Any:
    # multiply by 10 if match string contains 'cm' (not mm)
    def conv_one(x: Any) -> Any:
        try:
            val = float(str(x).replace(",", "."))
        except Exception:
            return x
        s = m.lower()
        if "cm" in s and "mm" not in s:
            return round(val * 10.0, 3)
        return val

    if isinstance(v, list):
        return [conv_one(x) for x in v]
    return conv_one(v)


def _unique_list(v: Any, m: str) -> Any:
    if not isinstance(v, list):
        return v
    seen: Set[Any] = set()
    out: List[Any] = []
    for x in v:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _as_string(v: Any, m: str) -> Any:
    if isinstance(v, list):
        return [str(x) for x in v]
    return str(v)


def _concat_dims(v: Any, m: str) -> Any:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]}×{v[1]}"
    return v


def _split_structured_list(v: Any, m: str) -> Any:
    """Split structured textual lists into individual elements."""

    def split_one(text: str) -> List[str]:
        parts = re.split(r"[;,/]|(?:\s+e\s+)|(?:\s+and\s+)", text)
        cleaned = [p.strip() for p in parts if p and p.strip()]
        return cleaned if cleaned else [text.strip()]

    if isinstance(v, list):
        return [item for x in v for item in split_one(str(x))]
    if isinstance(v, str):
        return split_one(v)
    return v


_EI_PATTERN = re.compile(r"(15|30|45|60|90|120|180|240)")


def _format_ei_from_last_int(v: Any, m: str) -> Any:
    if isinstance(v, (list, tuple)):
        hay = " ".join(str(x) for x in v)
    else:
        hay = str(v)
    nums = _EI_PATTERN.findall(hay)
    if not nums:
        nums = _EI_PATTERN.findall(m)
    if not nums:
        return v
    return f"EI {nums[-1]}"


def _take_last_int_to_ei(v: Any, m: str) -> Any:
    return _format_ei_from_last_int(v, m)


def _to_ei_class(v: Any, m: str) -> Any:
    formatted = _format_ei_from_last_int(v, m)
    if isinstance(formatted, str):
        return formatted.replace(" ", "").upper()
    return formatted


_FORATURA_MAP = {
    "pieno": "pieno",
    "forato": "forato",
    "semi pieno": "semipieno",
    "semi-pieno": "semipieno",
    "semipieno": "semipieno",
}


def _normalize_foratura(v: Any, m: str) -> Any:
    def norm_one(x: Any) -> Any:
        key = str(x).strip().lower().replace("  ", " ")
        return _FORATURA_MAP.get(key, x)

    if isinstance(v, list):
        return [norm_one(x) for x in v]
    return norm_one(v)


def _cm_to_mm_optional(v: Any, m: str) -> Any:
    lower = m.lower()
    convert = "cm" in lower and "mm" not in lower

    def conv_one(x: Any) -> Any:
        if not convert:
            return x
        try:
            return float(x) * 10.0
        except Exception:
            try:
                return float(str(x).replace(",", ".")) * 10.0
            except Exception:
                return x

    if isinstance(v, list):
        return [conv_one(x) for x in v]
    return conv_one(v)


def _cm_to_mm_if_cm(v: Any, m: str) -> Any:
    def _to_number(x: Any) -> Any:
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return x

    if isinstance(v, (list, tuple)) and v:
        value = v[0]
        unit = v[1] if len(v) > 1 else ""
        num = _to_number(value)
        if isinstance(num, (int, float)):
            unit_norm = str(unit).strip().lower()
            if unit_norm == "cm":
                return num * 10.0
            if unit_norm == "mm":
                return num
        return v

    num = _to_number(v)
    return num


_STRATI_WORDS = {
    "doppia": 2,
    "tripla": 3,
}


def _to_strati_count(v: Any, m: str) -> Any:
    if isinstance(v, (list, tuple)) and v:
        base = str(v[-1])
    else:
        base = str(v)
    key = base.strip().lower()
    if key in _STRATI_WORDS:
        return _STRATI_WORDS[key]
    try:
        return int(re.sub(r"[^0-9]", "", key))
    except Exception:
        digits = re.findall(r"\d+", m)
        if digits:
            try:
                return int(digits[-1])
            except Exception:
                pass
    return v


_TIPO_LASTRA_MAP = {
    "gkb": "standard",
    "standard": "standard",
    "gkfi": "fuoco",
    "gkf": "fuoco",
    "gklo": "fuoco",
    "fuoco": "fuoco",
    "hf": "fuoco",
    "h2": "idr",
    "idro": "idr",
    "idr": "idr",
    "idrorepellente": "idr",
    "acustico": "acustica",
    "acustica": "acustica",
    "antimuffa": "antimuffa",
    "fibrocemento": "fibrocemento",
}


def _map_tipo_lastra_enum(v: Any, m: str) -> Any:
    def infer_from_text(text: str) -> str | None:
        lower = text.strip().lower()
        for key, value in _TIPO_LASTRA_MAP.items():
            if key in lower:
                return value
        return None

    if isinstance(v, list):
        return [_map_tipo_lastra_enum(x, m) for x in v]

    match_hint = infer_from_text(m)
    value_hint = infer_from_text(str(v))

    if value_hint is not None:
        return value_hint
    if match_hint is not None:
        return match_hint
    return v


_YES_NO_MAP = {
    "si": True,
    "sì": True,
    "yes": True,
    "oui": True,
    "ja": True,
    "vero": True,
    "true": True,
    "no": False,
    "not": False,
    "non": False,
    "nope": False,
    "false": False,
}


def _map_yes_no_multilang(v: Any, m: str) -> Any:
    """Normalize multilingual affirmative/negative markers into booleans."""

    def normalize_token(token: str) -> Any:
        lowered = token.strip().lower().replace("ì", "i")
        if lowered in _YES_NO_MAP:
            return _YES_NO_MAP[lowered]
        return token

    if isinstance(v, list):
        return [normalize_token(str(x)) for x in v]
    if isinstance(v, str):
        return normalize_token(v)
    return v


BUILTIN_NORMALIZERS: Dict[str, Normalizer] = {
    "comma_to_dot": _comma_to_dot,
    "dot_to_comma": _dot_to_comma,
    "to_float": _to_float,
    "to_int": _to_int,
    "lower": _lower,
    "upper": _upper,
    "strip": _strip,
    "EI_from_any": _ei_from_any,
    "dims_join": _dims_join,
    "collect_many": _collect_many,
    "if_cm_to_mm": _if_cm_to_mm,
    "unique_list": _unique_list,
    "as_string": _as_string,
    "concat_dims": _concat_dims,
    "normalize_unit_symbols": _normalize_unit_symbols,
    "split_structured_list": _split_structured_list,
    "format_EI_from_last_int": _format_ei_from_last_int,
    "take_last_int->EI {n}": _take_last_int_to_ei,
    "normalize_foratura": _normalize_foratura,
    "cm_to_mm?": _cm_to_mm_optional,
    "to_ei_class": _to_ei_class,
    "cm_to_mm_if_cm": _cm_to_mm_if_cm,
    "to_strati_count": _to_strati_count,
    "map_tipo_lastra_enum": _map_tipo_lastra_enum,
    "map_yes_no_multilang": _map_yes_no_multilang,
    # dynamic "map_enum:<name>" supported by :func:`build_normalizer`.
}


def _map_enum_factory(mapping: Dict[str, str]) -> Normalizer:
    def _map_enum(v: Any, m: str) -> Any:
        def map_one(x: Any) -> Any:
            key = str(x).strip().lower()
            return mapping.get(key, x)

        if isinstance(v, list):
            return [map_one(x) for x in v]
        return map_one(v)

    return _map_enum


def build_normalizer(name: str, extractors_pack: ExtractorsPack) -> Normalizer:
    """Resolve ``name`` to a callable normalizer."""

    if name.startswith("map_enum:"):
        key = name.split(":", 1)[1]
        mapping = extractors_pack.get("normalizers", {}).get(key, {})
        return _map_enum_factory(mapping)

    fn = BUILTIN_NORMALIZERS.get(name)
    if fn is None:
        return lambda v, m: v
    return fn


__all__ = [
    "Normalizer",
    "NormalizerFactory",
    "BUILTIN_NORMALIZERS",
    "build_normalizer",
]
