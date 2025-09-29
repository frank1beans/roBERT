"""Normalizers registry used by the extraction engine."""

from __future__ import annotations
import json
import math
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set

from .dsl import ExtractorsPack


class Normalizer(Protocol):
    """Callable protocol for normalizer functions."""

    def __call__(self, value: Any, matched_text: str) -> Any:  # pragma: no cover - protocol definition
        ...


NormalizerFactory = Callable[[Any, str], Any]
"""Backward compatible alias for callables implementing :class:`Normalizer`."""



_MODEL_STOPWORDS: Set[str] = {
    "a",
    "ai",
    "al",
    "alla",
    "alle",
    "allo",
    "all'",
    "coi",
    "col",
    "con",
    "da",
    "dal",
    "dalla",
    "dalle",
    "dallo",
    "dei",
    "degli",
    "del",
    "della",
    "delle",
    "dello",
    "di",
    "e",
    "ed",
    "fra",
    "gli",
    "i",
    "il",
    "in",
    "la",
    "le",
    "lo",
    "nei",
    "negli",
    "nel",
    "nella",
    "nelle",
    "nello",
    "per",
    "su",
    "sui",
    "sul",
    "sulla",
    "sulle",
    "sullo",
    "tra",
    "verso",
    "ello",
    "so",
    "l",
}

_MODEL_FOLLOWUP_STOPWORDS: Set[str] = {
    "approvazione",
    "apertura",
    "assistenze",
    "bagno",
    "bagni",
    "campionare",
    "campionatura",
    "coordinare",
    "coordinamento",
    "compensati",
    "compensato",
    "completo",
    "completa",
    "completi",
    "complete",
    "comprende",
    "comprendono",
    "compreso",
    "compresa",
    "compresi",
    "compresse",
    "dimensione",
    "dimensioni",
    "dotato",
    "dotata",
    "dotati",
    "dotate",
    "dotazione",
    "finitura",
    "finiture",
    "fornita",
    "fornite",
    "forniti",
    "fornito",
    "fornitura",
    "forniture",
    "inclusa",
    "inclusi",
    "incluso",
    "installazione",
    "installazioni",
    "marcatura",
    "materiale",
    "materiali",
    "posa",
    "pose",
    "profondità",
    "scarico",
    "servizi",
    "trasporto",
}

_MODEL_INVALID_PATTERNS: Sequence[str] = (
    "da definire",
    "da campionare",
    "da sottoporre",
    "da verificare",
    "da approvare",
    "da approvazione",
    "da coordinare",
    "da determinare",
)


_MODEL_TRIM_CHARS = ".,;:()[]{}\"'“”‘’"


# ---------------------------------------------------------------------------
# Basic string utilities
# ---------------------------------------------------------------------------


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


def _strip_trailing_punct(v: Any, m: str) -> Any:
    if isinstance(v, str):
        return v.rstrip(" .,;:\u2026")
    return v





def _truncate_model_value(v: Any, m: str) -> Any:
    def _normalize(text: str, context: str) -> str | None:
        if not text:
            return None
        collapsed = re.sub(r"\s+", " ", text.strip())
        if not collapsed:
            return None
        lowered = collapsed.lower()
        normalized_context = re.sub(r"\s+", " ", context.strip().lower())
        for phrase in _MODEL_INVALID_PATTERNS:
            if phrase in lowered or phrase in normalized_context:
                return None
        tokens = collapsed.split()
        cleaned: list[str] = []
        for token in tokens:
            stripped = token.strip(_MODEL_TRIM_CHARS + "-")
            if not stripped:
                continue
            lowered_token = stripped.lower()
            if lowered_token in _MODEL_STOPWORDS:
                if not cleaned:
                    continue
                break
            if (
                lowered_token in _MODEL_FOLLOWUP_STOPWORDS
                or lowered_token.startswith("cod")
                or lowered_token.startswith("art")
            ):
                break
            cleaned.append(stripped)
            if len(cleaned) >= 6:
                break
        candidate = " ".join(cleaned).strip(_MODEL_TRIM_CHARS + "-")
        if not candidate:
            return None
        return candidate

    if isinstance(v, list):
        normalized_items: list[Any] = []
        for item in v:
            if isinstance(item, str):
                candidate = _normalize(item, m)
                if candidate:
                    normalized_items.append(candidate)
            elif item not in (None, ""):
                normalized_items.append(item)
        return normalized_items
    if isinstance(v, str):
        candidate = _normalize(v, m)
        return candidate if candidate is not None else None
    return v


def _collapse_plus_sequences(v: Any, m: str) -> Any:
    def collapse_one(token: Any) -> Any:
        if isinstance(token, str):
            collapsed = re.sub(r"\s*\+\s*", "+", token.strip())
            return collapsed
        return token

    if isinstance(v, list):
        return [collapse_one(x) for x in v]
    return collapse_one(v)


def _as_string(v: Any, m: str) -> Any:
    if isinstance(v, list):
        return [str(x) for x in v]
    return str(v)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


_NUMBER_CLEAN_RE = re.compile(r"[^0-9,.-]+")


def _parse_number(token: str) -> Optional[float]:
    """Parse ``token`` into a float supporting both comma and dot decimals."""

    if token is None:
        return None
    cleaned = token.strip().replace("\xa0", "")
    # treat stray semicolons as decimal separators (common OCR typo)
    cleaned = cleaned.replace(";", ",")
    if not cleaned:
        return None

    if cleaned.count("-") > 1:
        return None
    sign = -1.0 if cleaned.lstrip().startswith("-") else 1.0
    cleaned = _NUMBER_CLEAN_RE.sub("", cleaned)
    if not cleaned:
        return None

    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")
    decimal_sep = None
    if last_comma > last_dot:
        decimal_sep = ","
    elif last_dot > last_comma:
        decimal_sep = "."
    elif cleaned.count(",") == 1 and cleaned.count(".") == 0:
        decimal_sep = ","
    elif cleaned.count(".") == 1 and cleaned.count(",") == 0:
        decimal_sep = "."

    if decimal_sep == ",":
        normalized = cleaned.replace(".", "").replace(",", ".")
    elif decimal_sep == ".":
        normalized = cleaned.replace(",", "")
    else:
        normalized = cleaned.replace(",", "").replace(".", "")

    if normalized in {"", "-"}:
        return None

    if sign < 0 and normalized.startswith("-"):
        normalized = normalized[1:]

    try:
        value = float(normalized)
    except Exception:
        return None
    return value * sign


def _coerce_numeric_output(value: float) -> float | int:
    if math.isfinite(value) and float(value).is_integer():
        return int(value)
    return float(value)


def _to_number(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        if isinstance(x, (int, float)):
            return x
        parsed = _parse_number(str(x))
        if parsed is None:
            return x
        return _coerce_numeric_output(parsed)

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _join_range(v: Any, m: str) -> Any:
    def parse_values(parts: Sequence[Any]) -> List[Any]:
        out_parts: List[Any] = []
        for part in parts:
            parsed = _parse_number(str(part))
            if parsed is not None:
                out_parts.append(_coerce_numeric_output(parsed))
            else:
                cleaned = str(part).strip()
                if cleaned.lower() in {"mm", "cm", "m"}:
                    continue
                if cleaned:
                    out_parts.append(cleaned)
        return out_parts

    if isinstance(v, (list, tuple)):
        values = parse_values(v)
        return values

    tokens = re.split(r"\s*(?:[-–—]|(?:a[l]?|to)\s+|fino\s+a)\s*", str(v))
    if len(tokens) >= 2:
        values = parse_values(tokens[:2])
        return values
    return v


# ---------------------------------------------------------------------------
# Units helpers
# ---------------------------------------------------------------------------


_UNIT_VARIANTS: Dict[str, Sequence[str]] = {
    "m²": ["mq", "m2", "m²", "m^2", "metri quadri", "metri quadrati"],
    "m³": ["m3", "m³", "metri cubi", "metri cubici"],
    "m": ["m", "mt", "metri", "metro"],
    "cm": ["cm", "centimetri", "centimetro"],
    "mm": ["mm", "millimetri", "millimetro"],
    "kg": ["kg", "chilogrammo", "chilogrammi"],
    "t": ["t", "ton", "tonnellata", "tonnellate"],
    "g": ["g", "grammo", "grammi"],
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


def _percent_to_ratio(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        parsed = _parse_number(str(x))
        if parsed is None:
            return x
        hay = f"{x} {m}".lower()
        if "%" in hay or "percent" in hay:
            return _coerce_numeric_output(parsed / 100.0)
        return _coerce_numeric_output(parsed)

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _power_to_kw(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        parsed = _parse_number(str(x))
        if parsed is None:
            return x
        hay = f"{x} {m}".lower()
        if "kw" in hay or "kilowatt" in hay:
            return _coerce_numeric_output(parsed)
        if " w" in hay or hay.strip().endswith("w"):
            return _coerce_numeric_output(parsed / 1000.0)
        return _coerce_numeric_output(parsed)

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _flow_to_m3h(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        parsed = _parse_number(str(x))
        if parsed is None:
            return x
        hay = f"{x} {m}".lower()
        if "m3/s" in hay or "m³/s" in hay:
            return _coerce_numeric_output(parsed * 3600.0)
        if "l/s" in hay:
            return _coerce_numeric_output(parsed * 3.6)
        if "l/min" in hay:
            return _coerce_numeric_output(parsed * 0.06)
        return _coerce_numeric_output(parsed)

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _flow_to_l_s(v: Any, m: str) -> Any:
    def conv(x: Any) -> Any:
        parsed = _parse_number(str(x))
        if parsed is None:
            return x
        hay = f"{x} {m}".lower()
        if "l/min" in hay:
            return _coerce_numeric_output(parsed / 60.0)
        if "m3/h" in hay or "m³/h" in hay:
            return _coerce_numeric_output(parsed * (1000.0 / 3600.0))
        if "m3/s" in hay or "m³/s" in hay:
            return _coerce_numeric_output(parsed * 1000.0)
        return _coerce_numeric_output(parsed)

    if isinstance(v, list):
        return [conv(x) for x in v]
    return conv(v)


def _unit_from_context(matched_text: str) -> Optional[str]:
    normalized = matched_text.lower()
    normalized = normalized.replace("²", "2").replace("³", "3")
    replacements = {
        "metri quadrati": "m2",
        "metri quadri": "m2",
        "metri cubi": "m3",
        "metri cubes": "m3",
        "metri cube": "m3",
        "chilogrammi": "kg",
        "chilogrammo": "kg",
        "grammi": "g",
        "grammo": "g",
        "tonnellate": "t",
        "tonnellata": "t",
        "ton ": "t ",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)

    unit_pattern = re.compile(
        r"\b(mm3|cm3|m3|mm2|cm2|m2|mq|mc|mm|cm|m|kg|g|t)\b",
        re.IGNORECASE,
    )
    found = unit_pattern.findall(normalized)
    if not found:
        return None
    candidate = found[-1].lower()
    if candidate == "mq":
        return "m2"
    if candidate == "mc":
        return "m3"
    return candidate


def _to_unit_factory(target_unit: str) -> Normalizer:
    factors = {
        "mm": 1.0,
        "cm": 10.0,
        "m": 1000.0,
        "mm2": 1.0,
        "cm2": 100.0,
        "m2": 1_000_000.0,
        "mm3": 1.0,
        "cm3": 1000.0,
        "m3": 1_000_000_000.0,
        "g": 0.001,
        "kg": 1.0,
        "t": 1000.0,
    }
    if target_unit not in factors:
        raise ValueError(f"Unsupported unit normalizer target: {target_unit}")

    def convert(value: Any, matched_text: str) -> Any:
        def conv_one(x: Any, unit_hint: Optional[str]) -> Any:
            parsed = _parse_number(str(x))
            if parsed is None:
                return x
            unit = unit_hint or _unit_from_context(matched_text) or target_unit
            unit = unit.lower()
            if unit not in factors:
                return _coerce_numeric_output(parsed)
            base_value = parsed * factors[unit]
            converted = base_value / factors[target_unit]
            return _coerce_numeric_output(converted)

        if isinstance(value, (list, tuple)):
            if not value:
                return value
            if (
                len(value) == 2
                and isinstance(value[1], str)
                and str(value[1]).strip().lower() in factors
            ):
                return conv_one(value[0], str(value[1]).lower())
            return [conv_one(v, None) for v in value]
        return conv_one(value, None)

    return convert


# ---------------------------------------------------------------------------
# Domain specific utilities
# ---------------------------------------------------------------------------


def _ei_from_any(v: Any, m: str) -> Any:
    s = " ".join(v) if isinstance(v, (list, tuple)) else str(v)
    mnum = re.search(r"(15|30|45|60|90|120|180|240)", s)
    return f"EI {mnum.group(1)}" if mnum else s


def _dims_join(v: Any, m: str) -> Any:
    if isinstance(v, (list, tuple)):
        if len(v) == 2:
            return f"{v[0]}×{v[1]}"
        if len(v) > 2 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in v):
            return "; ".join([f"{a}×{b}" for a, b in v])
    return v


def _concat_dims(v: Any, m: str) -> Any:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return f"{v[0]}×{v[1]}"
    return v


def _collect_many(v: Any, m: str) -> Any:
    if isinstance(v, list):
        return v
    return [v]


def _if_cm_to_mm(v: Any, m: str) -> Any:
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
    def _to_number_inner(x: Any) -> Any:
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return x

    if isinstance(v, (list, tuple)) and v:
        value = v[0]
        unit = v[1] if len(v) > 1 else ""
        num = _to_number_inner(value)
        if isinstance(num, (int, float)):
            unit_norm = str(unit).strip().lower()
            if unit_norm == "cm":
                return num * 10.0
            if unit_norm == "mm":
                return num
        return v

    num = _to_number_inner(v)
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
    "antincendio": "fuoco",
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


def _to_bool_strict(v: Any, m: str) -> Any:
    mapping = {
        "si": True,
        "sì": True,
        "yes": True,
        "oui": True,
        "ja": True,
        "vero": True,
        "true": True,
        "no": False,
        "nein": False,
        "non": False,
        "false": False,
        "falso": False,
    }

    def normalize_token(token: str) -> Any:
        lowered = token.strip().lower().replace("ì", "i")
        if lowered in mapping:
            return mapping[lowered]
        return token

    if isinstance(v, list):
        return [normalize_token(str(x)) for x in v]
    if isinstance(v, str):
        return normalize_token(v)
    return v


def _class_norm_en12207(v: Any, m: str) -> Any:
    def normalize(text: Any) -> Optional[str]:
        if isinstance(text, (list, tuple)):
            for item in text:
                candidate = normalize(item)
                if candidate:
                    return candidate
            return None
        match = re.search(r"([1-4])", str(text))
        if match:
            return match.group(1)
        return None

    candidate = normalize(v)
    if candidate:
        return candidate
    candidate = normalize(m)
    if candidate:
        return candidate
    return v


def _class_norm_en12208(v: Any, m: str) -> Any:
    def normalize(text: Any) -> Optional[str]:
        if isinstance(text, (list, tuple)):
            for item in text:
                candidate = normalize(item)
                if candidate:
                    return candidate
            return None
        cleaned = re.sub(r"\s+", "", str(text).upper())
        match = re.search(r"([1-9])\s*([AB])", cleaned)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return None

    candidate = normalize(v)
    if candidate:
        return candidate
    candidate = normalize(m)
    if candidate:
        return candidate
    return v


def _class_norm_en12210(v: Any, m: str) -> Any:
    def normalize(text: Any) -> Optional[str]:
        if isinstance(text, (list, tuple)):
            for item in text:
                candidate = normalize(item)
                if candidate:
                    return candidate
            return None
        cleaned = re.sub(r"\s+", "", str(text).upper())
        match = re.search(r"([ABC])\s*([1-5])", cleaned)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return None

    candidate = normalize(v)
    if candidate:
        return candidate
    candidate = normalize(m)
    if candidate:
        return candidate
    return v


def _class_norm_en13501(v: Any, m: str) -> Any:
    def normalize(text: Any) -> Optional[str]:
        if isinstance(text, (list, tuple)):
            for item in text:
                candidate = normalize(item)
                if candidate:
                    return candidate
            return None
        cleaned = re.sub(r"\s+", "", str(text).upper())
        if not cleaned:
            return None
        main_match = re.match(r"(A1|A2|B|C|D|E|F)", cleaned)
        if not main_match:
            return None
        main = main_match.group(1)
        remainder = cleaned[len(main) :]
        s_match = re.search(r"S\s*-?\s*([0-3])", remainder)
        d_match = re.search(r"D\s*-?\s*([0-2])", remainder)
        parts: List[str] = []
        if s_match:
            parts.append(f"s{s_match.group(1)}")
        if d_match:
            parts.append(f"d{d_match.group(1)}")
        if parts:
            return f"{main}-" + ",".join(parts)
        remainder = remainder.strip("-_,")
        remainder = remainder.replace("-", ",").replace(";", ",").replace("_", ",")
        remainder = remainder.strip(",")
        if remainder:
            remainder = ",".join(filter(None, remainder.split(",")))
            remainder = remainder.lower()
            return f"{main}-{remainder}"
        return main

    candidate = normalize(v)
    if candidate:
        return candidate
    candidate = normalize(m)
    if candidate:
        return candidate
    return v


def _dims_to_mm_string(v: Any, m: str) -> Any:
    converter = _to_unit_factory("mm")

    def normalize_pair(value: Any, unit: Optional[str]) -> Optional[Any]:
        if value is None:
            return None
        payload = (value, unit or "") if unit else value
        converted = converter(payload, m)
        if isinstance(converted, (int, float)):
            return _coerce_numeric_output(float(converted))
        parsed = _parse_number(str(converted))
        if parsed is not None:
            return _coerce_numeric_output(parsed)
        return None

    if isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [v]

    values: List[Any] = []
    idx = 0
    while idx < len(items):
        value = items[idx]
        unit: Optional[str] = None
        next_idx = idx + 1
        if next_idx < len(items):
            candidate_unit = items[next_idx]
            if isinstance(candidate_unit, str) and candidate_unit.lower() in {"mm", "cm", "m"}:
                unit = candidate_unit
                idx += 2
            else:
                idx += 1
        else:
            idx += 1
        normalized = normalize_pair(value, unit)
        if normalized is not None:
            values.append(normalized)

    if len(values) >= 2:
        first, second = values[0], values[1]
        return f"{first}×{second}"
    return v


def _to_areal_density_kg_m2(value: Any, matched_text: str) -> Any:
    normalized = matched_text.lower()
    replacements = {
        "²": "2",
        "metri quadrati": "m2",
        "metri quadri": "m2",
        "metro quadrato": "m2",
        "al mq": "m2",
        "mq": "m2",
        "m q": "m2",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"\s+", " ", normalized)

    def factor_from_context() -> float:
        if re.search(r"\b(?:t|tonnellate?|ton)\b\s*/?\s*m2", normalized):
            return 1000.0
        if re.search(r"\b(?:g|gr|grammi|grammo)\b\s*/?\s*m2", normalized):
            return 0.001
        if re.search(r"\b(?:kg|chilogrammi|chilogrammo)\b\s*/?\s*m2", normalized):
            return 1.0
        # default to kilograms per square meter if unit is omitted
        return 1.0

    factor = factor_from_context()

    def convert(token: Any) -> Any:
        parsed = _parse_number(str(token))
        if parsed is None:
            return token
        converted = parsed * factor
        return _coerce_numeric_output(converted)

    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]
    return convert(value)


# ---------------------------------------------------------------------------
# Normalizers registry
# ---------------------------------------------------------------------------


BUILTIN_NORMALIZERS: Dict[str, Normalizer] = {
    "comma_to_dot": _comma_to_dot,
    "dot_to_comma": _dot_to_comma,
    "to_float": _to_float,
    "to_int": _to_int,
    "lower": _lower,
    "upper": _upper,
    "strip": _strip,
    "strip_trailing_punct": _strip_trailing_punct,
    "truncate_model_value": _truncate_model_value,
    "collapse_plus_sequences": _collapse_plus_sequences,
    "as_string": _as_string,
    "to_number": _to_number,
    "to_areal_density_kg_m2": _to_areal_density_kg_m2,
    "join_range": _join_range,
    "percent_to_ratio": _percent_to_ratio,
    "power_to_kw": _power_to_kw,
    "flow_to_m3h": _flow_to_m3h,
    "flow_to_l_s": _flow_to_l_s,
    "normalize_unit_symbols": _normalize_unit_symbols,
    "EI_from_any": _ei_from_any,
    "dims_join": _dims_join,
    "collect_many": _collect_many,
    "if_cm_to_mm": _if_cm_to_mm,
    "unique_list": _unique_list,
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
    "to_bool_strict": _to_bool_strict,
    "dims_to_mm_string": _dims_to_mm_string,
    "class_norm:EN12207": _class_norm_en12207,
    "class_norm:EN12208": _class_norm_en12208,
    "class_norm:EN12210": _class_norm_en12210,
    "class_norm:EN13501-1": _class_norm_en13501,
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
    if name.startswith("set_value:"):
            # payload JSON o stringa grezza
            raw = name.split(":", 1)[1]
            try:
                const = json.loads(raw)
            except Exception:
                const = raw
            return lambda v, m: const
    
    if name.startswith("map_enum:"):
        key = name.split(":", 1)[1]
        pack_normalizers = extractors_pack.get("normalizers", {})
        mapping = pack_normalizers.get(key) or pack_normalizers.get(name, {})
        return _map_enum_factory(mapping)

    if name.startswith("to_unit:"):
        target = name.split(":", 1)[1]
        return _to_unit_factory(target)

    if name.startswith("class_norm:") and name in BUILTIN_NORMALIZERS:
        return BUILTIN_NORMALIZERS[name]

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

