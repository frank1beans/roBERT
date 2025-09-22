
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------- Normalizers registry ----------

NormalizerFn = Callable[[Any, str], Any]
# value, matched_text -> new_value

def _comma_to_dot(v, m): 
    if isinstance(v, str): return v.replace(",", ".")
    if isinstance(v, list): return [str(x).replace(",", ".") for x in v]
    return v

def _dot_to_comma(v, m):
    if isinstance(v, str): return v.replace(".", ",")
    if isinstance(v, list): return [str(x).replace(".", ",") for x in v]
    return v

def _to_float(v, m):
    def conv(x):
        try: return float(str(x).replace(",", "."))
        except: return x
    if isinstance(v, list): return [conv(x) for x in v]
    return conv(v)

def _to_int(v, m):
    def conv(x):
        try: return int(re.sub(r"[^0-9-]", "", str(x)))
        except: return x
    if isinstance(v, list): return [conv(x) for x in v]
    return conv(v)

def _lower(v, m): return v.lower() if isinstance(v, str) else v
def _upper(v, m): return v.upper() if isinstance(v, str) else v
def _strip(v, m): return v.strip() if isinstance(v, str) else v

def _ei_from_any(v, m):
    s = " ".join(v) if isinstance(v, (list, tuple)) else str(v)
    mnum = re.search(r"(15|30|45|60|90|120|180|240)", s)
    return f"EI {mnum.group(1)}" if mnum else s

def _dims_join(v, m):
    # accept tuple/list like ('60','60') -> '60×60'
    if isinstance(v, (list, tuple)):
        if len(v) == 2: return f"{v[0]}×{v[1]}"
        # list of pairs
        if len(v) > 2 and all(isinstance(x, (list, tuple)) and len(x)==2 for x in v):
            return "; ".join([f"{a}×{b}" for a,b in v])
    return v

def _collect_many(v, m):
    # if list, keep list; else wrap into list
    if isinstance(v, list): return v
    return [v]

def _if_cm_to_mm(v, m):
    # multiply by 10 if match string contains 'cm' (not mm)
    def conv_one(x):
        try:
            val = float(str(x).replace(",", "."))
        except:
            return x
        s = m.lower()
        if "cm" in s and "mm" not in s:
            return round(val * 10.0, 3)
        return val
    if isinstance(v, list): return [conv_one(x) for x in v]
    return conv_one(v)

def _unique_list(v, m):
    if not isinstance(v, list): return v
    seen=set(); out=[]
    for x in v:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _map_enum_factory(mapping: Dict[str, str]):
    def _map_enum(v, m):
        def map_one(x):
            key = str(x).strip().lower()
            return mapping.get(key, x)
        if isinstance(v, list): return [map_one(x) for x in v]
        return map_one(v)
    return _map_enum

BUILTINS: Dict[str, NormalizerFn] = {
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
    # dynamic "map_enum:<name>" supported below
}

@dataclass
class Pattern:
    property_id: str
    regex: List[str]
    normalizers: List[str]

def _compile_patterns(extractors_pack: Dict, allowed_properties: Optional[Iterable[str]] = None) -> List[Pattern]:
    pats: List[Pattern] = []
    allowed: Optional[Set[str]] = None
    if allowed_properties is not None:
        allowed = {p for p in allowed_properties if p}
    for item in extractors_pack.get("patterns", []):
        pid = item["property_id"]
        if allowed is not None and pid not in allowed:
            continue
        pats.append(Pattern(
            property_id=pid,
            regex=item.get("regex", []),
            normalizers=item.get("normalizers", [])
        ))
    return pats

def _build_normalizer(name: str, extractors_pack: Dict) -> NormalizerFn:
    # allow "map_enum:<key>" to map values using pack.normalizers[<key>]
    if name.startswith("map_enum:"):
        key = name.split(":",1)[1]
        mapping = extractors_pack.get("normalizers", {}).get(key, {})
        return _map_enum_factory(mapping)
    fn = BUILTINS.get(name)
    if fn is None:
        # no-op unknown
        return lambda v,m: v
    return fn

def _apply_normalizers(value: Any, matched_text: str, normalizers: Sequence[str], extractors_pack: Dict) -> Any:
    v = value
    for n in normalizers:
        fn = _build_normalizer(n, extractors_pack)
        v = fn(v, matched_text)
    return v

def _coerce_capture(match: re.Match) -> Any:
    # No groups: return entire matched text
    if match.lastindex is None:
        return match.group(0)
    # 1 group: return that group
    if match.lastindex == 1:
        return match.group(1)
    # 2+ groups: return tuple of groups
    return tuple(match.group(i) for i in range(1, match.lastindex+1))

def extract_properties(text: str, extractors_pack: Dict, allowed_properties: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Applica i pattern in ordine; per default prende il primo match per proprietà.
    Se presente 'collect_many' nei normalizers, accumula tutti i match (deduplicabili con 'unique_list').
    'dims_join' trasforma tuple di due catture in 'AxB'.
    'if_cm_to_mm' converte cm→mm se nel match compare 'cm'.
    'map_enum:<name>' mappa valori verso enum/tabellati definiti nel pack.
    """
    out: Dict[str, Any] = {}
    pats = _compile_patterns(extractors_pack, allowed_properties=allowed_properties)
    # precompile all regex for speed
    compiled = [(p, [re.compile(rx, flags=re.IGNORECASE) for rx in p.regex]) for p in pats]
    for pat, regs in compiled:
        has_collect = "collect_many" in pat.normalizers
        for rx in regs:
            for m in rx.finditer(text):
                cap = _coerce_capture(m)
                # normalize immediate dims (tuples) before further steps
                val = cap
                # if dims_join later, keep as tuple until normalization
                val = _apply_normalizers(val, m.group(0), pat.normalizers, extractors_pack)
                if has_collect:
                    prev = out.get(pat.property_id, [])
                    if not isinstance(prev, list):
                        prev = [prev]
                    prev.append(val)
                    out[pat.property_id] = prev
                else:
                    # only first capture if not set yet
                    if pat.property_id not in out:
                        out[pat.property_id] = val
            # if property found and not collecting, skip remaining regexes
            if pat.property_id in out and not has_collect:
                break
        # post-process unique_list if needed
        if has_collect and pat.property_id in out and "unique_list" in pat.normalizers:
            out[pat.property_id] = BUILTINS["unique_list"](out[pat.property_id], "")
    return out

def dry_run(text: str, extractors_pack: Dict, allowed_properties: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Restituisce dettagli di matching per debug.
    """
    details = []
    pats = _compile_patterns(extractors_pack, allowed_properties=allowed_properties)
    for pat in pats:
        for rx in pat.regex:
            comp = re.compile(rx, re.IGNORECASE)
            for m in comp.finditer(text):
                details.append({
                    "property_id": pat.property_id,
                    "regex": rx,
                    "match": m.group(0),
                    "groups": m.groups()
                })
    return {"matches": details, "extracted": extract_properties(text, extractors_pack, allowed_properties=allowed_properties)}
