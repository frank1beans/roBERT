
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

@dataclass
class Issue:
    code: str
    message: str
    severity: str = "error"  # "error" | "warn"
    rule_id: Optional[str] = None
    property_id: Optional[str] = None

def _to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def _exists(prop: str, props: Dict[str, Any]) -> bool:
    return prop in props and props[prop] not in (None, "", [])

def _get(prop: str, props: Dict[str, Any], default=None):
    return props.get(prop, default)

def _safe_eval(expr: str, env: Dict[str, Any]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:
        return False

def _match_rule_if(rule: Dict, category_label: str, context: Dict[str, Any], cat_entry: Optional[Dict]) -> bool:
    # support "when" blocks like: {"category": "..."} or {"type_profile":"masonry"} or {"regex": "muratura"}
    when = rule.get("when")
    if not when:
        return True
    ok = True
    if "category" in when:
        ok = ok and (str(when["category"]).lower() == str(category_label or "").lower())
    if "type_profile" in when:
        ok = ok and (cat_entry and cat_entry.get("type_profile") == when["type_profile"])
    if "regex" in when:
        import re
        ok = ok and bool(re.search(when["regex"], category_label or "", flags=re.IGNORECASE))
    # context filters: e.g. {"context": {"climate_zone_UNI10349": "E"}}
    if "context" in when:
        ctx = when["context"]
        for k,v in ctx.items():
            if (context or {}).get(k) != v:
                return False
    return ok

def validate(category_label: str, props: Dict[str, Any], context: Dict[str, Any], rules_pack: Dict, cat_entry: Optional[Dict]=None) -> List[Dict[str, Any]]:
    issues: List[Issue] = []
    rules = rules_pack.get("rules", [])

    for rule in rules:
        severity = rule.get("severity", "error")
        rule_id = rule.get("id")
        if not _match_rule_if(rule, category_label, context, cat_entry):
            continue

        # 1) requires (list of property ids)
        if "requires" in rule:
            for pid in rule["requires"]:
                if not _exists(pid, props):
                    issues.append(Issue(code="missing_required", severity=severity, rule_id=rule_id,
                                        property_id=pid, message=f"Proprietà obbligatoria assente: {pid}"))
        # 2) range check: {"prop":"geo.spessore_elemento","min":10,"max":2000}
        if "range" in rule:
            r = rule["range"]
            pid = r["prop"]
            if _exists(pid, props):
                v = _to_float(_get(pid, props))
                if v is None or (("min" in r and v < r["min"]) or ("max" in r and v > r["max"])):
                    issues.append(Issue(code="out_of_range", severity=severity, rule_id=rule_id,
                                        property_id=pid, message=rule.get("message", f"Valore fuori range per {pid}")))
        # 3) enum membership: {"prop":"opn.tenuta_aria_EN12207","in":["0","1","2","3","4"]}
        if "enum" in rule:
            e = rule["enum"]; pid = e["prop"]
            if _exists(pid, props):
                v = str(_get(pid, props))
                if v not in e.get("in", []):
                    issues.append(Issue(code="enum_invalid", severity=severity, rule_id=rule_id,
                                        property_id=pid, message=rule.get("message", f"Valore non ammesso per {pid}: {v}")))
        # 4) regex assert on a property: {"prop":"idn.descrizione","regex":"\bmuratura\b"}
        if "regex" in rule and isinstance(rule["regex"], dict):
            rr = rule["regex"]; pid = rr["prop"]
            if _exists(pid, props):
                import re
                if not re.search(rr["pattern"], str(_get(pid, props)), flags=re.IGNORECASE):
                    issues.append(Issue(code="regex_mismatch", severity=severity, rule_id=rule_id,
                                        property_id=pid, message=rule.get("message", f"Regex non soddisfatta per {pid}")))

        # 5) generic expression-based rule: { "if": "...", "assert": "...", "message": "..." }
        if "if" in rule and "assert" in rule:
            env = {
                "category": category_label,
                "context": type("Ctx", (), context or {})(),
                "exists": lambda p: _exists(p, props),
                "val": lambda p, d=None: _get(p, props, d),
                "num": lambda p: _to_float(_get(p, props)),
                "re": re
            }
            if _safe_eval(rule.get("if",""), env):
                if not _safe_eval(rule.get("assert",""), env):
                    issues.append(Issue(code="rule_failed", severity=severity, rule_id=rule_id,
                                        message=rule.get("message","Regola non soddisfatta")))

    return [i.__dict__ for i in issues]
