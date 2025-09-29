"""Validation helpers migrated from :mod:`robimb.validators.engine`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import re

__all__ = ["Issue", "validate"]


@dataclass
class Issue:
    code: str
    message: str
    severity: str = "error"
    rule_id: Optional[str] = None
    property_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "property_id": self.property_id,
        }


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", "."))
    except Exception:  # pragma: no cover - defensive guard
        return None


def _exists(prop: str, props: Mapping[str, Any]) -> bool:
    return prop in props and props[prop] not in (None, "", [])


def _get(prop: str, props: Mapping[str, Any], default: Any = None) -> Any:
    return props.get(prop, default)


def _safe_eval(expr: str, env: Dict[str, Any]) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, env))
    except Exception:  # pragma: no cover - defensive guard
        return False


def _match_rule_if(
    rule: Mapping[str, Any],
    category_label: str,
    context: Mapping[str, Any],
    cat_entry: Optional[Mapping[str, Any]],
) -> bool:
    when = rule.get("when")
    if not when:
        return True
    ok = True
    if "category" in when:
        ok = ok and (str(when["category"]).lower() == str(category_label or "").lower())
    if "type_profile" in when:
        ok = ok and (cat_entry and cat_entry.get("type_profile") == when["type_profile"])
    if "regex" in when:
        ok = ok and bool(re.search(str(when["regex"]), category_label or "", flags=re.IGNORECASE))
    if "context" in when and isinstance(when["context"], Mapping):
        for key, value in when["context"].items():
            if context.get(key) != value:
                return False
    return ok


def validate(
    category_label: str,
    props: Mapping[str, Any],
    context: Mapping[str, Any],
    rules_pack: Mapping[str, Any],
    *,
    cat_entry: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    issues: List[Issue] = []
    rules = rules_pack.get("rules", []) if isinstance(rules_pack, Mapping) else []

    for rule in rules:
        if not isinstance(rule, Mapping):
            continue
        severity = str(rule.get("severity", "error"))
        rule_id = rule.get("id")
        if not _match_rule_if(rule, category_label, context, cat_entry):
            continue

        if "requires" in rule and isinstance(rule["requires"], Iterable):
            for pid in rule["requires"]:
                if not _exists(str(pid), props):
                    issues.append(
                        Issue(
                            code="missing_required",
                            severity=severity,
                            rule_id=rule_id,
                            property_id=str(pid),
                            message=f"Proprietà obbligatoria assente: {pid}",
                        )
                    )

        if "range" in rule and isinstance(rule["range"], Mapping):
            r = rule["range"]
            pid = str(r.get("prop"))
            if _exists(pid, props):
                value = _to_float(_get(pid, props))
                if value is None or (
                    ("min" in r and value < r["min"]) or ("max" in r and value > r["max"])
                ):
                    issues.append(
                        Issue(
                            code="out_of_range",
                            severity=severity,
                            rule_id=rule_id,
                            property_id=pid,
                            message=rule.get("message", f"Valore fuori range per {pid}"),
                        )
                    )

        if "enum" in rule and isinstance(rule["enum"], Mapping):
            spec = rule["enum"]
            pid = str(spec.get("prop"))
            if _exists(pid, props):
                value = str(_get(pid, props))
                allowed = [str(item) for item in spec.get("in", [])]
                if value not in allowed:
                    issues.append(
                        Issue(
                            code="enum_invalid",
                            severity=severity,
                            rule_id=rule_id,
                            property_id=pid,
                            message=rule.get("message", f"Valore non ammesso per {pid}: {value}"),
                        )
                    )

        if "regex" in rule and isinstance(rule["regex"], Mapping):
            spec = rule["regex"]
            pid = str(spec.get("prop"))
            if _exists(pid, props):
                pattern = str(spec.get("pattern"))
                if not re.search(pattern, str(_get(pid, props)), flags=re.IGNORECASE):
                    issues.append(
                        Issue(
                            code="regex_mismatch",
                            severity=severity,
                            rule_id=rule_id,
                            property_id=pid,
                            message=rule.get("message", f"Regex non soddisfatta per {pid}"),
                        )
                    )

        if "if" in rule and "assert" in rule:
            env = {
                "category": category_label,
                "context": type("Ctx", (), dict(context or {}))(),
                "exists": lambda prop_id: _exists(prop_id, props),
                "val": lambda prop_id, default=None: _get(prop_id, props, default),
                "num": lambda prop_id: _to_float(_get(prop_id, props)),
                "re": re,
            }
            if _safe_eval(str(rule.get("if", "")), env) and not _safe_eval(str(rule.get("assert", "")), env):
                issues.append(
                    Issue(
                        code="rule_failed",
                        severity=severity,
                        rule_id=rule_id,
                        message=rule.get("message", "Regola non soddisfatta"),
                    )
                )

    return [issue.as_dict() for issue in issues]
