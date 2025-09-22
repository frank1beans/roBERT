
from __future__ import annotations
from typing import Dict, Any
import re

def _resolve(template: str, props: Dict[str, Any], fallbacks: Dict[str, str] | None = None) -> str:
    fallbacks = fallbacks or {}
    def repl(m):
        key = m.group(1)
        if key in props and props[key] not in (None, "", []):
            return str(props[key])
        return str(fallbacks.get(key, ""))
    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", repl, template)

def render(category_label: str, props: Dict[str, Any], templates_pack: Dict) -> str:
    # pick first matching category template, fallback to generic
    chosen = None
    generic = None
    for t in templates_pack.get("templates", []):
        if t.get("category","").lower() == (category_label or "").lower():
            chosen = t; break
        if t.get("category","").lower() in ("*", "generic"):
            generic = t
    tpl = (chosen or generic or {}).get("template", category_label or "")
    fall = (chosen or generic or {}).get("fallbacks", {})
    # Support both {{prop.id}} and {prop.id} variants
    if "{{" not in tpl and "}}" not in tpl:
        # convert {a.b} -> {{a.b}} to reuse the same resolver
        tpl = re.sub(r"\{([^}{]+)\}", r"{{\1}}", tpl)
    return _resolve(tpl, props, fall)
