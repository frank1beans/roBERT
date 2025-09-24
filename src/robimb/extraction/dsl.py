import re
from typing import Dict, Any, List

class AutoPack:
    def __init__(self, registry: Dict[str, Any]):
        self.registry = registry
        self.compiled = []
        for prop, spec in registry.items():
            pats = spec.get("patterns", [])
            for p in pats:
                self.compiled.append((prop, re.compile(p, re.IGNORECASE), spec))

    @staticmethod
    def _to_mm(val: str) -> int:
        # “10 cm” -> 100; “75” (mm) -> 75 (se hai già mm)
        v = int(val)
        return v * 10 if v < 50 else v  # euristica semplice

    def apply_all(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for prop, rx, spec in self.compiled:
            m = rx.search(text)
            if not m:
                continue
            if "value_if_match" in spec:
                out[prop] = spec["value_if_match"]
                continue
            if m.groups():
                val = m.group(1)
                tp = spec.get("type")
                norm = spec.get("normalizer")
                if tp == "int":
                    val = int(val)
                if norm == "to_mm":
                    val = self._to_mm(str(val))
                out[prop] = val
        return out

def build_property_schema_from_registry(reg: Dict[str, Any]) -> Dict[str, Any]:
    # schema minimale
    schema = {}
    for k, spec in reg.items():
        schema[k] = {
            "type": spec.get("type", "string"),
            "patterns": spec.get("patterns", []),
        }
    return schema
