# src/robimb/extraction/dsl.py
from __future__ import annotations

import json
from collections.abc import Mapping, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACK_ROOT = _REPO_ROOT / "pack"
_LEGACY_DATA = _REPO_ROOT / "data" / "properties"
_DEFAULT_PACK_CANDIDATES = (
    _PACK_ROOT / "current" / "extractors.json",
    _PACK_ROOT / "current" / "extractors_extended.json",
    _PACK_ROOT / "v1" / "extractors.json",
    _LEGACY_DATA / "extractors_extended.json",
    _LEGACY_DATA / "extractors.json",
)

__all__ = ["PatternSpec", "PatternSpecs", "ExtractorsPack"]


# ---------------- Compat types (per import legacy) -----------------

@dataclass(frozen=True)
class PatternSpec:
    name: str
    pattern: str
    prop: str
    type: Optional[str] = None
    value: Any = None
    normalizer: Optional[str] = None

PatternSpecs = List[PatternSpec]


# ---------------- Helpers -----------------

def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_default_extractors() -> Dict[str, Any]:
    """Best-effort loader for the bundled regex extractors."""

    for candidate in _DEFAULT_PACK_CANDIDATES:
        if candidate.exists():
            payload = _load_json_if_exists(candidate)
            if isinstance(payload, dict):
                return payload
    return {"patterns": []}

def _rules_to_patterns(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converte form legacy 'rules' in items compatibili con engine.py (patterns[].regex, normalizers...)."""
    out: List[Dict[str, Any]] = []
    for r in rules:
        prop = r.get("prop") or r.get("property_id") or r.get("property")
        pattern = r.get("pattern")
        if not prop or not pattern:
            continue
        normals = r.get("normalizers") or r.get("normalizer") or r.get("type") or []
        normals = _ensure_list(normals)
        value = r.get("value", None)
        if value is not None:
            normals = list(normals) + [f"set_value:{json.dumps(value, ensure_ascii=False)}"]
        out.append({
            "property_id": prop,
            "regex": [pattern],
            "normalizers": normals,
            **({ "unit": r["unit"] } if "unit" in r else {}),
            **({ "tags": r["tags"] } if "tags" in r else {}),
            **({ "examples": r["examples"] } if "examples" in r else {}),
            **({ "first_wins": r["first_wins"] } if "first_wins" in r else {}),
            **({ "max_matches": r["max_matches"] } if "max_matches" in r else {}),
            **({ "confidence": r["confidence"] } if "confidence" in r else {}),
        })
    return out

def _registry_to_patterns(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converte registry {prop:{patterns:[...], normalizer/type, value_if_match}} in items patterns[]."""
    out: List[Dict[str, Any]] = []
    for prop, spec in registry.items():
        patterns = _ensure_list(spec.get("patterns"))
        normals: List[str] = []
        if spec.get("normalizer"):
            normals = _ensure_list(spec["normalizer"])
        elif spec.get("type"):
            normals = _ensure_list(spec["type"])
        value = spec.get("value_if_match", None)
        if value is not None:
            normals = list(normals) + [f"set_value:{json.dumps(value, ensure_ascii=False)}"]
        for rx in patterns:
            out.append({
                "property_id": prop,
                "regex": [rx],
                "normalizers": normals,
                **({ "examples": _ensure_list(spec["examples"]) } if spec.get("examples") else {}),
                **({ "tags": _ensure_list(spec["tags"]) } if spec.get("tags") else {}),
                **({ "first_wins": spec["first_wins"] } if "first_wins" in spec else {}),
                **({ "max_matches": spec["max_matches"] } if "max_matches" in spec else {}),
                **({ "confidence": spec["confidence"] } if "confidence" in spec else {}),
            })
    return out

def _normalize_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rende il pack conforme a engine.py: deve avere almeno 'patterns' (list)."""
    if not isinstance(payload, dict):
        return {"patterns": []}
    if isinstance(payload.get("patterns"), list):
        # moderno
        out = {
            "patterns": list(payload["patterns"]),
            "normalizers": dict(payload.get("normalizers", {})),
            "defaults": dict(payload.get("defaults", {})),
        }
        return out
    if isinstance(payload.get("rules"), list):
        # legacy
        patterns = _rules_to_patterns(payload["rules"])
        out = {
            "patterns": patterns,
            "normalizers": dict(payload.get("normalizers", {})),
            "defaults": dict(payload.get("defaults", {})),
        }
        return out
    if isinstance(payload.get("extractors"), dict):
        # annidato
        return _normalize_pack(payload["extractors"])
    return {"patterns": []}


# ---------------- ExtractorsPack (Mapping) -----------------

class ExtractorsPack(Mapping[str, Any]):
    """
    Wrapper dict-like su cui engine.py possa fare .get("patterns")/.get("normalizers").
    Permette di caricare extractors (moderni o legacy) e di **fonderli** con il registry.
    """

    def __init__(self, pack: Dict[str, Any]) -> None:
        self._pack: Dict[str, Any] = _normalize_pack(pack)

    # Mapping interface
    def __getitem__(self, key: str) -> Any:
        return self._pack[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._pack)

    def __len__(self) -> int:
        return len(self._pack)

    def get(self, key: str, default: Any = None) -> Any:
        return self._pack.get(key, default)

    @classmethod
    def from_files(
        cls,
        extractors_path: Optional[Path] = None,
        registry_path: Optional[Path] = None,
        validators_path: Optional[Path] = None,
        formulas_path: Optional[Path] = None,
        contexts_path: Optional[Path] = None,
    ) -> "ExtractorsPack":
        # 1) extractors
        payload = _load_json_if_exists(extractors_path)
        if payload is None:
            # fallback al default (già puntato a data/properties dal tuo resources.py)
            payload = _load_default_extractors()
        base = _normalize_pack(payload or {})

        # 2) registry → aggiungi patterns
        registry = _load_json_if_exists(registry_path)
        if isinstance(registry, dict) and registry:
            reg_patterns = _registry_to_patterns(registry)
            base["patterns"] = list(base.get("patterns", [])) + reg_patterns

        return cls(base)

    def build_property_schema(self) -> Dict[str, Any]:
        """Schema minimale (utile se vuoi attaccarlo alle righe)."""
        schema: Dict[str, Any] = {}
        for pat in self._pack.get("patterns", []):
            prop = pat.get("property_id")
            if not prop:
                continue
            entry = schema.setdefault(prop, {"patterns": [], "normalizers": []})
            entry["patterns"].extend(_ensure_list(pat.get("regex")))
            entry["normalizers"].extend(_ensure_list(pat.get("normalizers")))
        for v in schema.values():
            v["patterns"] = sorted(set(v["patterns"]))
            v["normalizers"] = sorted(set(v["normalizers"]))
        return schema
