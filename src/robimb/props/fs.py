from __future__ import annotations
import io, json, os
from pathlib import Path
from typing import Any, Mapping, Union, List

JsonLike = Union[Mapping[str, Any], List[Any], str, bytes, os.PathLike, io.TextIOBase, Path]

def load_json_any(obj: JsonLike):
    if isinstance(obj, (dict, list)):
        return obj
    if hasattr(obj, "read"):
        return json.load(obj)  # file-like
    if isinstance(obj, (Path, os.PathLike)):
        with open(obj, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    if isinstance(obj, bytes):
        try:
            return json.loads(obj.decode("utf-8"))
        except Exception:
            p = Path(obj.decode("utf-8"))
            if p.exists():
                with open(p, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            raise
    if isinstance(obj, str):
        # prova come JSON inline
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            p = Path(obj)
            if p.exists():
                with open(p, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
    raise TypeError(f"Impossibile caricare JSON da tipo: {type(obj).__name__}")
