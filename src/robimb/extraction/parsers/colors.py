"""Parser for RAL colour codes using a curated lexicon."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from ...config import get_settings

__all__ = ["RALColor", "load_ral_lexicon", "parse_ral_colors"]

_RAL_PATTERN = re.compile(r"RAL\s?(\d{4})", re.IGNORECASE)


@dataclass(frozen=True)
class RALColor:
    code: str
    name: Optional[str]
    span: tuple[int, int]


def load_ral_lexicon(path: str | Path | None = None) -> Dict[str, str]:
    lexicon_path = Path(path) if path is not None else get_settings().colors_ral
    if not lexicon_path.exists():
        return {}
    data = json.loads(lexicon_path.read_text(encoding="utf-8"))
    return {key.upper(): value for key, value in data.items()}


def parse_ral_colors(text: str, lexicon: Optional[Dict[str, str]] = None) -> Iterable[RALColor]:
    lex = lexicon or load_ral_lexicon()
    for match in _RAL_PATTERN.finditer(text):
        code = f"RAL {match.group(1)}".upper()
        yield RALColor(code=code, name=lex.get(code), span=(match.start(), match.end()))
