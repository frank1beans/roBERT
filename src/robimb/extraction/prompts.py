"""Lightweight prompt templating utilities for the hybrid extractor."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping
import re

from ..config import get_settings

__all__ = ["PromptTemplate", "PromptLibrary", "load_prompt_library"]

_PLACEHOLDER_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt template with metadata and a render helper."""

    name: str
    description: str
    template: str

    def render(self, **values: object) -> str:
        """Render the template replacing ``{{placeholder}}`` expressions."""

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise KeyError(f"Missing placeholder '{key}' for prompt '{self.name}'")
            return str(values[key])

        return _PLACEHOLDER_PATTERN.sub(_replace, self.template)


class PromptLibrary:
    """Container mapping prompt identifiers to templates."""

    def __init__(self, templates: Mapping[str, PromptTemplate]) -> None:
        self._templates: Dict[str, PromptTemplate] = dict(templates)

    def render(self, name: str, **values: object) -> str:
        template = self._templates.get(name)
        if not template:
            raise KeyError(f"Prompt '{name}' is not registered")
        return template.render(**values)

    def template(self, name: str) -> PromptTemplate:
        template = self._templates.get(name)
        if not template:
            raise KeyError(f"Prompt '{name}' is not registered")
        return template

    @classmethod
    def from_path(cls, path: str | Path) -> "PromptLibrary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        templates: Dict[str, PromptTemplate] = {}
        for name, data in payload.items():
            description = data.get("description", "")
            template = data.get("template")
            if not isinstance(template, str):
                raise ValueError(f"Template '{name}' must define a string 'template' field")
            templates[name] = PromptTemplate(name=name, description=description, template=template)
        return cls(templates)

    @classmethod
    def default(cls) -> "PromptLibrary":
        path = get_settings().prompts_path
        if not path.exists():
            raise FileNotFoundError(f"Prompt file '{path}' is missing")
        return cls.from_path(path)


def load_prompt_library(path: str | Path | None = None) -> PromptLibrary:
    """Load the prompt library from ``path`` or use the default location."""

    if path is None:
        return PromptLibrary.default()
    return PromptLibrary.from_path(path)
