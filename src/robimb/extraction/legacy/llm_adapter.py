"""Rigid JSON adapter used by the router to interface LLM stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .formats import ExtractionCandidate, StageResult

__all__ = ["SchemaField", "StructuredLLMAdapter"]


@dataclass
class SchemaField:
    """Definition of a single slot expected from the language model."""

    type: str
    description: Optional[str] = None

    def render(self, name: str) -> str:
        desc = f" - {self.description}" if self.description else ""
        return f"- {name}: {self.type}{desc}"


class StructuredLLMAdapter:
    """Compose prompts and parse JSON adhering to the provided schema."""

    def __init__(
        self,
        schema: Mapping[str, SchemaField | Mapping[str, Any] | str],
        *,
        stage: str = "L0",
    ) -> None:
        self.stage = stage
        self.schema: Dict[str, SchemaField] = {}
        for name, spec in schema.items():
            if isinstance(spec, SchemaField):
                self.schema[name] = spec
            elif isinstance(spec, Mapping):
                self.schema[name] = SchemaField(
                    type=str(spec.get("type", "string")),
                    description=(str(spec.get("description")) if spec.get("description") else None),
                )
            else:
                self.schema[name] = SchemaField(type=str(spec))

    def build_prompt(self, text: str, *, extra_instructions: Optional[str] = None) -> str:
        header = (
            "Estrai le seguenti proprietà dal testo fornito. "
            "Rispondi esclusivamente con un JSON valido. "
            "Ogni chiave deve restituire un oggetto con campi 'value' e opzionalmente 'confidence' (0-1)."
        )
        lines = [header]
        if extra_instructions:
            lines.append(extra_instructions)
        lines.append("Schema previsto:")
        for name, field in self.schema.items():
            lines.append(field.render(name))
        lines.append("\nTesto:\n" + text.strip())
        lines.append(
            "\nOutput JSON di esempio: {\"property\": {\"value\": \"...\", \"confidence\": 0.7}}"
        )
        return "\n".join(lines)

    def _parse_payload(self, payload: Any, property_id: str) -> tuple[Any, Optional[float]]:
        if isinstance(payload, Mapping):
            value = payload.get("value")
            confidence = payload.get("confidence")
        else:
            value = payload
            confidence = None
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Confidenza non valida per {property_id}: {confidence}") from exc
            confidence = max(0.0, min(1.0, confidence))
        return value, confidence

    def parse_response(self, response: str) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("Risposta LLM non è un JSON valido") from exc
        if not isinstance(data, Mapping):
            raise ValueError("La risposta dell'LLM deve essere un oggetto JSON")

        parsed: Dict[str, Dict[str, Any]] = {}
        for name in self.schema:
            if name not in data:
                continue
            value, confidence = self._parse_payload(data[name], name)
            parsed[name] = {"value": value, "confidence": confidence}
        return parsed

    def build_stage(self, response: str) -> StageResult:
        parsed = self.parse_response(response)
        stage = StageResult(stage=self.stage)
        for property_id, payload in parsed.items():
            stage.add(
                ExtractionCandidate(
                    property_id=property_id,
                    value=payload.get("value"),
                    confidence=payload.get("confidence"),
                    stage=self.stage,
                    provenance="llm:structured",
                    metadata={"source": "llm"},
                )
            )
        return stage

