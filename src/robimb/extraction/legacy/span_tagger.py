"""Stage R1: interface for span tagger or classifier based property inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .formats import ExtractionCandidate, StageResult

__all__ = ["SpanTagger", "SpanTaggerOutput", "build_stage"]


PredictionLike = Mapping[str, Any]


@dataclass
class SpanTaggerOutput:
    """Raw prediction returned by a downstream span tagger."""

    property_id: str
    value: Any
    confidence: Optional[float] = None
    provenance: Optional[str] = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: PredictionLike) -> "SpanTaggerOutput":
        return cls(
            property_id=str(payload.get("property_id")),
            value=payload.get("value"),
            confidence=payload.get("confidence"),
            provenance=payload.get("provenance"),
            metadata=payload.get("metadata"),
        )


def build_stage(predictions: Iterable[PredictionLike]) -> StageResult:
    """Convert raw predictions to a :class:`StageResult`."""

    stage = StageResult(stage="R1")
    for payload in predictions:
        candidate = SpanTaggerOutput.from_payload(payload)
        if not candidate.property_id:
            continue
        stage.add(
            ExtractionCandidate(
                property_id=candidate.property_id,
                value=candidate.value,
                confidence=(float(candidate.confidence) if candidate.confidence is not None else None),
                stage="R1",
                provenance=candidate.provenance or "span_tagger",
                metadata=candidate.metadata,
            )
        )
    return stage


class SpanTagger:
    """Lightweight adapter wrapping any callable exposing span predictions."""

    def __init__(
        self,
        predictor: Callable[[str, Optional[Sequence[str]]], Iterable[PredictionLike]],
    ) -> None:
        self._predictor = predictor

    def __call__(
        self,
        text: str,
        *,
        allowed_properties: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> StageResult:
        predictions = self._predictor(text, allowed_properties)
        filtered: Iterable[PredictionLike]
        if allowed_properties:
            allowed = {prop for prop in allowed_properties}

            def _filter(payload: PredictionLike) -> bool:
                pid = str(payload.get("property_id"))
                return pid in allowed if pid else False

            filtered = (payload for payload in predictions if _filter(payload))
        else:
            filtered = predictions
        return build_stage(filtered)

