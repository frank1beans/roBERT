"""Common data structures shared across extraction stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

__all__ = [
    "ExtractionCandidate",
    "StageResult",
    "ExtractionResult",
]


@dataclass
class ExtractionCandidate:
    """A single value produced by one stage of the extraction pipeline."""

    property_id: str
    value: Any
    confidence: Optional[float] = None
    stage: str = ""
    provenance: Optional[str] = None
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the candidate."""

        payload: Dict[str, Any] = {
            "property_id": self.property_id,
            "value": self.value,
            "stage": self.stage,
        }
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class StageResult:
    """Container tracking the candidates yielded by a single stage."""

    stage: str
    candidates: List[ExtractionCandidate] = field(default_factory=list)
    extra: Mapping[str, Any] | None = None

    def add(self, candidate: ExtractionCandidate) -> None:
        self.candidates.append(candidate)

    def by_property(self) -> Dict[str, List[ExtractionCandidate]]:
        mapping: Dict[str, List[ExtractionCandidate]] = {}
        for candidate in self.candidates:
            bucket = mapping.setdefault(candidate.property_id, [])
            bucket.append(candidate)
        return mapping


@dataclass
class ExtractionResult:
    """Aggregate view over all the stages executed by the router."""

    stages: List[StageResult] = field(default_factory=list)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self._priority = {stage.stage: idx for idx, stage in enumerate(self.stages)}

    def iter_candidates(self) -> Iterable[ExtractionCandidate]:
        for stage in self.stages:
            yield from stage.candidates

    def by_property(self) -> Dict[str, List[ExtractionCandidate]]:
        mapping: Dict[str, List[ExtractionCandidate]] = {}
        for candidate in self.iter_candidates():
            mapping.setdefault(candidate.property_id, []).append(candidate)
        return mapping

    def best_by_property(self) -> Dict[str, ExtractionCandidate]:
        """Return the preferred candidate for each property."""

        best: Dict[str, ExtractionCandidate] = {}
        priority = self._priority
        for property_id, candidates in self.by_property().items():
            sorted_candidates = sorted(
                candidates,
                key=lambda cand: (
                    priority.get(cand.stage, len(priority)),
                    -(cand.confidence if cand.confidence is not None else 0.0),
                ),
            )
            if sorted_candidates:
                best[property_id] = sorted_candidates[0]
        return best

    def as_value_mapping(self) -> Dict[str, Any]:
        """Expose the values of the preferred candidates."""

        return {prop: candidate.value for prop, candidate in self.best_by_property().items()}

