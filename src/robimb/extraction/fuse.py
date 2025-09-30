"""Candidate fusion policies for property extraction."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict

__all__ = ["FusePolicy", "Candidate", "Fuser"]


LOGGER = logging.getLogger(__name__)


class FusePolicy(str, Enum):
    """Supported fusion strategies."""

    VALIDATE_THEN_MAX_CONF = "validate_then_max_conf"


class Candidate(TypedDict, total=False):
    """Representation of a candidate produced by an extractor."""

    value: Any
    source: Optional[str]
    raw: Optional[str]
    span: Optional[tuple[int, int] | list[int]]
    confidence: float
    unit: Optional[str]
    errors: List[str]


class Fuser:
    """Fuse property candidates according to a configurable policy."""

    def __init__(
        self,
        policy: FusePolicy = FusePolicy.VALIDATE_THEN_MAX_CONF,
        *,
        source_priority: Sequence[str] | None = None,
    ) -> None:
        self._policy = policy
        self._source_priority = list(source_priority or ("parser", "matcher", "qa_llm", "fuse", "manual"))
        self._priority_index: Dict[str, int] = {name: idx for idx, name in enumerate(self._source_priority)}

    def fuse(
        self,
        candidates: List[Candidate],
        validator: Callable[[Candidate], bool | tuple[bool, List[str]]],
    ) -> Candidate:
        """Fuse ``candidates`` using the configured ``policy``.

        Parameters
        ----------
        candidates:
            Ordered list of candidates proposed by upstream extractors.
        validator:
            Callable that evaluates candidate compliance. The callable may
            return a boolean or a ``(bool, list[str])`` tuple describing the
            validation outcome and associated error messages.

        Returns
        -------
        Candidate
            The selected candidate or an empty placeholder if none were valid.
        """

        if self._policy is not FusePolicy.VALIDATE_THEN_MAX_CONF:
            raise NotImplementedError(f"Policy {self._policy!s} is not implemented")

        valid_candidates: List[Candidate] = []

        for candidate in candidates:
            result = validator(candidate)
            if isinstance(result, tuple):
                is_valid, messages = result
            else:
                is_valid, messages = bool(result), []
            if not is_valid:
                errors = list(candidate.get("errors", []))
                if messages:
                    errors.extend(messages)
                else:
                    errors.append("validation_failed")
                candidate["errors"] = errors
                LOGGER.debug("candidate_rejected", extra={"candidate": candidate})
                continue
            candidate["errors"] = list(candidate.get("errors", []))
            valid_candidates.append(candidate)

        if not valid_candidates:
            return Candidate(
                value=None,
                source=None,
                raw=None,
                span=None,
                confidence=0.0,
                unit=None,
                errors=["no_valid_candidate"],
            )

        def _sort_key(item: Candidate) -> tuple[float, int, int]:
            confidence = float(item.get("confidence") or 0.0)
            source = item.get("source")
            priority = self._priority_index.get(source or "", len(self._priority_index))
            original_index = candidates.index(item)
            return (-confidence, priority, original_index)

        winner = min(valid_candidates, key=_sort_key)
        LOGGER.debug("candidate_selected", extra={"candidate": winner})
        return winner
