"""Compatibility layer exposing the router from the classic inference API."""

from __future__ import annotations

from typing import Any, Mapping

from ..extraction import ExtractionRouter

__all__ = ["predict_properties"]


def predict_properties(text: str, pack: Any, categories: Any) -> Mapping[str, Any]:
    router = ExtractionRouter(pack)
    output = router.extract(text, categories=categories)
    return output.postprocess.values

