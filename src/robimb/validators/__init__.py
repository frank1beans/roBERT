"""Backward-compatible entry points for legacy validator imports."""
from __future__ import annotations

from ..registry import validate, Issue

__all__ = ["validate", "Issue"]
