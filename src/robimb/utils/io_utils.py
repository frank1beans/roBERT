"""File-system helpers used across the CLI commands."""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ensure_has_weights"]


def ensure_has_weights(model_dir: str | os.PathLike[str]) -> None:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model directory {path} does not exist")
    has_weights = any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    if not has_weights:
        raise FileNotFoundError(
            f"{path} does not contain model weights. Expected model.safetensors or pytorch_model.bin"
        )
