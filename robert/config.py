"""Configuration dataclasses shared across pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ModelConfig:
    """Generic configuration for Hugging Face models used in the project."""

    name_or_path: str
    tokenizer_name: Optional[str] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    nd_label_id: Optional[int] = None
    use_mean_pool: Optional[bool] = None
    proj_dim: Optional[int] = None
    arcface: bool = True


@dataclass(slots=True)
class TrainingConfig:
    """Common knobs shared by the supervised trainers."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate_head: float = 2e-4
    learning_rate_encoder: float = 1e-5
    weight_decay: float = 0.0
    max_length: int = 256
    gradient_accumulation_steps: int = 1
    seed: int = 42
    output_dir: Path = Path("outputs")
    save_total_limit: int = 1
    fp16: bool = False
    push_to_hub: bool = False
    hub_repo: Optional[str] = None


@dataclass(slots=True)
class InferenceConfig:
    """Configuration for the offline inference pipeline."""

    masked_model_path: Path
    label_model_path: Optional[Path] = None
    ontology_path: Optional[Path] = None
    label_maps_path: Optional[Path] = None
    properties_registry_path: Optional[Path] = None
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 256


@dataclass(slots=True)
class PipelineConfig:
    """High level configuration used to build pipelines programmatically."""

    model: ModelConfig
    training: TrainingConfig
    extra_args: Dict[str, Any] = field(default_factory=dict)
    label_texts_super: Optional[List[str]] = None
    label_texts_cat: Optional[List[str]] = None

    def merged_training_args(self) -> Dict[str, Any]:
        """Return a dictionary with standard training arguments."""

        data = {
            "num_train_epochs": self.training.epochs,
            "per_device_train_batch_size": self.training.batch_size,
            "per_device_eval_batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate_head": self.training.learning_rate_head,
            "learning_rate_encoder": self.training.learning_rate_encoder,
            "weight_decay": self.training.weight_decay,
            "max_length": self.training.max_length,
            "seed": self.training.seed,
            "output_dir": str(self.training.output_dir),
            "save_total_limit": self.training.save_total_limit,
            "fp16": self.training.fp16,
            "push_to_hub": self.training.push_to_hub,
            "hub_repo_id": self.training.hub_repo,
        }
        data.update(self.extra_args)
        return data
