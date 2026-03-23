"""
core/healing.py — Port: Model healing (fine-tuning) abstraction.

HealingPort provides a single interface for any training strategy.
The default adapter (adapters/healing/hf_trainer.py) covers HF Trainer +
LoRA/QLoRA out of the box.  Users can inject a custom trainer_class or
subclass HealingPort entirely for non-HF frameworks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from llm_flux.datasets.port import DatasetConfig


# ── LoRA config ───────────────────────────────────────────────────────────────


class LoRAConfig(BaseModel):
    """
    PEFT LoRA hyper-parameters.
    Set ``healing.lora = None`` to perform full fine-tuning instead.
    """

    r: int = 8
    lora_alpha: int = 32
    target_modules: list[str] = ["q_proj", "v_proj"]
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ── Main config ───────────────────────────────────────────────────────────────


class HealingConfig(BaseModel):
    """
    Declarative training configuration for model healing.

    Provides sane defaults so a minimal setup only needs ``dataset``.
    Advanced users can replace the ``trainer_class`` with any object
    that has the same constructor signature as ``transformers.Trainer``.
    """

    name: str = "model-healing"
    description: str = ""

    # Dataset ─────────────────────────────────────────────────────────────────
    dataset: DatasetConfig

    # Training knobs ──────────────────────────────────────────────────────────
    max_steps: int = 500
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    output_dir: str = "./healing_output"
    save_steps: int = 100
    logging_steps: int = 10
    max_seq_length: int | None = None

    # Optional PEFT / LoRA — None → full fine-tune
    lora: LoRAConfig | None = None

    # Advanced: supply your own Trainer-compatible class.
    # Excluded from serialization because classes aren't JSON-serialisable.
    trainer_class: Any | None = Field(None, exclude=True)


# ── Port ──────────────────────────────────────────────────────────────────────


class HealingPort(ABC):
    """
    Port: a training / healing strategy.

    Implement this to support any training framework (HF, Lightning, raw
    PyTorch loops, etc.) while keeping the pipeline executor framework-agnostic.
    """

    config: HealingConfig

    @abstractmethod
    def heal(self, model: Any) -> Any:
        """
        Fine-tune / heal ``model`` and return the healed version.

        The returned object replaces the model in the DAG's running state,
        so subsequent steps (profiling, further compression) receive the
        healed model automatically.
        """

    @property
    def label(self) -> str:
        return self.config.name
