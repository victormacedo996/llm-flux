"""
datasets/port.py — Port: Dataset loading abstraction.

Both HuggingFace Hub datasets and local datasets are loaded through
this interface, keeping the rest of the framework dataset-source-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, model_validator


class DatasetConfig(BaseModel):
    """
    Validated descriptor for a dataset source.

    - If ``source`` is a path that exists on disk → ``LocalDatasetAdapter`` should be used.
    - Otherwise → treated as a HuggingFace Hub dataset name.
    """

    source: str  # HF dataset name OR local directory / file path
    split: str = "train"
    subset: str | None = None  # HF dataset config name (e.g. "ARC-Challenge")
    max_samples: int | None = None  # None = use all available; otherwise first N rows
    streaming: bool = False
    seed: int = 42  # kept for backward compatibility (not used by first-N mode)

    @model_validator(mode="after")
    def _max_samples_positive(self) -> DatasetConfig:
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be a positive integer, got {self.max_samples}")
        return self


class DatasetPort(ABC):
    """
    Port: anything that can produce a dataset object.

    The returned object must be compatible with ``transformers.Trainer``
    (i.e. it should behave like a ``torch.utils.data.Dataset`` or a
    HuggingFace ``datasets.Dataset``).
    """

    config: DatasetConfig

    @abstractmethod
    def load(self) -> object:
        """Return a dataset object ready for use by a Trainer."""
