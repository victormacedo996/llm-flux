"""
core/compression.py — Port: Compression algorithm abstraction.

Add a new compression technique by subclassing CompressionPort and
implementing `compress()`. No existing code needs to change (Open/Closed).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class CompressionConfig(BaseModel):
    """
    Base configuration shared by all compression algorithms.

    Subclass this to declare algorithm-specific hyper-parameters while
    getting Pydantic validation for free.

    Example::

        class GPTQConfig(CompressionConfig):
            bits: int = Field(4, ge=2, le=8)
            dataset_name: str = "c4"
    """

    name: str
    description: str = ""


class CompressionNotSupportedError(Exception):
    """
    Raised when a compression technique is incompatible with the
    received model architecture.

    The pipeline executor catches this and logs it descriptively so the
    researcher knows exactly which step failed and why.
    """


class CompressionPort(ABC):
    """
    Port: a single compression algorithm.

    Implementors must:
    - Store their config in ``self.config``.
    - Raise ``CompressionNotSupportedError`` when the model cannot be compressed
      by this technique (e.g. wrong architecture, missing quantization targets).
    """

    config: CompressionConfig

    @abstractmethod
    def compress(self, model: Any) -> Any:
        """
        Apply compression and return the (possibly new) model object.

        Args:
            model: Raw model object returned by a previous ``ModelHandle.load()``
                   or ``CompressionPort.compress()`` call.

        Returns:
            The compressed model object.  May be the same object mutated
            in-place or a brand-new object, depending on the algorithm.

        Raises:
            CompressionNotSupportedError: If this technique cannot handle the model.
        """

    @property
    def label(self) -> str:
        """Short label for DAG nodes and log lines."""
        return self.config.name
