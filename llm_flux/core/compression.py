"""
core/compression.py — Port: Compression algorithm abstraction.

Add a new compression technique by subclassing CompressionPort and
implementing `compress()`. No existing code needs to change (Open/Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from llm_flux.core.model import ModelHandle


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

    def __init__(self, config: CompressionConfig):
        self.config = config

    @abstractmethod
    def compress(self, model_handle: ModelHandle) -> ModelHandle:
        """
        Apply compression and return a CompressedModelHandle.

        The adapter MUST save the compressed model to disk and return a
        CompressedModelHandle pointing to that saved artifact. This enables:
          - Loading the compressed model independently later
          - Using the compressed model as student in knowledge distillation

        Args:
            model_handle: ModelHandle of the uncompressed model (as loaded
                by a preceding LOAD step in the pipeline).

        Returns:
            CompressedModelHandle wrapping the saved compressed model.
            The compressed model is saved to disk at the path in
            CompressedModelHandle.source.identifier.

        Raises:
            CompressionNotSupportedError: If this technique cannot handle
                the model (wrong architecture, missing deps, etc.).
        """
        raise NotImplementedError("Implement this method to apply compression to the model.")

    @property
    def label(self) -> str:
        """Short label for DAG nodes and log lines."""
        return self.config.name

    def should_reload_after_compress(self) -> bool:
        """
        Whether the compressed model should be reloaded from disk before use.

        Override in adapters that modify model structure in-place (e.g., DepthPruning)
        where internal state needs to be reinitialized via save+load cycle.
        Default: False (most compression adapters don't need this).
        """
        return False
