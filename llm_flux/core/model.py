"""
core/model.py — Port: Model loading abstraction.

ModelHandle is the single interface through which the pipeline interacts
with any ML model, regardless of where it comes from (HuggingFace hub or
a local directory). Concrete implementations live in adapters/model/.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, model_validator


class ModelSource(BaseModel):
    """
    Validated descriptor for a model source.

    - If `identifier` is a path that exists on disk → treated as a local model directory.
    - Otherwise → treated as a HuggingFace Hub repo id (e.g. "meta-llama/Meta-Llama-3-8B").
    """

    identifier: str
    revision: str = "main"
    trust_remote_code: bool = False
    cache_dir: str | None = None

    @model_validator(mode="after")
    def _validate_identifier(self) -> "ModelSource":
        path = Path(self.identifier)
        if not path.exists():
            # Basic sanity-check for a Hub id: must contain "/" or be a valid identifier
            if "/" not in self.identifier:
                raise ValueError(
                    f"'{self.identifier}' is neither an existing local path "
                    "nor a valid HuggingFace Hub id (expected 'org/repo' format)."
                )
        return self

    @property
    def is_local(self) -> bool:
        return Path(self.identifier).exists()


class ModelHandle(ABC):
    """
    Port: anything that can produce a loadable model artifact.

    The pipeline executor always receives a ModelHandle, never a raw model
    object directly. This keeps the DAG builder free of ML library imports.
    """

    source: ModelSource

    @abstractmethod
    def load(self) -> object:
        """
        Load and return the raw model object (e.g. a ``torch.nn.Module``).
        Should move the model to the appropriate device before returning.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release resources (GPU memory, file handles, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label used in DAG node names and logs."""
