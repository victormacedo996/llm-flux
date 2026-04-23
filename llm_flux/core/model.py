"""
core/model.py — Port: Model loading abstraction.

ModelHandle is the single interface through which the pipeline interacts
with any ML model, regardless of where it comes from (HuggingFace hub or
a local directory). Concrete implementations live in adapters/model/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
    def _validate_identifier(self) -> ModelSource:
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

    def __init__(
        self, model_source: ModelSource | None = None, model_instance: Any | None = None
    ) -> None:
        self.model_source = model_source
        self.model_instance = model_instance
        self.is_model_loaded = False

    @abstractmethod
    def load(self) -> object:
        """
        Load and return the raw model object (e.g. a ``torch.nn.Module``).
        Should move the model to the appropriate device before returning.
        """
        raise NotImplementedError("load() must be implemented by subclasses of ModelHandle.")

    @abstractmethod
    def unload(self) -> None:
        """Release resources (GPU memory, file handles, etc.)."""
        raise NotImplementedError("unload() must be implemented by subclasses of ModelHandle.")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label used in DAG node names and logs."""
        raise NotImplementedError("name() must be implemented by subclasses of ModelHandle.")

    @abstractmethod
    def get_tokenizer(self) -> object:
        """Return the tokenizer associated with this model, if applicable."""
        raise NotImplementedError(
            "get_tokenizer() must be implemented by subclasses of ModelHandle."
        )

    @abstractmethod
    def get_model_instance(self) -> Any:
        """Return the loaded model instance, if already loaded."""
        raise NotImplementedError(
            "get_model_instance() must be implemented by subclasses of ModelHandle."
        )

    @abstractmethod
    def get_is_model_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory."""
        return self.is_model_loaded


class CompressedModelHandle(ModelHandle):
    """
    ModelHandle wrapping a compressed model artifact.

    Returned by compression adapters. Points to a saved compressed model
    on disk while keeping a reference to the original uncompressed handle
    for provenance (teacher model in knowledge distillation scenarios).

    Args:
        original_handle: The uncompressed ModelHandle used as source.
        compressed_path: Path to the saved compressed model directory.
        compressed_model: The loaded compressed model instance.
        tokenizer: Tokenizer instance (reused from original if not provided).
    """

    def __init__(
        self,
        original_handle: ModelHandle,
        compressed_path: str,
        compressed_model: Any,
        tokenizer: Any | None = None,
    ) -> None:
        self.original_handle = original_handle
        self._source = ModelSource(identifier=compressed_path)
        self._model = compressed_model
        self._tokenizer = tokenizer or (
            original_handle.get_tokenizer() if original_handle.get_is_model_loaded() else None
        )

    @property
    def name(self) -> str:
        return f"{self.original_handle.name} (compressed)"

    def load(self) -> object:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self._source.identifier,
            trust_remote_code=self.original_handle.source.trust_remote_code,
            device_map="auto",
        )
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._source.identifier,
                trust_remote_code=self.original_handle.source.trust_remote_code,
            )
        self.is_model_loaded = True
        return self._model

    def unload(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_model_loaded = False

    def get_tokenizer(self) -> object:
        if self._tokenizer is None:
            raise RuntimeError("Call load() or provide tokenizer to access the tokenizer.")
        return self._tokenizer

    def get_model_instance(self) -> Any:
        if self._model is None:
            raise RuntimeError("Call load() before accessing the model instance.")
        return self._model

    def get_is_model_loaded(self) -> bool:
        return self._model is not None
